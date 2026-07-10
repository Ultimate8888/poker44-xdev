"""
Train xdev-trainer-v2: HistGBM + IsotonicRegression on top-60 features,
extracted from validator-PROJECTED payloads, within-batch normalized.

Key insight vs v1: the validator projects every hand through
poker44.validator.payload_view.prepare_hand_for_miner (bet-size bucketing with
deterministic noise, 5-8 sampled actions per hand, seat aliasing, zeroed
outcomes) before querying miners. Training on raw benchmark hands therefore
creates a train/serve mismatch; v2 projects all training data through the same
canonicalizer first. The XDEV_FEATURE_NAMES top-60 list in xdev/features.py was
selected on projected data by LightGBM gain (0.6) + |Cohen's d| (0.4).

Usage:
  1) Fetch sessions:
     python /root/work/Poker44-subnet/scripts/miner/train/build_sequences.py \
         --out <sessions.pkl> --discover-limit 120 --per-date-limit 500
  2) python scripts/train.py <sessions.pkl> <out_model.joblib>
"""
import sys, pickle, random, time
sys.path.insert(0, "/root/work/Poker44-subnet")
sys.path.insert(0, "/root/work/poker44-xdev")

import numpy as np, warnings
warnings.filterwarnings("ignore")

from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward
from xdev.features import XDEV_FEATURE_NAMES, N_XDEV_FEATURES, extract_xdev_features
from xdev.model import XdevModel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

SESSIONS_IN, MODEL_OUT = sys.argv[1], sys.argv[2]
T_STAR = 0.70  # deployed sigmoid_score midpoint; hard threshold 0.5 <=> p >= T_STAR
BS = 100      # chunks per simulated validator cycle

def ts(): return time.strftime("[%H:%M:%S]")

assert N_XDEV_FEATURES == 60

with open(SESSIONS_IN, "rb") as f:
    sessions = pickle.load(f)
print(f"{ts()} {len(sessions)} sessions", flush=True)

t0 = time.time()
X_all = np.array([
    extract_xdev_features([prepare_hand_for_miner(h) for h in hands])
    for hands, _ in sessions
], dtype=np.float32)
y_all = np.array([int(y) for _, y in sessions], dtype=np.int32)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
print(f"{ts()} projected+extracted {X_all.shape} in {time.time()-t0:.0f}s", flush=True)

# stratified split: 60% train / 20% calibration / 20% test (same seed as v2 run)
rng = np.random.RandomState(42)
idx = np.arange(len(y_all))
tr_i, cal_i, te_i = [], [], []
for lbl in (0, 1):
    li = idx[y_all == lbl]; rng.shuffle(li)
    n = len(li); a, b = int(n * 0.6), int(n * 0.8)
    tr_i += list(li[:a]); cal_i += list(li[a:b]); te_i += list(li[b:])

def make_batches(pool_idx, n_batches, seed):
    r = random.Random(seed)
    bots = [i for i in pool_idx if y_all[i] == 1]
    hums = [i for i in pool_idx if y_all[i] == 0]
    Xb, yb = [], []
    for _ in range(n_batches):
        nb = r.randint(30, 70); nh = BS - nb
        sel = [bots[r.randrange(len(bots))] for _ in range(nb)] + \
              [hums[r.randrange(len(hums))] for _ in range(nh)]
        Xr = X_all[sel]
        mu, sig = Xr.mean(0), Xr.std(0); sig[sig < 1e-9] = 1.0
        Xb.append(np.clip((Xr - mu) / sig, -5, 5))
        yb.append(np.array([1] * nb + [0] * nh))
    return np.vstack(Xb), np.concatenate(yb)

Xtr, ytr = make_batches(tr_i, 400, seed=202)
print(f"{ts()} training HistGBM on {Xtr.shape} ...", flush=True)
hgb = HistGradientBoostingClassifier(
    max_iter=800, learning_rate=0.035, max_leaf_nodes=63,
    min_samples_leaf=25, l2_regularization=0.3,
    early_stopping=False, random_state=42)
hgb.fit(Xtr, ytr)

Xc, yc = make_batches(cal_i, 150, seed=303)
cal = IsotonicRegression(out_of_bounds="clip")
cal.fit(np.clip(hgb.predict_proba(Xc)[:, 1], 0, 1), yc)
model = XdevModel(hgb=hgb, calibrator=cal)

# evaluate with the validator's literal reward() on held-out test cycles
def eval_cycles(pool_idx, n_cycles, seed):
    r = random.Random(seed)
    bots = [i for i in pool_idx if y_all[i] == 1]
    hums = [i for i in pool_idx if y_all[i] == 0]
    rews, aps = [], []
    for _ in range(n_cycles):
        nb = r.randint(30, 70); nh = BS - nb
        sel = [bots[r.randrange(len(bots))] for _ in range(nb)] + \
              [hums[r.randrange(len(hums))] for _ in range(nh)]
        lab = np.array([1] * nb + [0] * nh)
        Xr = X_all[sel]
        mu, sig = Xr.mean(0), Xr.std(0); sig[sig < 1e-9] = 1.0
        p = model.predict_proba(np.clip((Xr - mu) / sig, -5, 5))
        s = 1.0 / (1.0 + np.exp(-10.0 * (p - T_STAR)))
        rr, m = reward(s, lab)
        rews.append(rr); aps.append(m["ap_score"])
    return float(np.mean(rews)), float(np.mean(aps))

r_te, ap_te = eval_cycles(te_i, 100, seed=606)
print(f"{ts()} TEST reward={r_te:.4f} AP={ap_te:.4f}", flush=True)

model.save(MODEL_OUT)
print(f"{ts()} saved {MODEL_OUT}", flush=True)
