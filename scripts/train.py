"""
Train xdev-trainer-v1: HistGBM + IsotonicRegression calibration on top-25 temporal features.
Saves to scratchpad first; call deploy.py to push to models/.
"""
import sys, os, time, pickle, random, hashlib

SCRATCHPAD = "/tmp/claude-0/-root-work-Poker44-subnet/af349465-764e-476f-a1d1-e210d196bbe9/scratchpad"
SESSIONS_IN = f"{SCRATCHPAD}/sessions_fresh.pkl"
CANDIDATE_OUT = f"{SCRATCHPAD}/xdev_v1_candidate.joblib"

sys.path.insert(0, "/root/work/Poker44-subnet")
sys.path.insert(0, "/root/work/poker44-xdev")

import numpy as np, warnings
warnings.filterwarnings("ignore")

from xdev.features import XDEV_FEATURE_NAMES, N_XDEV_FEATURES, extract_xdev_batch
from xdev.model import XdevModel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

def ts(): return time.strftime("[%H:%M:%S]")

assert N_XDEV_FEATURES == 25, f"Expected 25, got {N_XDEV_FEATURES}"

print(f"{ts()} Loading sessions from {SESSIONS_IN}")
with open(SESSIONS_IN, "rb") as f:
    sessions = pickle.load(f)

bot_s   = [h for h,l in sessions if l==1]
human_s = [h for h,l in sessions if l==0]
print(f"{ts()} {len(sessions)} sessions: {len(bot_s)} bot, {len(human_s)} human")

print(f"{ts()} Extracting features...", flush=True)
t0 = time.time()
bot_f   = extract_xdev_batch(bot_s)
human_f = extract_xdev_batch(human_s)
print(f"{ts()} Done in {time.time()-t0:.1f}s — shape={bot_f.shape}")

# Cohen's d for all 25 features
import math
def cd(a,b):
    na,nb=len(a),len(b)
    p=math.sqrt(((na-1)*a.var()+(nb-1)*b.var())/(na+nb-2))
    return float((a.mean()-b.mean())/p) if p>1e-9 else 0.0

print(f"\n{ts()} Feature Cohen's d (all 25):")
for i,nm in enumerate(XDEV_FEATURE_NAMES):
    d=cd(bot_f[:,i],human_f[:,i])
    print(f"  {nm:<40s}  d={d:+.3f}")

# Augment
rng=random.Random(42)
def aug(s,rng):
    s=list(s);rng.shuffle(s);n=len(s);a=[]
    for i in range(0,(n//2)*2,2):a.append(s[i]+s[i+1])
    for i in range(0,(n//3)*3,3):a.append(s[i]+s[i+1]+s[i+2])
    return a

bot_aug  = bot_s  + aug(bot_s,  rng)
human_aug= human_s+ aug(human_s,rng)
print(f"\n{ts()} Augmented: {len(bot_aug)} bot, {len(human_aug)} human")
t1=time.time()
bot_all   = extract_xdev_batch(bot_aug)
human_all = extract_xdev_batch(human_aug)
print(f"{ts()} Done in {time.time()-t1:.1f}s")

# Build synthetic within-batch training set (300 batches, 30-70 bots)
rng2=random.Random(1234); N_B=300; BS=100
X_list,y_list=[],[]
for _ in range(N_B):
    nb=rng2.randint(30,70); nh=BS-nb
    bi=[rng2.randrange(len(bot_all)) for _ in range(nb)]
    hi=[rng2.randrange(len(human_all)) for _ in range(nh)]
    b=np.vstack([bot_all[bi],human_all[hi]])
    mu=b.mean(0);sig=b.std(0);sig[sig<1e-9]=1.0
    X_list.append(np.clip((b-mu)/sig,-5,5))
    y_list.extend([1]*nb+[0]*nh)
X=np.vstack(X_list); y=np.array(y_list,dtype=np.int32)

holdout_n=50*BS
X_tr,X_ho=X[:-holdout_n],X[-holdout_n:]
y_tr,y_ho=y[:-holdout_n],y[-holdout_n:]
print(f"\n{ts()} Train={len(X_tr)}, Holdout={len(X_ho)}, shape={X.shape}")

# Train HistGBM
print(f"\n{ts()} Training HistGradientBoostingClassifier...", flush=True)
t2=time.time()
hgb=HistGradientBoostingClassifier(
    max_iter=600, learning_rate=0.04, max_leaf_nodes=63,
    min_samples_leaf=30, l2_regularization=0.1,
    random_state=42, verbose=0
)
hgb.fit(X_tr,y_tr)
raw=hgb.predict_proba(X_ho)[:,1]
ap_raw=average_precision_score(y_ho,raw)
print(f"  HistGBM: AP={ap_raw:.4f}  AUROC={roc_auc_score(y_ho,raw):.4f}  ({time.time()-t2:.1f}s)")

# Isotonic calibration
cal=IsotonicRegression(out_of_bounds="clip")
cal.fit(raw,y_ho)
cal_p=np.clip(cal.predict(raw),0,1)
ap_cal=average_precision_score(y_ho,cal_p)
print(f"  Calibrated: AP={ap_cal:.4f}")

model=XdevModel(hgb=hgb,calibrator=cal)
import joblib
joblib.dump(model,CANDIDATE_OUT,compress=3)
with open(CANDIDATE_OUT,"rb") as f:
    digest=hashlib.sha256(f.read()).hexdigest()

print(f"\n{ts()} ✅ Candidate saved: {CANDIDATE_OUT}")
print(f"   AP={ap_cal:.4f}  digest={digest[:20]}...")
print(f"   25 features  model=HistGBM+IsotonicRegression")
print(f"\n=== DONE ===")
