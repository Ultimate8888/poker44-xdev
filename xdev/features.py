"""
Feature selection for xdev-trainer-v1.
Imports all features from the main repo and selects the top-25 temporal/behavioral subset.
This produces outputs with different characteristics than the full 317-feature ensemble.
"""
import sys
import numpy as np
sys.path.insert(0, "/root/work/Poker44-subnet")

from poker44.miner_model.features import FEATURE_NAMES as _ALL_NAMES, extract_chunk_features as _extract_all

# Top-25 features by combined Cohen's d + LightGBM importance signal
XDEV_FEATURE_NAMES = [
    "size_cv_half_delta",            # d=-2.80  dominant discriminator
    "lag2_autocorr_aggression",      # d=-2.23  bots have flat aggression patterns
    "aggro_q4_minus_q1",             # d=-1.36  quartile consistency
    "fold_half_delta",               # d=-1.24  bots don't adjust fold rate
    "chunk_flop_action_share",       # d=-1.12  bots fold preflop → less flop action
    "lag3_autocorr_size_cv",         # lgbm rank 6 — captures longer-term size patterns
    "trend_slope_aggression",        # d=-0.88  bots don't ramp up aggression over time
    "lag2_autocorr_size_cv",         # d=-0.84  bet sizing repetition
    "lag3_autocorr_fold",            # lgbm rank 12 — longer-term fold autocorrelation
    "size_cv_q4_minus_q1",           # d=-0.80  size variance quartile spread
    "seq_long_action_hand_rate",     # d=-0.69  bots play shorter action sequences
    "seq_street_sig_unique_share",   # d=-0.68  bot action signatures less unique
    "lag3_autocorr_aggression",      # d=-0.88  lgbm rank 19
    "seq_low_action_entropy_hand_rate", # d=-0.64 bots have low entropy hands more often
    "turn_aggro_share",              # lgbm rank 15 — bots less aggressive on turn
    "top_bb_bucket_share",           # d=+0.57  bots more often in top BB bucket
    "seq_high_actor_entropy_hand_rate", # d=+0.43 signal in actor diversity
    "seq_street_sig_top_share",      # d=+0.42
    "lag2_autocorr_fold",            # d=-0.36  fold pattern repetition
    "seq_high_aggression_hand_rate", # d=+0.36
    "q50_frac_bet",                  # d=+0.34  bots bet more at median
    "std_action_entropy",            # d=-0.34  bots less variable in entropy
    "hero_flop_action_share",        # d=-0.31
    "flop_aggro_share",              # d=+0.27  flop aggression proportion
    "chunk_river_action_share",      # d=+0.27
]

N_XDEV_FEATURES = len(XDEV_FEATURE_NAMES)
assert N_XDEV_FEATURES == 25

_XDEV_INDICES = [_ALL_NAMES.index(n) for n in XDEV_FEATURE_NAMES]


def extract_xdev_features(hands) -> np.ndarray:
    """Extract 25 xdev features from a session (list of hands)."""
    all_feats = _extract_all(hands)
    return np.array([all_feats[i] for i in _XDEV_INDICES], dtype=np.float32)


def extract_xdev_batch(sessions_list) -> np.ndarray:
    """Extract features for a list of sessions, returns shape (N, 25)."""
    return np.array([extract_xdev_features(h) for h in sessions_list], dtype=np.float32)
