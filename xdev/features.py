"""
Feature selection for xdev-trainer-v3.
Imports all features from the main repo and selects a top-60 subset ranked
on validator-projected payloads (prepare_hand_for_miner view) over the full
benchmark history INCLUDING the 2026-07-11/12 competition-era releases,
since new bot families drifted strongly on table-size and action-pattern
features. Ranking: LightGBM gain (0.6) + |Cohen's d| (0.4) on
within-batch-normalized features.
"""
import sys
import numpy as np
sys.path.insert(0, "/root/work/Poker44-subnet")

from poker44.miner_model.features import FEATURE_NAMES as _ALL_NAMES, extract_chunk_features as _extract_all

# Top-60 features ranked on validator-projected data (benchmark through 2026-07-12)
XDEV_FEATURE_NAMES = [
    "min_frac_fold",
    "q50_frac_fold",
    "max_frac_check",
    "q50_frac_check",
    "q50_frac_call",
    "q50_frac_bet",
    "max_frac_raise",
    "std_action_entropy",
    "q10_aggression_factor",
    "q10_size_modal_frac",
    "q10_size_bucket_snap",
    "q10_pot_growth_mean",
    "max_reach_turn_plus",
    "min_schema_actor_switch_rate",
    "min_schema_actor_run_max_share",
    "max_schema_actor_run_max_share",
    "std_schema_action_run_max_share",
    "q10_schema_action_run_max_share",
    "max_schema_street_entropy",
    "max_schema_raise_to_share",
    "q10_schema_raise_to_share",
    "max_schema_nonzero_amount_share",
    "max_schema_starting_stack_iqr_bb",
    "min_extra_unique_actor_share",
    "q90_extra_unique_actor_share",
    "max_extra_street_count",
    "q10_extra_amount_mean_bb",
    "max_extra_amount_q90_bb",
    "q10_extra_amount_q90_bb",
    "min_extra_stack_mean_bb",
    "max_extra_stack_std_bb",
    "unique_bb_buckets",
    "top_bb_bucket_share",
    "lag1_autocorr_aggression",
    "lag1_autocorr_fold",
    "lag2_autocorr_fold",
    "lag2_autocorr_size_cv",
    "trend_slope_aggression",
    "trend_slope_fold",
    "trend_slope_size_cv",
    "lag3_autocorr_aggression",
    "lag3_autocorr_size_cv",
    "seq_action_sig_unique_share",
    "seq_actor_sig_top_share",
    "seq_street_sig_unique_share",
    "seq_amount_bucket_sig_unique_share",
    "seq_high_aggression_hand_rate",
    "seq_low_action_entropy_hand_rate",
    "seq_high_actor_entropy_hand_rate",
    "seq_long_action_hand_rate",
    "bet_pot_ratio_cv",
    "bet_pot_ratio_cluster_frac",
    "hero_bet_pot_ratio_cv",
    "aggro_q4_minus_q1",
    "chunk_flop_action_share",
    "chunk_river_action_share",
    "hero_flop_action_share",
    "hero_river_action_share",
    "flop_aggro_share",
    "turn_aggro_share",
]

N_XDEV_FEATURES = len(XDEV_FEATURE_NAMES)
assert N_XDEV_FEATURES == 60

_XDEV_INDICES = [_ALL_NAMES.index(n) for n in XDEV_FEATURE_NAMES]


def extract_xdev_features(hands) -> np.ndarray:
    """Extract 60 xdev features from a session (list of hands)."""
    all_feats = _extract_all(hands)
    feats = np.array([all_feats[i] for i in _XDEV_INDICES], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_xdev_batch(sessions_list) -> np.ndarray:
    """Extract features for a list of sessions, returns shape (N, 60)."""
    return np.array([extract_xdev_features(h) for h in sessions_list], dtype=np.float32)
