"""
Feature selection for xdev-trainer-v6.
Top-120 subset of the public 317-feature extractor, ranked by our own LightGBM
gain over within-batch-normalized validator-projected payloads. Larger view than
v5 (70) because the competition metric is AP-dominated (within-batch ranking),
which rewards richer signal; 120 was the walk-forward optimum (120 > 70/200/317).
All extraction on the prepare_hand_for_miner view.
"""
import sys
import numpy as np
sys.path.insert(0, "/root/work/Poker44-subnet")

from poker44.miner_model.features import FEATURE_NAMES as _ALL_NAMES, extract_chunk_features as _extract_all

# Top-120 features by LightGBM gain on projected data (benchmark through 2026-07-12)
XDEV_FEATURE_NAMES = [
    "std_frac_fold",
    "min_frac_fold",
    "q10_frac_fold",
    "std_frac_check",
    "max_frac_check",
    "mean_frac_call",
    "std_frac_call",
    "max_frac_call",
    "q50_frac_call",
    "q90_frac_call",
    "mean_frac_bet",
    "q50_frac_bet",
    "q90_frac_bet",
    "std_frac_raise",
    "q50_frac_raise",
    "q90_frac_raise",
    "mean_action_entropy",
    "std_action_entropy",
    "q90_action_entropy",
    "std_aggression_factor",
    "mean_size_cv",
    "std_size_cv",
    "q90_size_cv",
    "mean_size_modal_frac",
    "std_size_bucket_snap",
    "max_size_bucket_snap",
    "q10_size_bucket_snap",
    "q50_hero_frac",
    "std_hero_fold_frac",
    "max_hero_size_cv",
    "q90_hero_size_cv",
    "mean_pot_growth_mean",
    "q10_pot_growth_mean",
    "q50_pot_growth_mean",
    "q90_pot_growth_mean",
    "mean_reach_turn_plus",
    "std_reach_turn_plus",
    "max_reach_turn_plus",
    "min_schema_actor_run_max_share",
    "q10_schema_actor_run_max_share",
    "q50_schema_actor_run_max_share",
    "std_schema_action_run_max_share",
    "q10_schema_action_run_max_share",
    "q50_schema_action_run_max_share",
    "std_schema_street_entropy",
    "max_schema_street_entropy",
    "q50_schema_street_entropy",
    "std_schema_preflop_share",
    "mean_schema_pot_monotonic_rate",
    "mean_schema_raise_to_share",
    "std_schema_raise_to_share",
    "max_schema_raise_to_share",
    "q50_schema_raise_to_share",
    "q90_schema_raise_to_share",
    "std_schema_nonzero_amount_share",
    "max_schema_nonzero_amount_share",
    "q50_schema_nonzero_amount_share",
    "q90_schema_nonzero_amount_share",
    "mean_extra_unique_actor_share",
    "std_extra_unique_actor_share",
    "q50_extra_unique_actor_share",
    "q90_extra_unique_actor_share",
    "std_extra_street_count",
    "max_extra_street_count",
    "q10_extra_street_count",
    "mean_extra_n_players",
    "std_extra_n_players",
    "min_extra_n_players",
    "q50_extra_n_players",
    "std_extra_amount_mean_bb",
    "q90_extra_amount_mean_bb",
    "std_extra_amount_q90_bb",
    "max_extra_amount_q90_bb",
    "q50_extra_amount_q90_bb",
    "q90_extra_amount_q90_bb",
    "min_extra_stack_std_bb",
    "q10_extra_stack_std_bb",
    "n_hands",
    "unique_bb_buckets",
    "top_action_share",
    "top_bb_bucket_share",
    "hero_participation_rate",
    "global_action_entropy",
    "lag1_autocorr_aggression",
    "lag1_autocorr_fold",
    "lag1_autocorr_size_cv",
    "aggro_half_delta",
    "size_cv_half_delta",
    "lag2_autocorr_fold",
    "lag2_autocorr_size_cv",
    "trend_slope_aggression",
    "trend_slope_fold",
    "trend_slope_size_cv",
    "lag3_autocorr_aggression",
    "lag3_autocorr_fold",
    "lag3_autocorr_size_cv",
    "seq_action_sig_top_share",
    "seq_action_sig_unique_share",
    "seq_actor_sig_top_share",
    "seq_actor_sig_unique_share",
    "seq_street_sig_top_share",
    "seq_street_sig_unique_share",
    "seq_amount_bucket_sig_top_share",
    "seq_amount_bucket_sig_unique_share",
    "seq_high_aggression_hand_rate",
    "seq_low_action_entropy_hand_rate",
    "seq_high_actor_entropy_hand_rate",
    "seq_long_action_hand_rate",
    "bet_pot_ratio_cv",
    "bet_pot_ratio_cluster_frac",
    "hero_bet_pot_ratio_cv",
    "lag1_autocorr_hero_bet_pot",
    "action_bigram_entropy",
    "size_cv_q4_minus_q1",
    "chunk_flop_action_share",
    "chunk_river_action_share",
    "hero_flop_action_share",
    "hero_river_action_share",
    "flop_aggro_share",
    "turn_aggro_share",
]

N_XDEV_FEATURES = len(XDEV_FEATURE_NAMES)
assert N_XDEV_FEATURES == 120

_XDEV_INDICES = [_ALL_NAMES.index(n) for n in XDEV_FEATURE_NAMES]


def extract_xdev_features(hands) -> np.ndarray:
    """Extract 120 xdev features from a session (list of hands)."""
    all_feats = _extract_all(hands)
    feats = np.array([all_feats[i] for i in _XDEV_INDICES], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_xdev_batch(sessions_list) -> np.ndarray:
    """Extract features for a list of sessions, returns shape (N, 120)."""
    return np.array([extract_xdev_features(h) for h in sessions_list], dtype=np.float32)
