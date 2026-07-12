"""
poker44-xdev miner — xdev-trainer-v3
HistGradientBoostingClassifier on top-60 features re-ranked on validator-projected
payloads, within-batch normalized.
Hotkey: zero-1 (UID 66), port 8094.

Must be run from /root/work/Poker44-subnet (or with PYTHONPATH set to it)
so that the poker44 base package is importable.
"""
import sys
import os
import math
import time
import hashlib
import subprocess
import traceback
import warnings
from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore")

# Ensure both repos are on the path
_SUBNET_ROOT = Path("/root/work/Poker44-subnet")
_XDEV_ROOT   = Path("/root/work/poker44-xdev")
sys.path.insert(0, str(_SUBNET_ROOT))
sys.path.insert(0, str(_XDEV_ROOT))

import numpy as np
import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)

from xdev.features import XDEV_FEATURE_NAMES, N_XDEV_FEATURES, extract_xdev_features
from xdev.model import XdevModel, sigmoid_score

_MODEL_PATH = str(_XDEV_ROOT / "models" / "xdev_v3.joblib")


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


class XdevMiner(BaseMinerNeuron):
    """Poker44 xdev miner: HistGBM on 25 temporal/behavioral features."""

    def __init__(self, config=None):
        super().__init__(config=config)

        self.xdev_model = XdevModel.load(_MODEL_PATH)
        bt.logging.info(
            f"xdev miner loaded | features={N_XDEV_FEATURES} | "
            f"model={type(self.xdev_model.hgb).__name__}"
        )

        # Build manifest
        self.model_manifest = build_local_model_manifest(
            repo_root=_XDEV_ROOT,
            implementation_files=[
                _XDEV_ROOT / "neurons" / "miner.py",
                _XDEV_ROOT / "xdev" / "features.py",
                _XDEV_ROOT / "xdev" / "model.py",
            ],
            defaults={
                "model_name":    "xdev-trainer-v3",
                "model_version": "hgb-within-batch-60feat-v3",
                "framework":     "sklearn-histgbm",
                "license":       "MIT",
                "repo_url":      "https://github.com/Ultimate8888/poker44-xdev",
                "repo_commit":   _git_head(_XDEV_ROOT),
                "open_source":   True,
                "inference_mode": "remote",
                "notes": (
                    "Single HistGradientBoostingClassifier + IsotonicRegression calibration. "
                    "Top-60 temporal and behavioral features re-ranked on validator-projected "
                    "payloads (prepare_hand_for_miner view). Within-batch normalization. "
                    "Trained on 1588 sessions incl. competition-era releases, 600 batches."
                ),
                "training_data_statement": (
                    "Trained on 1588 labeled poker sessions (794 bot, 794 human) from the Poker44 "
                    "benchmark API (all releases through 2026-07-12, including competition-era "
                    "data). All hands projected through the validator's prepare_hand_for_miner "
                    "canonicalizer before feature extraction to match the miner-visible payload "
                    "distribution. Synthetic within-batch training: 600 batches x 100 sessions "
                    "(30-70 bots per batch). Feature set: top-60 features selected by LightGBM "
                    "gain and Cohen's d on projected data. Model: sklearn "
                    "HistGradientBoostingClassifier + IsotonicRegression calibration. "
                    "No private data used."
                ),
                "private_data_attestation": False,
            },
        )
        compliance = evaluate_manifest_compliance(self.model_manifest)
        missing = compliance.get("missing_fields", [])
        status = compliance.get("status", "unknown")
        digest = manifest_digest(self.model_manifest)[:16]
        bt.logging.info(f"xdev manifest status={status} missing={missing} digest={digest}...")
        bt.logging.info(f"xdev TrainedMiner running...")
        bt.logging.info(f"Miner UID: {self.uid} | Incentive: {float(self.metagraph.I[self.uid]):.4f}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        try:
            synapse.model_manifest = self.model_manifest
            chunks = synapse.chunks or []
            if not chunks:
                synapse.risk_scores = []
                synapse.predictions = []
                return synapse

            # Extract 25 features per chunk
            feats = np.array([extract_xdev_features(c) for c in chunks], dtype=np.float32)

            # Within-batch normalization
            mu  = feats.mean(0)
            sig = feats.std(0)
            sig[sig < 1e-9] = 1.0
            feats_norm = np.clip((feats - mu) / sig, -5.0, 5.0)

            # Score
            probs  = self.xdev_model.predict_proba(feats_norm)
            scores = [sigmoid_score(float(p), t_star=0.70, sharpness=10.0) for p in probs]

            n_flagged = sum(1 for s in scores if s > 0.5)
            bt.logging.info(
                f"Scored {len(scores)} chunks | flagged={n_flagged} | "
                f"range=[{min(scores):.3f}, {max(scores):.3f}]"
            )
            synapse.risk_scores = scores
            synapse.predictions = [s >= 0.5 for s in scores]
        except Exception:
            bt.logging.error(traceback.format_exc())
            synapse.risk_scores = [0.5] * len(synapse.chunks or [])
            synapse.predictions = [False] * len(synapse.chunks or [])
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)

    def get_model_manifest(self):
        return self.model_manifest


if __name__ == "__main__":
    with XdevMiner() as miner:
        while True:
            bt.logging.info(
                f"xdev UID={miner.uid} | "
                f"incentive={float(miner.metagraph.I[miner.uid]):.6f} | "
                f"block={int(miner.metagraph.block.item())}"
            )
            time.sleep(60)
