"""
poker44-xdev miner — xdev-trainer-v8 (self-contained, bittensor 10.x)

Rank-blended HGB+LightGBM+ExtraTrees ensemble on 168 features (120 aggregate
ranked on validator-projected payloads + 48 own action n-gram features),
within-batch normalized. Trained on benchmark through 2026-07-17.
Hotkey: zero-1 (UID 66), port 8094.

Self-contained chain plumbing on bittensor 10.x (the finney runtime upgrade of
2026-07-17 broke the 9.x base package's config layer). Runs in its own venv
(/root/work/xdev-venv) so the shared Poker44-subnet venv is untouched. Depends
only on version-stable pieces from the base repo: DetectionSynapse and the
model_manifest utilities.
"""
import sys
import os
import time
import argparse
import subprocess
import traceback
import warnings
from pathlib import Path
from typing import Tuple

warnings.filterwarnings("ignore")

_SUBNET_ROOT = Path("/root/work/Poker44-subnet")
_XDEV_ROOT   = Path("/root/work/poker44-xdev")
sys.path.insert(0, str(_SUBNET_ROOT))
sys.path.insert(0, str(_XDEV_ROOT))

import numpy as np
import bittensor as bt

from poker44.validator.synapse import DetectionSynapse
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)

from xdev.features import XDEV_FEATURE_NAMES, N_XDEV_FEATURES, extract_xdev_features
from xdev.ngram_features import NGRAM_VOCAB, N_NGRAM_FEATURES, extract_ngram_features
from xdev.model import XdevRankBlend, sigmoid_score

_MODEL_PATH = str(_XDEV_ROOT / "models" / "xdev_v8.joblib")  # 168 feats: 120 base + 48 n-gram


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--netuid", type=int, default=126)
    p.add_argument("--wallet.name", dest="wallet_name", default="zero")
    p.add_argument("--wallet.hotkey", dest="wallet_hotkey", default="zero-1")
    p.add_argument("--axon.port", dest="axon_port", type=int, default=8094)
    p.add_argument("--subtensor.network", dest="network", default="finney")
    p.add_argument("--force-validator-permit", dest="force_permit", action="store_true", default=True)
    args, _ = p.parse_known_args()
    return args


class XdevMiner:
    """Self-contained Poker44 xdev miner on bittensor 10.x."""

    def __init__(self, args):
        self.netuid = args.netuid
        self.force_permit = args.force_permit
        bt.logging.info("Setting up bittensor objects (bt %s)." % bt.__version__)
        self.wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        self.subtensor = bt.Subtensor(network=args.network)
        self.metagraph = self.subtensor.metagraph(self.netuid)
        self.hotkey = self.wallet.hotkey.ss58_address
        if self.hotkey not in self.metagraph.hotkeys:
            raise RuntimeError(f"Hotkey {self.hotkey} not registered on netuid {self.netuid}")
        self.uid = self.metagraph.hotkeys.index(self.hotkey)
        bt.logging.info(f"Running on subnet {self.netuid} uid {self.uid} network {args.network}")

        self.xdev_model = XdevRankBlend.load(_MODEL_PATH)
        bt.logging.info(
            f"xdev miner loaded | features={N_XDEV_FEATURES}+{N_NGRAM_FEATURES}="
            f"{N_XDEV_FEATURES + N_NGRAM_FEATURES} | model={type(self.xdev_model).__name__}"
        )

        self.model_manifest = build_local_model_manifest(
            repo_root=_XDEV_ROOT,
            implementation_files=[
                _XDEV_ROOT / "neurons" / "miner.py",
                _XDEV_ROOT / "xdev" / "features.py",
                _XDEV_ROOT / "xdev" / "ngram_features.py",
                _XDEV_ROOT / "xdev" / "model.py",
            ],
            defaults={
                "model_name":    "xdev-trainer-v8",
                "model_version": "rankblend3-ngram-168feat-v8",
                "framework":     "sklearn-lightgbm-rankblend",
                "license":       "MIT",
                "repo_url":      "https://github.com/Ultimate8888/poker44-xdev",
                "repo_commit":   _git_head(_XDEV_ROOT),
                "open_source":   True,
                "inference_mode": "remote",
                "notes": (
                    "Rank-blended ensemble (HistGradientBoosting + LightGBM + ExtraTrees), "
                    "combined by mean within-batch rank-percentile. 168 features: 120 aggregate "
                    "ranked by LightGBM gain on validator-projected payloads, plus 48 own action "
                    "n-gram features. Within-batch normalization. Trained on 2340 benchmark "
                    "sessions (through 2026-07-17) plus 916 real-human sessions."
                ),
                "training_data_statement": (
                    "Trained on 2340 labeled poker sessions (1170 bot, 1170 human) from the Poker44 "
                    "benchmark API (all releases through 2026-07-17) plus 916 real-human sessions "
                    "built from the subnet repo's public hands_generator/human_hands corpus "
                    "(32088 hands). All hands projected through prepare_hand_for_miner before "
                    "feature extraction. Synthetic within-batch training: 700 batches x 100 "
                    "sessions, wide bot prevalence (5-70). Feature set: 120 features by LightGBM "
                    "gain plus 48 own action n-gram features. Model: rank-blended ensemble of "
                    "sklearn HistGradientBoosting, LightGBM and ExtraTrees. No private data used."
                ),
                "private_data_attestation": False,
            },
        )
        compliance = evaluate_manifest_compliance(self.model_manifest)
        status = compliance.get("status", "unknown")
        digest = manifest_digest(self.model_manifest)[:16]
        bt.logging.info(f"xdev manifest status={status} digest={digest}...")

        # Axon: serve + attach
        self.axon = bt.Axon(wallet=self.wallet, port=args.axon_port)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        self.subtensor.serve_axon(netuid=self.netuid, axon=self.axon)
        self.axon.start()
        bt.logging.info(f"Axon serving on netuid {self.netuid}: {self.axon}")
        bt.logging.info(f"Miner UID: {self.uid} | Incentive: {float(self.metagraph.I[self.uid]):.4f}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        try:
            synapse.model_manifest = self.model_manifest
            chunks = synapse.chunks or []
            if not chunks:
                synapse.risk_scores = []
                synapse.predictions = []
                return synapse

            base = np.array([extract_xdev_features(c) for c in chunks], dtype=np.float32)
            ngram = np.array([extract_ngram_features(c) for c in chunks], dtype=np.float32)
            feats = np.hstack([base, ngram])

            mu = feats.mean(0)
            sig = feats.std(0)
            sig[sig < 1e-9] = 1.0
            feats_norm = np.clip((feats - mu) / sig, -5.0, 5.0)

            probs = self.xdev_model.predict_proba(feats_norm)
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
        caller = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
        if caller is None or caller not in self.metagraph.hotkeys:
            return True, "unrecognized hotkey"
        uid = self.metagraph.hotkeys.index(caller)
        if self.force_permit and not bool(self.metagraph.validator_permit[uid]):
            return True, "non-validator hotkey"
        return False, "ok"

    async def priority(self, synapse: DetectionSynapse) -> float:
        caller = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
        if caller in self.metagraph.hotkeys:
            uid = self.metagraph.hotkeys.index(caller)
            return float(self.metagraph.S[uid])
        return 0.0

    def run(self):
        bt.logging.info("xdev TrainedMiner running...")
        step = 0
        while True:
            time.sleep(60)
            step += 1
            if step % 5 == 0:
                try:
                    self.metagraph.sync(subtensor=self.subtensor)
                except Exception as exc:
                    bt.logging.warning(f"resync_metagraph failed: {exc}")
            try:
                inc = float(self.metagraph.I[self.uid])
                blk = int(self.metagraph.block)
            except Exception:
                inc, blk = -1.0, -1
            bt.logging.info(f"xdev UID={self.uid} | incentive={inc:.6f} | block={blk}")


if __name__ == "__main__":
    bt.logging.enable_info()
    miner = XdevMiner(_parse_args())
    miner.run()
