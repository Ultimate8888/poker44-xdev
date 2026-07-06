"""
poker44-xdev miner — xdev-trainer-v1
Single HistGradientBoostingClassifier, top-25 temporal features, within-batch normalization.
Hotkey: zero-1 (UID 66), port 8094.
"""
import sys
import os
import time
import traceback
import numpy as np

sys.path.insert(0, "/root/work/Poker44-subnet")
sys.path.insert(0, "/root/work/poker44-xdev")

import bittensor as bt

from poker44.protocol import DetectionSynapse
from xdev.features import XDEV_FEATURE_NAMES, N_XDEV_FEATURES, extract_xdev_features
from xdev.model import XdevModel, sigmoid_score
from xdev.manifest import build_manifest

MODEL_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "models", "xdev_v1.joblib")


class XdevMiner:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.wallet = bt.wallet(
            name=self.config.wallet.name,
            hotkey=self.config.wallet.hotkey,
        )
        self.subtensor = bt.subtensor(network=self.config.subtensor.network)
        self.metagraph = bt.metagraph(netuid=126, network=self.config.subtensor.network)
        self.metagraph.sync()
        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)

        self.model = XdevModel.load(MODEL_PATH)
        bt.logging.info(
            f"xdev miner loaded: {N_XDEV_FEATURES} features, "
            f"model={type(self.model.hgb).__name__}"
        )

        manifest, missing = build_manifest()
        status = "transparent" if not missing else f"incomplete (missing: {missing})"
        bt.logging.info(f"Manifest status={status} digest={manifest.get('manifest_digest','')[:16]}...")

        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
        )

    def _default_config(self):
        parser = bt.ArgumentParser()
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        config.axon.port = config.axon.port or 8094
        config.wallet.name = config.wallet.name or "zero"
        config.wallet.hotkey = config.wallet.hotkey or "zero-1"
        config.subtensor.network = config.subtensor.network or "finney"
        return config

    def blacklist(self, synapse: DetectionSynapse) -> tuple[bool, str]:
        return False, ""

    def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        try:
            sessions = synapse.sessions  # list of 100 hand-lists
            if not sessions:
                synapse.predictions = []
                return synapse

            n = len(sessions)
            # Extract 25 features per session
            feats = np.array([extract_xdev_features(s) for s in sessions], dtype=np.float32)

            # Within-batch normalization
            mu = feats.mean(0)
            sig = feats.std(0)
            sig[sig < 1e-9] = 1.0
            feats_norm = np.clip((feats - mu) / sig, -5.0, 5.0)

            # Score
            probs = self.model.predict_proba(feats_norm)
            scores = [sigmoid_score(float(p), t_star=0.48, sharpness=10.0) for p in probs]

            n_flagged = sum(1 for s in scores if s > 0.5)
            bt.logging.info(
                f"Scored {n} sessions | flagged={n_flagged} | "
                f"range=[{min(scores):.3f}, {max(scores):.3f}]"
            )

            synapse.predictions = scores
        except Exception:
            bt.logging.error(traceback.format_exc())
            synapse.predictions = [0.5] * len(synapse.sessions or [])
        return synapse

    def run(self):
        uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"xdev TrainedMiner running | UID={uid} | port={self.config.axon.port}")

        self.subtensor.serve_axon(netuid=126, axon=self.axon)
        self.axon.start()

        step = 0
        while True:
            try:
                if step % 60 == 0:
                    self.metagraph.sync()
                    incentive = float(self.metagraph.I[uid])
                    bt.logging.info(
                        f"UID={uid} | incentive={incentive:.6f} | "
                        f"block={int(self.metagraph.block.item())}"
                    )
                time.sleep(10)
                step += 1
            except KeyboardInterrupt:
                self.axon.stop()
                break
            except Exception:
                bt.logging.error(traceback.format_exc())
                time.sleep(30)


if __name__ == "__main__":
    bt.logging.set_trace(False)
    miner = XdevMiner()
    miner.run()
