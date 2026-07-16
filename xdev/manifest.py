"""Build and verify the Poker44 transparency manifest for xdev-trainer-v1."""
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return ""


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(model_path: Path = None) -> dict:
    source_files = {
        "features.py": REPO_ROOT / "xdev" / "features.py",
        "model.py":    REPO_ROOT / "xdev" / "model.py",
        "ngram_features.py": REPO_ROOT / "xdev" / "ngram_features.py",
        "miner.py":    REPO_ROOT / "neurons" / "miner.py",
    }

    manifest = {
        "model_name":    "xdev-trainer-v7",
        "model_version": "rankblend3-ngram-168feat-v7",
        "framework":     "sklearn-lightgbm-rankblend",
        "license":       "MIT",
        "repo_url":      "https://github.com/Ultimate8888/poker44-xdev",
        "repo_commit":   _git_head(REPO_ROOT),
        "source_hashes": {k: _hash_file(v) for k, v in source_files.items()},
        "open_source":   True,
        "private_data_attestation": False,
        "training_data_statement": (
            "Trained on 1588 labeled poker sessions (794 bot, 794 human) from the Poker44 "
            "benchmark API (all releases through 2026-07-12) plus 916 real-human sessions from "
            "the public hands_generator/human_hands corpus. All hands projected through the "
            "validator's prepare_hand_for_miner canonicalizer before feature extraction. "
            "Synthetic within-batch training: 800 batches × 100 sessions (5-70 bots per batch). "
            "Feature set: top-60 features selected by LightGBM gain and Cohen's d on "
            "projected data from a 317-feature analysis, plus 10 competition-era drift "
            "features. Model: soft-vote ensemble (HistGradientBoosting + LightGBM + "
            "ExtraTrees) + IsotonicRegression calibration."
        ),
    }

    if model_path and model_path.exists():
        manifest["model_digest"] = _hash_file(model_path)

    content = json.dumps(manifest, sort_keys=True).encode()
    manifest["manifest_digest"] = hashlib.sha256(content).hexdigest()

    missing = [k for k in ["open_source", "repo_url", "repo_commit",
                            "model_name", "model_version", "training_data_statement",
                            "private_data_attestation"]
               if not manifest.get(k) and manifest.get(k) is not False]

    return manifest, missing
