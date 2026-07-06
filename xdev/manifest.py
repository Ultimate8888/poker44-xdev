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
        "miner.py":    REPO_ROOT / "neurons" / "miner.py",
    }

    manifest = {
        "model_name":    "xdev-trainer-v1",
        "model_version": "hgb-within-batch-25feat-v1",
        "framework":     "sklearn-histgbm",
        "license":       "MIT",
        "repo_url":      "https://github.com/SerGem811/poker44-xdev",
        "repo_commit":   _git_head(REPO_ROOT),
        "source_hashes": {k: _hash_file(v) for k, v in source_files.items()},
        "open_source":   True,
        "private_data_attestation": False,
        "training_data_statement": (
            "Trained on 724 labeled poker sessions (362 bot, 362 human) from the Poker44 "
            "benchmark API (fetched 2026-07-06). Sessions augmented by concatenation (pairs "
            "and triples) to 1326 per class. Synthetic within-batch training: 300 batches × 100 "
            "sessions (30-70 bots per batch, variable composition for robustness). "
            "Feature set: top-25 temporal and behavioral features selected by Cohen's d and "
            "LightGBM importance from a 317-feature analysis. "
            "Model: sklearn HistGradientBoostingClassifier + IsotonicRegression calibration."
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
