"""
xdev-trainer-v2 model: HistGradientBoostingClassifier + IsotonicRegression calibration.
Single model (not an ensemble) trained on within-batch normalized top-60 features
extracted from validator-projected payloads.
Genuinely different architecture from the main 7-model GBDT ensemble.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import joblib


class XdevModel:
    def __init__(self, hgb: HistGradientBoostingClassifier, calibrator: IsotonicRegression):
        self.hgb = hgb
        self.calibrator = calibrator

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        return self.hgb.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_proba_raw(X)
        cal = self.calibrator.predict(np.clip(raw, 0, 1))
        return np.clip(cal, 0, 1)

    def save(self, path: str):
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "XdevModel":
        return joblib.load(path)


class XdevEnsemble:
    """Calibrated soft-vote ensemble over heterogeneous sklearn/LightGBM members.

    predict_proba(X) = isotonic( mean_i member_i.predict_proba(X)[:, 1] ).
    Members are trained on within-batch-normalized features extracted from
    validator-projected payloads.
    """

    def __init__(self, members, calibrator):
        self.members = list(members)
        self.calibrator = calibrator

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cal = self.calibrator.predict(np.clip(self.predict_proba_raw(X), 0, 1))
        return np.clip(cal, 0, 1)

    def save(self, path: str):
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "XdevEnsemble":
        return joblib.load(path)


def sigmoid_score(prob: float, t_star: float = 0.70, sharpness: float = 10.0) -> float:
    """Map calibrated probability → final score [0,1] via shifted sigmoid."""
    import math
    return 1.0 / (1.0 + math.exp(-sharpness * (prob - t_star)))
