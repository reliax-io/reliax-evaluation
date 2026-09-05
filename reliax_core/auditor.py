"""Error meta-model ("the auditor"): a second model trained to predict when the
base credit model is wrong.

Trained on the CALIBRATION split (never the base model's training data): inputs are
the raw applicant features plus the base model's own outputs (p_default, margin);
target is 1{base model misclassified}. At inference it returns p_error - the
auditor's estimate that this specific prediction is a miss.

A ranking signal folded into the composite score, NOT a guarantee. Its flag
threshold is set empirically at startup to the calibration split's 90th percentile
of p_error, so ~10% of typical traffic gets an auditor flag by construction.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FLAG_QUANTILE = 0.90


class ErrorAuditor:
    def __init__(self, seed: int = 5):
        self.model = HistGradientBoostingClassifier(max_iter=120, max_depth=4,
                                                    random_state=seed)

    @staticmethod
    def _features(X: np.ndarray, probs: np.ndarray) -> np.ndarray:
        p_sorted = np.sort(probs, axis=1)[:, ::-1]
        margin = p_sorted[:, 0] - p_sorted[:, 1]
        return np.column_stack([X, probs[:, 1], margin])

    def fit(self, X_cal: np.ndarray, probs_cal: np.ndarray, y_cal: np.ndarray,
            pred_cal: np.ndarray):
        wrong = (pred_cal != y_cal).astype(int)
        F = self._features(X_cal, probs_cal)
        self.model.fit(F, wrong)
        p_err_cal = self.model.predict_proba(F)[:, 1]
        self.flag_threshold = float(np.quantile(p_err_cal, FLAG_QUANTILE))
        self.base_error_rate = float(wrong.mean())
        return self

    def assess(self, x: np.ndarray, probs: np.ndarray) -> dict:
        F = self._features(np.atleast_2d(x), np.atleast_2d(probs))
        p_error = float(self.model.predict_proba(F)[0, 1])
        return {
            "p_error": round(p_error, 4),
            "flag": p_error > self.flag_threshold,
            "flag_threshold": round(self.flag_threshold, 4),
        }

    def p_error_batch(self, X: np.ndarray, probs: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._features(X, probs))[:, 1]
