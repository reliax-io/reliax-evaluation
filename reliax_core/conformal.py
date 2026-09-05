"""Split conformal prediction for classification.

Nonconformity score s(x, y) = 1 - p_hat_y(x). With calibration scores s_1..s_n,
q_hat(alpha) is the ceil((n+1)(1-alpha))/n empirical quantile and the prediction set
C(x) = { y : 1 - p_hat_y(x) <= q_hat } satisfies P(Y in C(X)) >= 1 - alpha marginally,
distribution-free, on exchangeable data (Vovk et al., 2005).
"""
import math

import numpy as np


class ConformalCalibrator:
    def __init__(self, calib_probs: np.ndarray, calib_labels: np.ndarray):
        # calib_probs: (n, n_classes) predicted probabilities on the calibration split
        n = len(calib_labels)
        scores = 1.0 - calib_probs[np.arange(n), calib_labels]
        self.sorted_scores = np.sort(scores)
        self.n = n

    def qhat(self, alpha: float) -> float:
        level = min(1.0, math.ceil((self.n + 1) * (1 - alpha)) / self.n)
        return float(np.quantile(self.sorted_scores, level, method="higher"))

    def prediction_set(self, probs: np.ndarray, alpha: float) -> list[int]:
        q = self.qhat(alpha)
        return [int(i) for i, p in enumerate(probs) if 1.0 - p <= q]

    def p_value(self, prob_of_class: float) -> float:
        """Conformal p-value of one class: how typical its nonconformity is vs calibration."""
        s = 1.0 - prob_of_class
        idx = int(np.searchsorted(self.sorted_scores, s, side="left"))
        n_ge = self.n - idx
        return (n_ge + 1) / (self.n + 1)

    def empirical_coverage(self, probs: np.ndarray, labels: np.ndarray, alpha: float) -> float:
        q = self.qhat(alpha)
        n = len(labels)
        return float(np.mean(1.0 - probs[np.arange(n), labels] <= q))
