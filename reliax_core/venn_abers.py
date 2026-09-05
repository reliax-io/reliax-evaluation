"""Inductive Venn-Abers predictor (IVAP) for binary PD calibration.

Given calibration pairs (s_i, y_i) where s_i is the base model's raw p(default),
a test score s gets TWO isotonic calibrations: one with (s, 0) appended, one with
(s, 1) appended. The pair (p0, p1) brackets the calibrated probability:

    p0 = g0(s)  with g0 = isotonic fit on cal + {(s, 0)}
    p1 = g1(s)  with g1 = isotonic fit on cal + {(s, 1)}

Guarantee (Vovk & Petej, 2014): one of the two predictors is perfectly calibrated,
so the true calibrated PD lies in [p0, p1] in the Venn sense. The interval WIDTH
is itself a reliability signal: wide = the calibration data cannot pin the PD down.
Merged point estimate p = p1 / (1 - p0 + p1) minimises log-loss regret.

O(n log n) per query via sklearn isotonic; fine at demo calibration sizes (~2-3k).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class VennAbersCalibrator:
    def __init__(self, calib_scores: np.ndarray, calib_labels: np.ndarray):
        # jitter identical scores minimally so isotonic pooling is stable
        self.s = np.asarray(calib_scores, dtype=float)
        self.y = np.asarray(calib_labels, dtype=float)
        self.n = len(self.s)

    def interval(self, score: float) -> dict:
        p = []
        for hypothetical in (0.0, 1.0):
            xs = np.append(self.s, score)
            ys = np.append(self.y, hypothetical)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(xs, ys)
            p.append(float(iso.predict([score])[0]))
        p0, p1 = min(p), max(p)
        merged = p1 / (1.0 - p0 + p1) if (1.0 - p0 + p1) > 0 else p1
        return {
            "p0": round(p0, 4),
            "p1": round(p1, 4),
            "point": round(merged, 4),
            "width": round(p1 - p0, 4),
            "method": "inductive-venn-abers",
        }

    def widths_batch(self, scores: np.ndarray) -> np.ndarray:
        """Interval widths over a batch (used for scenario search at startup)."""
        return np.array([self.interval(float(s))["width"] for s in scores])
