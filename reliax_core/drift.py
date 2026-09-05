"""PSI / CSI drift monitor - the metric credit model-validation teams already use.

Population Stability Index between the calibration distribution and a rolling
window of recent production inputs, per feature (CSI) and on the model's PD score
(PSI proper):

    PSI = sum_b (q_b - p_b) * ln(q_b / p_b)

with p_b = calibration share in decile bin b, q_b = recent-window share.
Industry reading: < 0.10 stable · 0.10-0.25 moderate shift · > 0.25 major shift.

A descriptive monitor, not a test - the martingale is the theorem; PSI is the
vocabulary credit validators expect on the same screen.
"""
from collections import deque

import numpy as np

N_BINS = 10
MIN_SHARE = 1e-4
WINDOW = 150
MIN_SAMPLES = 40


def _edges(col: np.ndarray) -> np.ndarray:
    e = np.quantile(col, np.linspace(0, 1, N_BINS + 1))
    e[0], e[-1] = -np.inf, np.inf
    return np.unique(e)


def _psi(ref_counts: np.ndarray, cur: np.ndarray, edges: np.ndarray) -> float:
    p = np.maximum(ref_counts / ref_counts.sum(), MIN_SHARE)
    q_counts, _ = np.histogram(cur, bins=edges)
    q = np.maximum(q_counts / max(q_counts.sum(), 1), MIN_SHARE)
    return float(np.sum((q - p) * np.log(q / p)))


def band(psi: float) -> str:
    return "MAJOR" if psi > 0.25 else "MODERATE" if psi > 0.10 else "STABLE"


class PSIMonitor:
    def __init__(self, X_calib: np.ndarray, pd_calib: np.ndarray, feature_names: list[str]):
        self.feature_names = feature_names
        self.f_edges = [_edges(X_calib[:, j]) for j in range(X_calib.shape[1])]
        self.f_ref = [np.histogram(X_calib[:, j], bins=self.f_edges[j])[0]
                      for j in range(X_calib.shape[1])]
        self.s_edges = _edges(pd_calib)
        self.s_ref = np.histogram(pd_calib, bins=self.s_edges)[0]
        self.window_X: deque = deque(maxlen=WINDOW)
        self.window_pd: deque = deque(maxlen=WINDOW)

    def observe(self, x: np.ndarray, pd_score: float):
        self.window_X.append(np.asarray(x, dtype=float))
        self.window_pd.append(float(pd_score))

    def report(self) -> dict:
        n = len(self.window_X)
        if n < MIN_SAMPLES:
            return {"ready": False, "window_n": n, "min_samples": MIN_SAMPLES,
                    "features": [], "score_psi": None}
        W = np.vstack(self.window_X)
        feats = []
        for j, name in enumerate(self.feature_names):
            v = round(_psi(self.f_ref[j], W[:, j], self.f_edges[j]), 4)
            feats.append({"feature": name, "psi": v, "band": band(v)})
        feats.sort(key=lambda d: -d["psi"])
        score_psi = round(_psi(self.s_ref, np.array(self.window_pd), self.s_edges), 4)
        return {
            "ready": True,
            "window_n": n,
            "score_psi": score_psi,
            "score_band": band(score_psi),
            "features": feats,
            "max_feature_psi": feats[0]["psi"] if feats else 0.0,
            "max_feature": feats[0]["feature"] if feats else None,
        }
