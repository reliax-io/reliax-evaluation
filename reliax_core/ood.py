"""kNN out-of-distribution detector.

Distance of an input to its k nearest calibration points, in standardized feature
space (optionally after a transform such as PCA for vision), converted to a
percentile against the calibration set's own leave-one-out distances.
Heuristic warning signal, not a proof (deck p.6).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FLAG_PERCENTILE = 99.0      # ood_flag above this
EXTREME_PERCENTILE = 99.9   # envelope validity trips above this


class KNNOODDetector:
    def __init__(self, k: int = 10, transform=None):
        self.k = k
        self.transform = transform  # optional callable X -> features (e.g. fitted PCA)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        Z = self.transform(X) if self.transform else X
        Z = self.scaler.fit_transform(Z)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)
        dists, _ = self.nn.kneighbors(Z)
        self.calib_dists = np.sort(dists[:, 1:].mean(axis=1))  # drop self-distance
        return self

    def distance(self, x: np.ndarray) -> float:
        """Raw mean kNN distance in standardized space - the label-free
        nonconformity score fed to the conformal test martingale."""
        z = np.atleast_2d(x)
        if self.transform:
            z = self.transform(z)
        z = self.scaler.transform(z)
        dists, _ = self.nn.kneighbors(z, n_neighbors=self.k)
        return float(dists.mean())

    def percentile(self, x: np.ndarray) -> float:
        z = np.atleast_2d(x)
        if self.transform:
            z = self.transform(z)
        z = self.scaler.transform(z)
        dists, _ = self.nn.kneighbors(z, n_neighbors=self.k)
        d = float(dists.mean())
        idx = int(np.searchsorted(self.calib_dists, d, side="right"))
        return 100.0 * idx / len(self.calib_dists)

    def percentiles_batch(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform(X) if self.transform else X
        Z = self.scaler.transform(Z)
        dists, _ = self.nn.kneighbors(Z, n_neighbors=self.k)
        d = dists.mean(axis=1)
        idx = np.searchsorted(self.calib_dists, d, side="right")
        return 100.0 * idx / len(self.calib_dists)

    def assess(self, x: np.ndarray) -> dict:
        pct = self.percentile(x)
        return {
            "percentile": round(pct, 1),
            "flag": pct > FLAG_PERCENTILE,
            "extreme": pct > EXTREME_PERCENTILE,
        }
