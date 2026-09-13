"""Supplementary diagnostic (added 13 Sep 2026, after the first TableShift runs):
why does the input-space martingale raise false alarms on diabetes_readmission?

Not pre-registered. It was written after the pre-specified 5-stream and the
supplementary 100-stream false-alarm figures on `diabetes_readmission` came
out at 60% and 27% WATCH against Ville's 5%. It changes nothing in
`reliax_core`; it measures where the assumption breaks.

Question: are the conformal p-values of held-out in-distribution rows,
computed against the leave-one-out kNN distances of the calibration set, uniform?
They must be for the martingale's guarantee to hold. Four variants of the
distance are compared on the same rows:

  frozen        StandardScaler fit on the calibration set, LOO calibration scores (reliax_core as shipped)
  scaler_train  StandardScaler fit on the train split, LOO calibration scores
  no_scaler     raw one-hot / numeric features, LOO calibration scores
  split_half    scaler and index fit on half the calibration set; calibration
                scores are the other half scored exactly as test rows are

For each: KS test of the held-out p-values against Uniform(0, 1), mean p,
and the WATCH / ALARM rate over 40 ID streams and 40 OOD streams of 600 rows.
Run on every exported TableShift task, so the pattern (sparse one-hot inputs
vs dense numeric inputs) is visible rather than asserted.

Writes results/expansion/martingale_diag.json.
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import kstest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "eval" / "expansion"))
from reliax_core.martingale import ALARM_THRESHOLD, ConformalMartingale, WATCH_THRESHOLD  # noqa: E402
from reliax_core.ood import KNNOODDetector                                                # noqa: E402
from run_tableshift import CACHE, CAP, ORDER, load_split                                  # noqa: E402

OUT = BASE / "results" / "expansion" / "martingale_diag.json"
K, STREAMS, STREAM_LEN, N_PV = 10, 40, 600, 4000


class Variant:
    def __init__(self, X_fit, X_score=None, scaler="fit", X_scaler=None):
        self.scaler = None
        if scaler == "fit":
            self.scaler = StandardScaler().fit(X_scaler if X_scaler is not None else X_fit)
        Z = self._z(X_fit)
        self.nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
        if X_score is None:                                    # leave-one-out
            self.calib = np.sort(self.nn.kneighbors(Z)[0][:, 1:].mean(axis=1))
        else:                                                  # held-out half scored as test rows
            self.calib = np.sort(self.dist(X_score))

    def _z(self, X):
        return self.scaler.transform(X) if self.scaler is not None else X

    def dist(self, X):
        return self.nn.kneighbors(self._z(np.atleast_2d(X)), n_neighbors=K)[0].mean(axis=1)

    def pvals(self, d, rng):
        c = self.calib
        gt = len(c) - np.searchsorted(c, d, side="right")
        eq = np.searchsorted(c, d, side="right") - np.searchsorted(c, d, side="left")
        return (gt + rng.random(len(d)) * (eq + 1)) / (len(c) + 1)

    def streams(self, X, rng):
        w = a = 0
        for j in range(STREAMS):
            m = ConformalMartingale(self.calib, seed=j)
            rows = rng.choice(len(X), STREAM_LEN, replace=True)
            d = self.dist(X[rows])
            mx = 0.0
            for s in d:
                mx = max(mx, m.update(float(s))["martingale"])
                if mx >= ALARM_THRESHOLD:
                    break
            w += mx >= WATCH_THRESHOLD
            a += mx >= ALARM_THRESHOLD
        return {"watch": w / STREAMS, "alarm": a / STREAMS}


def run_task(task):
    meta = json.loads((CACHE / f"{task}.meta.json").read_text())
    npz = np.load(CACHE / f"{task}.npz")
    rng = np.random.default_rng(0)
    X_tr, _, _ = load_split(npz, meta, "train", rng, CAP["train"])
    X_cal, _, _ = load_split(npz, meta, "validation", rng, CAP["validation"])
    X_id, _, _ = load_split(npz, meta, "id_test", rng, CAP["id_test"])
    X_ood, _, _ = load_split(npz, meta, "ood_test", rng, CAP["ood_test"])
    perm = rng.permutation(len(X_cal))
    h = len(X_cal) // 2
    variants = {
        "frozen": Variant(X_cal),
        "scaler_train": Variant(X_cal, scaler="fit", X_scaler=X_tr),
        "no_scaler": Variant(X_cal, scaler=None),
        "split_half": Variant(X_cal[perm[:h]], X_score=X_cal[perm[h:]]),
    }
    binary = int(np.sum([set(np.unique(X_cal[:, j])) <= {0.0, 1.0} for j in range(X_cal.shape[1])]))
    out = {"n_features": X_cal.shape[1], "n_binary_features": binary, "n_cal": len(X_cal),
           "n_id_test": len(X_id), "variants": {}}
    sub = rng.choice(len(X_id), min(N_PV, len(X_id)), replace=False)
    for name, v in variants.items():
        p = v.pvals(v.dist(X_id[sub]), np.random.default_rng(1))
        out["variants"][name] = {
            "ks_p_uniform": float(kstest(p, "uniform").pvalue), "mean_p": float(p.mean()),
            "p_below_0.05": float(np.mean(p < 0.05)),
            "id_streams": v.streams(X_id, np.random.default_rng(2)),
            "ood_streams": v.streams(X_ood, np.random.default_rng(3))}
        r = out["variants"][name]
        print(f"  {name:13s} KS p {r['ks_p_uniform']:.4f}  mean p {r['mean_p']:.3f}  "
              f"ID WATCH/ALARM {r['id_streams']['watch']:.0%}/{r['id_streams']['alarm']:.0%}  "
              f"OOD ALARM {r['ood_streams']['alarm']:.0%}", flush=True)
    return out


def main(tasks):
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["_meta"] = {"k": K, "streams": STREAMS, "stream_len": STREAM_LEN, "pvalue_rows": N_PV,
                        "status": "supplementary diagnostic, not pre-registered; added after the first "
                                  "TableShift runs; reliax_core unchanged"}
    for task in tasks or [t for t in ORDER if (CACHE / f"{t}.npz").exists()]:
        print(f"=== {task}", flush=True)
        results[task] = run_task(task)
        OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1:])
