"""Accuracy of the confirmatory model, descriptive only (not part of the bar).

Recomputes, on exactly the splits and seeds of eval/run_confirmatory.py, the
model's accuracy at its 0.5 cut on (a) the i.i.d. control and (b) the calm tier
(reference population, no shift), next to the accuracy of always predicting
"repays" (the majority class). It imports the runner and changes nothing in it.

Usage: .venv/bin/python eval/report_accuracy.py gmsc [homecredit2018 taiwan]
"""
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

import run_confirmatory as RC
import confirmatory_data as CD


def main():
    for name in [a for a in sys.argv[1:] if not a.startswith("--")] or ["gmsc"]:
        ds = CD.LOADERS[name]()
        X, y = ds["X"], ds["y"]
        need = RC.N_TRAIN + RC.N_CAL + RC.N_TEST
        ref_pool = np.flatnonzero(ds["reference"])
        out = {"iid": [], "calm": []}
        for seed in RC.SEEDS:
            pool = np.random.default_rng(seed).permutation(len(y))[:need]
            tr, cal, te = pool[:RC.N_TRAIN], pool[RC.N_TRAIN:RC.N_TRAIN + RC.N_CAL], pool[RC.N_TRAIN + RC.N_CAL:]
            C = RC.Reference(X[tr], y[tr], X[cal], y[cal], seed)
            out["iid"].append(score(C, X[te], y[te]))
            rng = np.random.default_rng(seed)                     # same draws as run_dataset, lambda = 0
            ref = rng.permutation(ref_pool)
            tr, cal, rest = ref[:RC.N_TRAIN], ref[RC.N_TRAIN:RC.N_TRAIN + RC.N_CAL], ref[RC.N_TRAIN + RC.N_CAL:]
            rng.permutation(np.flatnonzero(ds["shifted"]))
            R = RC.Reference(X[tr], y[tr], X[cal], y[cal], seed)
            te = rng.permutation(rest[:RC.N_TEST])
            out["calm"].append(score(R, X[te], y[te]))
        print(ds["title"])
        for k, label in (("iid", "random split of everyone"), ("calm", "reference population (calm)")):
            a = np.array(out[k])
            print(f"  {label:30s} accuracy {a[:, 0].mean():.4f} ± {a[:, 0].std():.4f} | always 'repays' {a[:, 1].mean():.4f} | "
                  f"AUC {a[:, 2].mean():.4f} | default rate {a[:, 3].mean():.4f} | defaulters caught at 0.5 {a[:, 4].mean():.4f}")


def score(R, X, y):
    p = R.model.predict_proba(R.impute(X))[:, 1]
    pred = (p >= 0.5).astype(int)
    return [(pred == y).mean(), (y == 0).mean(), roc_auc_score(y, p), y.mean(), pred[y == 1].mean() if y.any() else np.nan]


if __name__ == "__main__":
    main()
