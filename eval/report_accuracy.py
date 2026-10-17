"""Accuracy of the confirmatory model, descriptive only (not part of the bar).

Recomputes, on exactly the splits and seeds of eval/run_confirmatory.py, the
model's accuracy, precision and recall at its 0.5 cut, and when the
10% it rates riskiest are flagged, on (a) the i.i.d. control and (b) the calm tier
(reference population, no shift), next to the accuracy of always predicting
"repays" (the majority class). It imports the runner and changes nothing in it.

Usage: .venv/bin/python eval/report_accuracy.py gmsc [homecredit2018 taiwan]
"""
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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
            m = a.mean(axis=0)
            print(f"  {label:30s} accuracy {m[0]:.4f} ± {a[:, 0].std():.4f} | always 'repays' {m[1]:.4f} | "
                  f"AUC {m[2]:.4f} | default rate {m[3]:.4f} | defaulters caught at 0.5 {m[4]:.4f}")
            print(f"  {'':30s} at 0.5: precision {m[5]:.4f} recall {m[4]:.4f} F1 {m[6]:.4f} | "
                  f"riskiest 10% flagged: precision {m[7]:.4f} recall {m[8]:.4f} | PR-AUC {m[9]:.4f} ({m[9] / m[3]:.2f}x the default rate)")


def score(R, X, y):
    p = R.model.predict_proba(R.impute(X))[:, 1]
    pred = (p >= 0.5).astype(int)
    rec = pred[y == 1].mean() if y.any() else np.nan
    prec = y[pred == 1].mean() if pred.any() else np.nan
    f1 = 2 * prec * rec / (prec + rec) if pred.any() and prec + rec > 0 else 0.0
    top = np.argsort(-p, kind="stable")[:int(round(0.10 * len(y)))]          # the 10% the model rates riskiest
    return [(pred == y).mean(), (y == 0).mean(), roc_auc_score(y, p), y.mean(), rec, prec, f1,
            y[top].mean(), y[top].sum() / max(y.sum(), 1), average_precision_score(y, p)]


if __name__ == "__main__":
    main()
