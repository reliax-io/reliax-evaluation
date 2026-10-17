"""Precision, recall and F1 of the paper's model (sections 3 to 5), descriptive only.

Same splits, seeds and model as eval/run_eval.py (50/25/25 stratified, the
paper's HistGradientBoosting), measured on the test split at the 0.5 cut and
for the 10% the model rates riskiest. It changes nothing in results.json.

Usage: .venv/bin/python eval/report_paper_f1.py
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import data as D
import models as MODELS


def main():
    for ds, seeds in ((D.load_taiwan(), range(5)), (D.load_german(), range(10))):
        X, y = ds["X"], ds["y"]
        rows = []
        for seed in seeds:
            idx = np.arange(len(y))
            tr, rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
            _, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=y[rest])
            p = MODELS.make_model(ds["name"], seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            yt = y[te]
            pred = p >= 0.5
            tp = (pred & (yt == 1)).sum()
            prec, rec = tp / max(pred.sum(), 1), tp / max((yt == 1).sum(), 1)
            top = np.argsort(-p, kind="stable")[:int(round(0.1 * len(yt)))]
            rows.append([(pred == yt).mean(), roc_auc_score(yt, p), prec, rec, 2 * prec * rec / max(prec + rec, 1e-12),
                         yt[top].mean(), yt[top].sum() / max(yt.sum(), 1), average_precision_score(yt, p), yt.mean()])
        m, s = np.mean(rows, axis=0), np.std(rows, axis=0)
        print(f"{ds['name']}: accuracy {m[0]:.4f} AUC {m[1]:.4f} | at 0.5: precision {m[2]:.4f} recall {m[3]:.4f} "
              f"F1 {m[4]:.4f} ± {s[4]:.4f} | riskiest 10%: precision {m[5]:.4f} recall {m[6]:.4f} | "
              f"PR-AUC {m[7]:.4f} (default rate {m[8]:.4f})")


if __name__ == "__main__":
    main()
