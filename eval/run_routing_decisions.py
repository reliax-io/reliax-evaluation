"""How many mistakes does certified routing avoid? Conformal set size as the
REVIEW rule, on real Taiwan data, in the calm and the shifted regimes.

Rule: at miscoverage alpha, an applicant whose prediction set holds both
labels {repay, default} is routed to REVIEW; a singleton set is auto-acted.
Mistakes: model errors, and bad approvals (auto-approved, then defaulted).
Comparators at the SAME referral rate: random referral, and confidence
referral (refer the least confident decisions, the lender's default option).

Regimes from eval/run_shift_bench.py: iid (control), util, pay0 (severe).
Usage: .venv/bin/python eval/run_routing_decisions.py
Writes results/routing_decisions.json
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reliax_core.conformal import ConformalCalibrator  # noqa: E402
import data as D                                        # noqa: E402
import run_shift_bench as SB                            # noqa: E402

RESULTS = BASE / "results"
ALPHAS = (0.05, 0.10, 0.20)


def agg(v):
    a = np.asarray(v, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


def evaluate(probs_te, y_te, conf, alpha, seed):
    qhat = conf.qhat(alpha)
    sizes = (1.0 - probs_te <= qhat).sum(axis=1)
    pred = probs_te.argmax(axis=1)
    wrong = pred != y_te
    approve = pred == 0
    bad = approve & (y_te == 1)
    review = sizes > 1
    n = len(y_te)
    r = review.mean()
    k = int(round(r * n))
    msp = probs_te.max(axis=1)
    conf_ref = np.zeros(n, bool)
    conf_ref[np.argsort(msp)[:k]] = True
    rng = np.random.default_rng(seed)
    rand_ref = np.zeros(n, bool)
    rand_ref[rng.choice(n, k, replace=False)] = True
    def caught(mask, what):
        return float((mask & what).sum() / max(what.sum(), 1))
    return {
        "coverage": float(np.mean([(y in np.flatnonzero(1.0 - probs_te[i] <= qhat)) for i, y in enumerate(y_te)])),
        "review_rate": float(r),
        "error_rate_all": float(wrong.mean()),
        "error_rate_auto": float(wrong[~review].mean()) if (~review).sum() else np.nan,
        "bad_approval_rate_all_approvals": float(bad.sum() / max(approve.sum(), 1)),
        "bad_approval_rate_auto_approvals": float((bad & ~review).sum() / max((approve & ~review).sum(), 1)),
        "errors_caught_review": caught(review, wrong), "errors_caught_random": caught(rand_ref, wrong),
        "errors_caught_confidence": caught(conf_ref, wrong),
        "bad_caught_review": caught(review, bad), "bad_caught_random": caught(rand_ref, bad),
        "bad_caught_confidence": caught(conf_ref, bad),
        "bad_per_1000_all": 1000 * bad.mean(), "bad_per_1000_auto": 1000 * (bad & ~review).mean(),
    }


def main():
    ds = D.load_taiwan()
    X, y, names = ds["X"], ds["y"], ds["feature_names"]
    regs = SB.regimes(X, y, names)
    out = {"meta": {"alphas": ALPHAS, "dataset": "taiwan", "seeds": 5}}
    for reg in ("iid", "util", "pay0"):
        ref_mask, test_mask, desc = regs[reg]
        rows = {str(a): [] for a in ALPHAS}
        for seed in range(5):
            rng = np.random.default_rng(seed)
            idx_tr, idx_cal, idx_te = SB.draw(rng, ref_mask, test_mask, disjoint=(reg != "iid"))
            model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X[idx_tr], y[idx_tr])
            probs_cal, probs_te = model.predict_proba(X[idx_cal]), model.predict_proba(X[idx_te])
            conf = ConformalCalibrator(probs_cal, y[idx_cal])
            for a in ALPHAS:
                rows[str(a)].append(evaluate(probs_te, y[idx_te], conf, a, seed))
        out[reg] = {"description": desc, **{a: {k: agg([r[k] for r in rs]) for k in rs[0]} for a, rs in rows.items()}}
        for a in ALPHAS:
            v = out[reg][str(a)]
            print(f"  {reg} alpha={a}: cov {v['coverage']['mean']:.3f} review {v['review_rate']['mean']:.1%} | "
                  f"err all {v['error_rate_all']['mean']:.3f} auto {v['error_rate_auto']['mean']:.3f} | "
                  f"bad/1000 all {v['bad_per_1000_all']['mean']:.0f} auto {v['bad_per_1000_auto']['mean']:.0f} | "
                  f"errors caught review {v['errors_caught_review']['mean']:.2f} rand {v['errors_caught_random']['mean']:.2f} conf {v['errors_caught_confidence']['mean']:.2f} | "
                  f"bad caught review {v['bad_caught_review']['mean']:.2f} rand {v['bad_caught_random']['mean']:.2f} conf {v['bad_caught_confidence']['mean']:.2f}")
    (RESULTS / "routing_decisions.json").write_text(json.dumps(out, indent=1))
    print("wrote results/routing_decisions.json")


if __name__ == "__main__":
    main()
