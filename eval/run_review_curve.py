"""Review rate against the reliability target (EXPLORATORY, descriptive).

How the share of decisions sent to a person moves with the coverage target the
lender sets, on the production-grade model of Amendment 3 (eval/run_production.py):
same data, i.i.d. split, sizes, tuning and seeds 0 to 4. For each seed the model and
the calibration split are fitted once; the conformal threshold is then read at each
target. No shift, no corruption, no bar: it describes a trade-off, it tests nothing.

Targets run from 0.80 to 0.99 in steps of 0.01, starting at the first one above the
share of borrowers who repay: below it, "everyone repays" already meets the target
without any model, and the threshold turns so strict that sets come out empty.

Reported per target, mean and spread over the seeds: coverage on the test rows, the
REVIEW rate (sets that are not a single outcome), split into sets holding both
outcomes and empty sets.

Usage: .venv/bin/python eval/run_review_curve.py [gmsc homecredit2018 taiwan] [--selftest]
Writes results/review_curve/summary.json
"""
import json
import sys

import numpy as np

import run_confirmatory as RC
import run_production as RP
import production_data as PD
from reliax_core.conformal import ConformalCalibrator

OUT = RC.BASE / "results" / ("review_curve_selftest" if RC.SELFTEST else "review_curve")
GRID = [round(0.80 + 0.01 * i, 2) for i in range(20)]


def run_dataset(name, ds):
    X, y, cat = ds["X"], ds["y"], ds["categorical"]
    repay = 1.0 - float(np.mean(y))
    TARGETS = [t for t in GRID if t > repay]
    n_cal = RP.N_CAL.get(name, 10_000)
    N = RP.N_TEST
    pool0 = np.random.default_rng(0).permutation(len(y))
    params, _ = RP.tune(X[pool0[n_cal + N:]], y[pool0[n_cal + N:]], cat)
    rows = {str(t): [] for t in TARGETS}
    for seed in RC.SEEDS:
        pool = np.random.default_rng(seed).permutation(len(y))
        cal, te, tr = pool[:n_cal], pool[n_cal:n_cal + N], pool[n_cal + N:]
        model = RP.fit(X[tr], y[tr], cat, params, seed)
        cc = ConformalCalibrator(model.predict_proba(X[cal]), y[cal])
        probs = model.predict_proba(X[te])
        for t in TARGETS:
            sets = 1.0 - probs <= cc.qhat(1.0 - t)
            size = sets.sum(axis=1)
            rows[str(t)].append({"coverage": float(sets[np.arange(len(te)), y[te]].mean()),
                                 "review": float((size != 1).mean()),
                                 "both": float((size == 2).mean()), "empty": float((size == 0).mean())})
        print(f"  {name} seed {seed}: " + " ".join(f"{t:.2f}->{rows[str(t)][-1]['review']:.1%}" for t in TARGETS), flush=True)
    out = {}
    for t, rs in rows.items():
        out[t] = {k: {"mean": round(float(np.mean([r[k] for r in rs])), 4),
                      "min": round(float(np.min([r[k] for r in rs])), 4),
                      "max": round(float(np.max([r[k] for r in rs])), 4)} for k in rs[0]}
    return {"meta": {"dataset": name, "status": "exploratory, descriptive", "model": "production-grade (Amendment 3)",
                     "repay_share": round(repay, 4), "seeds": list(RC.SEEDS), "n_cal": n_cal, "n_test": N, "params": params, "selftest": RC.SELFTEST},
            "targets": out}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaders = RP.selftest_loaders() if RC.SELFTEST else PD.LOADERS
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(loaders)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        print(f"[{name}] loading", flush=True)
        summary[name] = run_dataset(name, loaders[name]())
    path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {path.relative_to(RC.BASE)}")


if __name__ == "__main__":
    main()
