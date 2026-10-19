"""Amendment 5 (EXPLORATORY): approve, review and decline zones guaranteed on both sides.

The lender keeps its cut-off PD* = M / (M + L). Reliax's three_zones
(reliax-core 0.4.0, carried in eval/vendor/) splits the calibration scores into
equal-count slices and guarantees, jointly at 1 - delta (delta = 0.05), that
every approved slice defaults at most PD* and every declined slice at least
PD*; the slices neither side can show go to a person.

Compared on the same test rows as eval/run_cutoff.py (production-grade model,
i.i.d. split; seeds 0 to 4):
  P0   the lender alone: approve below PD*, decline above
  Z20  three zones on 20 slices (5% of the calibration cases each), primary
  Z10  three zones on 10 slices, secondary
Reported per lender (M = 600 and 300, L = 5,000): the shares approved, reviewed
and declined automatically; the default rate among automatic approvals and
among automatic declines; the good-customer share among declines; the default
rate among the reviewed cases; the margin of automatic decisions per 1,000
applications; the realised default rate in the riskiest approved slice and the
safest declined slice. Validity over 20 re-splits of the calibration and test
rows per seed (the model fixed): how often the realised default rate among
approvals exceeds PD*, and among declines falls below it. Under the moderate
shift (25% of test rows from the shifted population, the reference model and
calibration): the same figures, where no guarantee is claimed.

Usage: .venv/bin/python eval/run_zones.py [taiwan gmsc homecredit2018] [--selftest]
Writes results/zones/<dataset>.json and results/zones/summary.json
"""
import json
import sys
import time

import numpy as np

import run_confirmatory as RC
import run_production as RP
import production_data as PD
from run_cutoff import LENDERS, RESPLITS, mean_tree
from vendor import reliax_cutoff as CUT

try:                                                                   # the vendored copy must match the package
    from reliax_core import cutoff as _installed
    import inspect
    assert inspect.getsource(_installed.three_zones) == inspect.getsource(CUT.three_zones)
except (ImportError, AttributeError):
    pass

OUT = RC.BASE / "results" / ("zones_selftest" if RC.SELFTEST else "zones")
DELTA = 0.05
SLICES = {"Z20": 20, "Z10": 10}


def route(p, z):
    a = p <= z["approve_to"] if z["approve_to"] is not None else np.zeros(len(p), bool)
    d = p > z["decline_from"] if z["decline_from"] is not None else np.zeros(len(p), bool)
    return a, d & ~a


def figures(p, y, approve, decline, M, L, z=None):
    review = ~(approve | decline)
    n = len(y)
    out = {"approve": float(approve.mean()), "review": float(review.mean()), "decline": float(decline.mean()),
           "default_among_approved": float(y[approve].mean()) if approve.any() else None,
           "default_among_declined": float(y[decline].mean()) if decline.any() else None,
           "good_among_declined": float((1 - y[decline]).mean()) if decline.any() else None,
           "default_among_reviewed": float(y[review].mean()) if review.any() else None,
           "margin_per_1000": float(1000.0 * (M * ((y == 0) & approve).sum() - L * ((y == 1) & approve).sum()) / n)}
    if z is not None:                                          # the slices at the edge of each guaranteed zone
        rows = z["slices_table"]
        last_a = max((i for i, r in enumerate(rows) if r["zone"] == "approve"), default=None)
        first_d = min((i for i, r in enumerate(rows) if r["zone"] == "decline"), default=None)
        for key, i in (("edge_approved", last_a), ("edge_declined", first_d)):
            if i is None:
                out[key] = None
                continue
            lo = -np.inf if rows[i]["from"] is None else rows[i]["from"]
            hi = np.inf if rows[i]["to"] is None else rows[i]["to"]
            m = (p > lo) & (p <= hi)
            out[key] = float(y[m].mean()) if m.any() else None
        out["calibration"] = {"approved_upper_bound": z["approved"]["upper_bound"],
                              "declined_lower_bound": z["declined"]["lower_bound"]}
    return out


def compare(p_cal, y_cal, p_te, y_te):
    out = {}
    for lender, (M, L) in LENDERS.items():
        pd_star = M / (M + L)
        a0 = p_te < pd_star
        out[lender] = {"pd_star": pd_star, "P0": figures(p_te, y_te, a0, ~a0, M, L)}
        for name, S in SLICES.items():
            z = CUT.three_zones(p_cal, y_cal, pd_star, DELTA, slices=S)
            a, d = route(p_te, z)
            out[lender][name] = figures(p_te, y_te, a, d, M, L, z)
    return out


def validity(p_pool, y_pool, n_cal, seed):
    res = {}
    for lender, (M, L) in LENDERS.items():
        pd_star = M / (M + L)
        res[lender] = {}
        for name, S in SLICES.items():
            above, below = [], []
            for r in range(RESPLITS):
                idx = np.random.default_rng(1000 * seed + r).permutation(len(y_pool))
                c, t = idx[:n_cal], idx[n_cal:]
                z = CUT.three_zones(p_pool[c], y_pool[c], pd_star, DELTA, slices=S)
                a, d = route(p_pool[t], z)
                above.append(bool(a.any() and y_pool[t][a].mean() > pd_star))
                below.append(bool(d.any() and y_pool[t][d].mean() < pd_star))
            res[lender][name] = {"approvals_above_pd_star": float(np.mean(above)),
                                 "declines_below_pd_star": float(np.mean(below))}
    return res


def run_dataset(name, ds):
    t0 = time.time()
    X, y, cat = ds["X"], ds["y"], ds["categorical"]
    n_cal = RP.N_CAL.get(name, 10_000)
    N = RP.N_TEST
    pool0 = np.random.default_rng(0).permutation(len(y))
    iid_params, _ = RP.tune(X[pool0[n_cal + N:]], y[pool0[n_cal + N:]], cat)
    ref_pool, shift_pool = np.flatnonzero(ds["reference"]), np.flatnonzero(ds["shifted"])
    ref0 = np.random.default_rng(0).permutation(ref_pool)
    ref_params, _ = RP.tune(X[ref0[n_cal + N:]], y[ref0[n_cal + N:]], cat)
    per = {"iid": [], "validity": [], "shift": []}
    for seed in RC.SEEDS:
        pool = np.random.default_rng(seed).permutation(len(y))
        cal, te, tr = pool[:n_cal], pool[n_cal:n_cal + N], pool[n_cal + N:]
        R = RP.Production(X[tr], y[tr], X[cal], y[cal], cat, iid_params, seed)
        p_cal, p_te = R.model.predict_proba(X[cal])[:, 1], R.model.predict_proba(X[te])[:, 1]
        per["iid"].append(compare(p_cal, y[cal], p_te, y[te]))
        both = np.concatenate([cal, te])
        per["validity"].append(validity(R.model.predict_proba(X[both])[:, 1], y[both], n_cal, seed))
        rng = np.random.default_rng(seed)                                  # the reference draws of run_production.py
        ref = rng.permutation(ref_pool)
        rcal, rest, rtr = ref[:n_cal], ref[n_cal:n_cal + N], ref[n_cal + N:]
        sh = rng.permutation(shift_pool)
        Rr = RP.Production(X[rtr], y[rtr], X[rcal], y[rcal], cat, ref_params, seed)
        n_sh = int(round(0.25 * N))
        i_te = rng.permutation(np.concatenate([rest[:N - n_sh], sh[:n_sh]]))
        per["shift"].append(compare(Rr.model.predict_proba(X[rcal])[:, 1], y[rcal], Rr.model.predict_proba(X[i_te])[:, 1], y[i_te]))
        w = per["iid"][-1]["worked_example"]
        print(f"  {name} seed {seed}: PD* {w['pd_star']:.3f} | approve/review/decline P0 {w['P0']['approve']:.1%}/0/{w['P0']['decline']:.1%} "
              f"Z20 {w['Z20']['approve']:.1%}/{w['Z20']['review']:.1%}/{w['Z20']['decline']:.1%} | default approved P0 "
              f"{w['P0']['default_among_approved']:.3f} Z20 {w['Z20']['default_among_approved']} | margin/1000 P0 {w['P0']['margin_per_1000']:.0f} "
              f"Z20 {w['Z20']['margin_per_1000']:.0f}", flush=True)
    res = {"meta": {"dataset": name, "title": ds["title"], "status": "exploratory (Amendment 5)", "seeds": list(RC.SEEDS),
                    "n_cal": n_cal, "n_test": N, "delta": DELTA, "slices": SLICES, "resplits": RESPLITS,
                    "lenders": {k: {"M": m, "L": l, "pd_star": m / (m + l)} for k, (m, l) in LENDERS.items()},
                    "params": {"iid": iid_params, "reference": ref_params}, "selftest": RC.SELFTEST},
           "iid": mean_tree(per["iid"]), "validity": mean_tree(per["validity"]), "shift": mean_tree(per["shift"]),
           "per_seed": per}
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaders = RP.selftest_loaders() if RC.SELFTEST else PD.LOADERS
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(loaders)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        print(f"[{name}] loading", flush=True)
        res = run_dataset(name, loaders[name]())
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        summary[name] = {k: res[k] for k in ("iid", "validity", "shift")}
        print(f"[{name}] done in {res['meta']['seconds']} s", flush=True)
    path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {path.relative_to(RC.BASE)}")


if __name__ == "__main__":
    main()
