"""Amendment 4 (EXPLORATORY): Reliax at the lender's cut-off.

Three questions from the October 2026 vision note "Credit cut-offs and Reliax",
asked on the production-grade model of Amendment 3 (eval/run_production.py):

  A. Routing at the lender's cut-off PD* = M / (M + L). On the same test set,
     four ways of deciding:
       P0  the lender alone: approve below PD*, decline above, nobody reviews
       P1  Reliax today: the conformal set at alpha 0.05; a single label is
           decided automatically, both labels go to REVIEW
       P2  the Venn-Abers bracket against PD*: approve if p1 < PD*, decline if
           p0 > PD*, REVIEW only when the bracket straddles PD*
       P3  P2 with the bracket calibrated within reliability classes: thirds of
           the calibration set by kNN distance (near, middle, far); each test
           case uses its own class's bracket
     Reported: automatic approvals, automatic declines and REVIEW as shares;
     the default rate among automatic approvals; the good-customer share among
     automatic declines; the margin of automatic decisions per 1,000
     applications (+M for an approved good customer, -L for an approved
     default); and, for P2 and P3, the default rate among automatic approvals
     in each reliability class.
  B. The guaranteed cut-off (reliax-core 0.4.0, guaranteed_cutoff, delta 0.05)
     against the naive one (the loosest calibration cut-off whose observed
     default rate meets the target), at targets of 0.5 and 0.75 times the
     calibration base rate. Within each seed, 20 random re-splits of the
     calibration and test rows (the model fixed): the share of approvals, the
     default rate among them on the test rows, and how often it exceeds the
     target.
  C. Both, under the moderate shift of Amendment 1 (25% of test rows from the
     shifted population, the model and calibration from the reference one),
     where exchangeability fails and no guarantee is claimed: how far they move.

Lender economics, fixed here: the note's worked example (M = 600, L = 5,000,
PD* = 0.107) and a thinner margin (M = 300, L = 5,000, PD* = 0.057).
Model, splits, sizes and seeds: those of run_production.py (i.i.d. split of
the whole population for A and B; its reference model for C).

Usage: .venv/bin/python eval/run_cutoff.py [taiwan gmsc homecredit2018] [--selftest]
Writes results/cutoff/<dataset>.json and results/cutoff/summary.json
"""
import json
import sys
import time

import numpy as np

import run_confirmatory as RC
import run_production as RP
import production_data as PD
from reliax_core.venn_abers import VennAbersCalibrator
from vendor import reliax_cutoff as CUT

try:                                                                   # the vendored copy must match the package
    from reliax_core import cutoff as _installed
    import inspect
    assert inspect.getsource(_installed.guaranteed_cutoff) == inspect.getsource(CUT.guaranteed_cutoff)
except ImportError:
    pass

OUT = RC.BASE / "results" / ("cutoff_selftest" if RC.SELFTEST else "cutoff")
LENDERS = {"worked_example": (600.0, 5000.0), "thin_margin": (300.0, 5000.0)}
TARGET_X = (0.5, 0.75)
DELTA = 0.05
RESPLITS = 20
GRID = 401
CLASSES = ("near", "middle", "far")


class Bracket:
    """Venn-Abers bracket of the model's PD, precomputed on a grid of scores and interpolated."""

    def __init__(self, p_cal, y_cal):
        va = VennAbersCalibrator(p_cal, y_cal)
        self.g = np.linspace(0.0, 1.0, GRID)
        iv = [va.interval(float(s)) for s in self.g]
        self.p0 = np.array([v["p0"] for v in iv])
        self.p1 = np.array([v["p1"] for v in iv])

    def __call__(self, p):
        return np.interp(p, self.g, self.p0), np.interp(p, self.g, self.p1)


def cal_distances(R, X_cal):
    """Each calibration row's leave-one-out kNN distance (the reference KNNOODDetector computes the same, sorted)."""
    Z = R.ood.scaler.transform(R.ood_inputs(X_cal))
    d, _ = R.ood.nn.kneighbors(Z, n_neighbors=RC.K + 1)
    return d[:, 1:].mean(axis=1)


def test_distances(R, X):
    d, _ = R.ood.nn.kneighbors(R.ood.scaler.transform(R.ood_inputs(X)), n_neighbors=RC.K)
    return d.mean(axis=1)


def decide(name, p, y, sets_single, label, b0, b1, pd_star, M, L, cls):
    if name == "P0":
        approve, decline = p < pd_star, p >= pd_star
    elif name == "P1":
        approve, decline = sets_single & (label == 0), sets_single & (label == 1)
    else:
        approve, decline = b1 < pd_star, b0 > pd_star
    review = ~(approve | decline)
    n = len(y)
    out = {"approve": float(approve.mean()), "decline": float(decline.mean()), "review": float(review.mean()),
           "default_among_approved": float(y[approve].mean()) if approve.any() else None,
           "good_among_declined": float((1 - y[decline]).mean()) if decline.any() else None,
           "margin_per_1000": float(1000.0 * (M * ((y == 0) & approve).sum() - L * ((y == 1) & approve).sum()) / n)}
    if name in ("P2", "P3"):
        out["default_among_approved_by_class"] = {c: (float(y[approve & (cls == i)].mean()) if (approve & (cls == i)).any() else None)
                                                  for i, c in enumerate(CLASSES)}
        out["approve_by_class"] = {c: float(approve[cls == i].mean()) if (cls == i).any() else None for i, c in enumerate(CLASSES)}
    return out


def naive_cutoff(p, y, target):
    o = np.argsort(p, kind="stable")
    rate = np.cumsum(y[o]) / np.arange(1, len(p) + 1)
    ok = np.flatnonzero(rate <= target)
    return float(p[o][ok[-1]]) if len(ok) else None


def routing(R, X_cal, y_cal, X_te, y_te):
    """Experiment A on one calibration / test pair."""
    p_cal = R.model.predict_proba(X_cal)[:, 1]
    probs = R.model.predict_proba(X_te)
    p = probs[:, 1]
    sets = (1.0 - probs <= R.qhat)
    single = sets.sum(axis=1) == 1
    label = np.where(sets[:, 1] & ~sets[:, 0], 1, 0)
    marg = Bracket(p_cal, y_cal)
    m0, m1 = marg(p)
    dc = cal_distances(R, X_cal)
    cuts = np.quantile(dc, [1 / 3, 2 / 3])
    cls_cal = np.digitize(dc, cuts)
    cls = np.digitize(test_distances(R, X_te), cuts)
    c0, c1 = m0.copy(), m1.copy()
    for i in range(3):
        k = cls_cal == i
        bi = Bracket(p_cal[k], y_cal[k])
        t = cls == i
        if t.any():
            c0[t], c1[t] = bi(p[t])
    out = {}
    for lender, (M, L) in LENDERS.items():
        pd_star = M / (M + L)
        out[lender] = {"pd_star": pd_star}
        for name, (b0, b1) in {"P0": (m0, m1), "P1": (m0, m1), "P2": (m0, m1), "P3": (c0, c1)}.items():
            out[lender][name] = decide(name, p, y_te, single, label, b0, b1, pd_star, M, L, cls)
    out["class_share_test"] = {c: float((cls == i).mean()) for i, c in enumerate(CLASSES)}
    out["default_rate_by_class_test"] = {c: float(y_te[cls == i].mean()) if (cls == i).any() else None for i, c in enumerate(CLASSES)}
    out["bracket_width_median"] = {"marginal": float(np.median(m1 - m0)), "by_class": float(np.median(c1 - c0))}
    return out


def cutoffs(R, X_pool, y_pool, n_cal, seed):
    """Experiment B: the model fixed, the calibration and test rows re-drawn RESPLITS times."""
    p_all = R.model.predict_proba(X_pool)[:, 1]
    base = float(y_pool.mean())
    res = {}
    for x in TARGET_X:
        target = x * base
        rows = []
        for r in range(RESPLITS):
            idx = np.random.default_rng(1000 * seed + r).permutation(len(y_pool))
            c, t = idx[:n_cal], idx[n_cal:]
            g = CUT.guaranteed_cutoff(p_all[c], y_pool[c], target, DELTA)
            nv = naive_cutoff(p_all[c], y_pool[c], target)
            row = {}
            for name, cut in (("guaranteed", g["cutoff"]), ("naive", nv)):
                if cut is None:
                    row[name] = {"approve": 0.0, "default_among_approved": None, "above_target": False}
                    continue
                a = p_all[t] <= cut
                rate = float(y_pool[t][a].mean()) if a.any() else None
                row[name] = {"approve": float(a.mean()), "default_among_approved": rate,
                             "above_target": bool(rate is not None and rate > target)}
            rows.append(row)
        res[str(x)] = {"target": target, **{name: {
            "approve": RC.ms([r[name]["approve"] for r in rows]),
            "default_among_approved": RC.ms([r[name]["default_among_approved"] if r[name]["default_among_approved"] is not None else np.nan for r in rows]),
            "above_target_share": float(np.mean([r[name]["above_target"] for r in rows])),
            "no_cutoff_share": float(np.mean([r[name]["default_among_approved"] is None for r in rows]))} for name in ("guaranteed", "naive")}}
    return res


def shifted(R, X_cal, y_cal, X_te, y_te):
    """Experiment C: routing and the guaranteed cut-off when 25% of test rows come from the shifted population."""
    out = {"routing": routing(R, X_cal, y_cal, X_te, y_te), "cutoff": {}}
    p_cal, p_te = R.model.predict_proba(X_cal)[:, 1], R.model.predict_proba(X_te)[:, 1]
    for x in TARGET_X:
        target = x * float(y_cal.mean())
        g = CUT.guaranteed_cutoff(p_cal, y_cal, target, DELTA)
        a = p_te <= g["cutoff"] if g["cutoff"] is not None else np.zeros(len(p_te), bool)
        out["cutoff"][str(x)] = {"target": target, "approve": float(a.mean()),
                                 "default_among_approved": float(y_te[a].mean()) if a.any() else None}
    return out


def mean_tree(rows):
    """Mean over seeds of nested dicts of numbers (None and NaN ignored)."""
    if isinstance(rows[0], dict):
        return {k: mean_tree([r[k] for r in rows]) for k in rows[0]}
    vals = [v for v in rows if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None
    if isinstance(vals[0], bool):
        return float(np.mean(vals))
    return round(float(np.mean(vals)), 4)


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
    per = {"A": [], "B": [], "C": []}
    for seed in RC.SEEDS:
        pool = np.random.default_rng(seed).permutation(len(y))
        cal, te, tr = pool[:n_cal], pool[n_cal:n_cal + N], pool[n_cal + N:]
        R = RP.Production(X[tr], y[tr], X[cal], y[cal], cat, iid_params, seed)
        per["A"].append(routing(R, X[cal], y[cal], X[te], y[te]))
        both = np.concatenate([cal, te])
        per["B"].append(cutoffs(R, X[both], y[both], n_cal, seed))
        rng = np.random.default_rng(seed)                                  # the reference draws of run_production.py
        ref = rng.permutation(ref_pool)
        rcal, rest, rtr = ref[:n_cal], ref[n_cal:n_cal + N], ref[n_cal + N:]
        sh = rng.permutation(shift_pool)
        Rr = RP.Production(X[rtr], y[rtr], X[rcal], y[rcal], cat, ref_params, seed)
        n_sh = int(round(0.25 * N))
        i_te = rng.permutation(np.concatenate([rest[:N - n_sh], sh[:n_sh]]))
        per["C"].append(shifted(Rr, X[rcal], y[rcal], X[i_te], y[i_te]))
        a = per["A"][-1]["worked_example"]
        print(f"  {name} seed {seed}: PD* {a['pd_star']:.3f} | review P1 {a['P1']['review']:.1%} P2 {a['P2']['review']:.1%} P3 {a['P3']['review']:.1%} | "
              f"default among approved P0 {a['P0']['default_among_approved']:.3f} P2 {a['P2']['default_among_approved']:.3f} "
              f"P3 {a['P3']['default_among_approved']:.3f} | margin/1000 P0 {a['P0']['margin_per_1000']:.0f} P2 {a['P2']['margin_per_1000']:.0f} "
              f"P3 {a['P3']['margin_per_1000']:.0f}", flush=True)
    res = {"meta": {"dataset": name, "title": ds["title"], "status": "exploratory (Amendment 4)", "seeds": list(RC.SEEDS),
                    "n_cal": n_cal, "n_test": N, "lenders": {k: {"M": m, "L": l, "pd_star": m / (m + l)} for k, (m, l) in LENDERS.items()},
                    "targets_x_base_rate": list(TARGET_X), "delta": DELTA, "resplits": RESPLITS,
                    "params": {"iid": iid_params, "reference": ref_params}, "selftest": RC.SELFTEST},
           "A_routing": mean_tree(per["A"]), "B_cutoff": mean_tree(per["B"]), "C_shift": mean_tree(per["C"]),
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
        summary[name] = {k: res[k] for k in ("A_routing", "B_cutoff", "C_shift")}
        print(f"[{name}] done in {res['meta']['seconds']} s", flush=True)
    path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {path.relative_to(RC.BASE)}")


if __name__ == "__main__":
    main()
