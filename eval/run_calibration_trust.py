"""Calibration-trust experiments (the FUSION 2025 method, made usable for credit).

Answers, with numbers written to results/calibration_trust.json:

  S1  M sweep on synthetic data with KNOWN calibration error. Does the global
      disbelief track the true ECE, and what does it converge to as M grows?
      (Propositions 1 and 2 of CALIBRATION_TRUST.md, plus the debiasing constants.)
  S2  Consistency: M_N = N^(1/3) bins, N from 1e3 to 1e6 (Proposition 4).
  S3  Finite-sample bound (Proposition 3): how often the Hoeffding envelope
      contains the population binned ECE, and how wide it is.
  S4  Real credit models (Taiwan default, German credit; the whitepaper's
      HistGradientBoosting models): opinions for the raw, temperature-scaled
      and Venn-Abers-calibrated scorers, per segment x score-bin cells, thin
      cells, the alpha/beta knob, isotonic (Venn-Abers) bins, and the
      per-decision accuracy-mode lookup used as a ranking signal.
  S5  Shift: the opinion fitted on the reference population versus the
      realized calibration on a shifted population (Taiwan payment-delay and
      utilisation splits of eval/run_shift_bench.py). The honest limit.

Usage: .venv/bin/python eval/run_calibration_trust.py [--fast]
"""
import json
import pathlib
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from reliax_core.calibration_trust import (CalibrationTrust, MARGINAL,          # noqa: E402
                                           hoeffding_bound, noise_floor)
from reliax_core.venn_abers import VennAbersCalibrator                         # noqa: E402

import data as D                                                                # noqa: E402
import methods as M                                                             # noqa: E402

RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
FAST = "--fast" in sys.argv
W = 2.0
M_GRID = [2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000]
ESTIMATORS = {
    "chapter (fixed bins, midpoint)":      dict(binning="fixed", representative="mid", debias="none"),
    "quantile bins, mean rep":             dict(binning="quantile", representative="mean", debias="none"),
    "quantile, mean, floor-debiased":      dict(binning="quantile", representative="mean", debias="floor"),
    "quantile, mean, l2-debiased":         dict(binning="quantile", representative="mean", debias="l2"),
}


def agg(v):
    a = np.asarray(v, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 5), "std": round(float(np.nanstd(a)), 5)}


def synth(n, rng, T=1.0):
    """True default probability p ~ Beta(2, 6) (mean 0.25, credit-like); the
    model STATES q = sigmoid(logit(p) / T). T = 1 exactly calibrated, T < 1
    over-confident (q more extreme than p), T > 1 under-confident."""
    p = rng.beta(2, 6, n)
    y = (rng.random(n) < p).astype(int)
    q = 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))
    return q, y, p


def truth(T, rng, n=2_000_000):
    """True L1 calibration error E|p - q| and the M -> inf limit E|Y - q|."""
    q, _, p = synth(n, rng, T)
    return {"ece_true": float(np.mean(np.abs(p - q))),
            "l1_limit": float(np.mean(p + q - 2 * p * q))}


def unshrunk_d(ct):
    """Global disbelief with the prior shrink N/(N+W) removed, so it is an
    estimate of the (debiased) binned ECE."""
    return ct.global_opinion["d"] * (ct.n + W) / ct.n


# --------------------------------------------------------------------------- #
def s1_msweep():
    out = {}
    seeds = 3 if FAST else 5
    for T, name in ((1.0, "calibrated"), (0.6, "over-confident"), (1.8, "under-confident")):
        tr = truth(T, np.random.default_rng(999))
        reg = {"T": T, **tr, "N": {}}
        for N in ((2000, 20000) if not FAST else (2000,)):
            per = {k: {str(m): [] for m in M_GRID} for k in ESTIMATORS}
            for seed in range(seeds):
                q, y, _ = synth(N, np.random.default_rng(seed), T)
                for k, cfg in ESTIMATORS.items():
                    for m in M_GRID:
                        ct = CalibrationTrust(M=m, W=W, **cfg).fit(q, y)
                        per[k][str(m)].append(unshrunk_d(ct))
            reg["N"][str(N)] = {k: {m: agg(v) for m, v in d.items()} for k, d in per.items()}
        out[name] = reg
        print(f"  S1 {name}: ece_true={tr['ece_true']:.4f} l1_limit={tr['l1_limit']:.4f}")
    return out


def s2_consistency():
    out = {}
    T = 1.8
    tr = truth(T, np.random.default_rng(998))
    Ns = [1000, 10000, 100000] + ([] if FAST else [1000000])
    for N in Ns:
        m = max(2, int(round(N ** (1 / 3))))
        row = {"M": m}
        for k in ("quantile bins, mean rep", "quantile, mean, l2-debiased", "chapter (fixed bins, midpoint)"):
            vals = []
            for seed in range(3):
                q, y, _ = synth(N, np.random.default_rng(100 + seed), T)
                ct = CalibrationTrust(M=m, W=W, **ESTIMATORS[k]).fit(q, y)
                vals.append(unshrunk_d(ct))
            row[k] = agg(vals)
            row[k]["abs_error_vs_true"] = round(abs(row[k]["mean"] - tr["ece_true"]), 5)
        out[str(N)] = row
        print(f"  S2 N={N} M={m}: " + ", ".join(f"{k[:18]}={row[k]['mean']:.4f}" for k in row if k != "M"))
    return {"T": T, **tr, "rows": out}


def s3_bound():
    rng = np.random.default_rng(7)
    q, _, p = synth(400000, rng, 1.6)
    edges = np.linspace(0, 1, 11)
    hits, widths, errs = 0, [], []
    trials = 100 if FAST else 500
    for seed in range(trials):
        r = np.random.default_rng(1000 + seed)
        idx = r.choice(len(q), 3000, replace=False)
        y = (r.random(3000) < p[idx]).astype(int)
        ct = CalibrationTrust(M=10, binning="fixed", representative="mean", W=W).fit(q[idx], y)
        pop = 0.0
        for (g, i), c in ct.cells.items():
            m = (q >= edges[i]) & (q < edges[i + 1]) if i < 9 else (q >= edges[9])
            pop += c["n"] / 3000 * abs(p[m].mean() - c["rp"])
        bound = hoeffding_bound([c["n"] for c in ct.cells.values()], delta=0.05)
        err = abs(ct.binned_ece() - pop)
        hits += err <= bound
        widths.append(bound)
        errs.append(err)
    res = {"N": 3000, "M": 10, "delta": 0.05, "trials": trials, "coverage_of_bound": hits / trials,
           "mean_bound": agg(widths), "mean_abs_error": agg(errs)}
    print(f"  S3 bound holds in {hits}/{trials}; mean bound {np.mean(widths):.4f}, mean error {np.mean(errs):.4f}")
    return res


# --------------------------------------------------------------------------- #
def fit_scorers(model, X_cal, y_cal, logit_cal, probs_cal):
    """Split the calibration set: A fits the calibrators, B fits the opinions."""
    idx = np.arange(len(y_cal))
    a, b = train_test_split(idx, test_size=0.5, random_state=0, stratify=y_cal)
    t = M.fit_temperature(logit_cal[a], y_cal[a])
    va = VennAbersCalibrator(probs_cal[a, 1], y_cal[a])
    return t, va, a, b


def scores_for(name, p_raw, logit, t, va):
    if name == "raw":
        return p_raw
    if name == "temperature-scaled":
        return 1 / (1 + np.exp(-logit / t))
    return np.array([va.interval(float(s))["point"] for s in p_raw])


def s4_real(ds, seeds):
    out = {"scorers": {}, "per_decision": {}, "isotonic": {}, "knob": {}, "segments": {}}
    acc_collect = {k: [] for k in ("raw", "temperature-scaled", "venn-abers")}
    seg_attr = "age_band"
    per_dec = {"opinion_belief": {"aurc": [], "capture10": []}, "msp": {"aurc": [], "capture10": []}}
    iso = []
    knob = {"(1,1)": [], "(2,1)": [], "(1,2)": []}
    cells_dump = None
    for seed in range(seeds):
        X, y = ds["X"], ds["y"]
        idx = np.arange(len(y))
        idx_tr, idx_rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
        idx_cal, idx_te = train_test_split(idx_rest, test_size=0.5, random_state=seed, stratify=y[idx_rest])
        model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X[idx_tr], y[idx_tr])
        probs_cal, probs_te = model.predict_proba(X[idx_cal]), model.predict_proba(X[idx_te])
        logit_cal, logit_te = model.decision_function(X[idx_cal]), model.decision_function(X[idx_te])
        y_cal, y_te = y[idx_cal], y[idx_te]
        seg_cal, seg_te = ds["segments"][seg_attr][idx_cal], ds["segments"][seg_attr][idx_te]
        t, va, a, b = fit_scorers(model, X[idx_cal], y_cal, logit_cal, probs_cal)

        for name in acc_collect:
            s_b = scores_for(name, probs_cal[b, 1], logit_cal[b], t, va)
            s_te = scores_for(name, probs_te[:, 1], logit_te, t, va)
            ct = CalibrationTrust(M=10, binning="quantile", representative="mean", evidence="rate",
                                  debias="l2", W=W, min_cell_n=30).fit(s_b, y_cal[b], segments=seg_cal[b])
            rep = ct.report()
            real = ct.realized(s_te, y_te, segments=seg_te)
            row = {"opinion": rep["global"], "binned_ece_calB": rep["binned_ece"],
                   "ece15_test": M.ece(s_te, y_te), "brier_test": M.brier(s_te, y_te),
                   "realized_test_d": real[MARGINAL]["opinion"]["d"], "gap_test_d": real[MARGINAL]["gap_d"],
                   "thin_cells": sum(v["thin_cells"] for v in rep["segments"].values()),
                   "segments": {g: v for g, v in rep["segments"].items() if g != MARGINAL}}
            acc_collect[name].append(row)
            if name == "venn-abers" and seed == 0:
                cells_dump = {f"{g}|{i}": {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in c.items() if k != "opinion"} | {"d": round(c["opinion"]["d"], 4), "u": round(c["opinion"]["u"], 4)}
                              for (g, i), c in ct.cells.items()}
            if name == "raw":
                # alpha/beta knob on the raw scorer
                for key, (al, be) in (("(1,1)", (1, 1)), ("(2,1)", (2, 1)), ("(1,2)", (1, 2))):
                    k_ct = CalibrationTrust(M=10, binning="quantile", representative="mean", alpha=al, beta=be,
                                            debias="l2", W=W).fit(s_b, y_cal[b])
                    knob[key].append(k_ct.global_opinion["d"])
                # isotonic (Venn-Abers) partition
                i_ct = CalibrationTrust(binning="isotonic", representative="mean", debias="l2", W=W).fit(s_b, y_cal[b])
                iso.append({"n_blocks": i_ct.n_bins, "d": i_ct.global_opinion["d"], "b": i_ct.global_opinion["b"]})

        # per-decision, chapter mode: belief of the (confidence, correctness) cell as a ranking signal
        pred_b = (probs_cal[b, 1] >= 0.5).astype(int)
        conf_b = np.where(pred_b == 1, probs_cal[b, 1], probs_cal[b, 0])
        corr_b = (pred_b == y_cal[b]).astype(int)
        pred_te = (probs_te[:, 1] >= 0.5).astype(int)
        conf_te = np.where(pred_te == 1, probs_te[:, 1], probs_te[:, 0])
        wrong_te = (pred_te != y_te).astype(int)
        ch = CalibrationTrust(M=10, binning="fixed", representative="mid", evidence="accuracy", W=W).fit(conf_b, corr_b)
        belief = np.array([ch.lookup(float(c))["opinion"]["b"] for c in conf_te])
        # add a tiny tie-break by confidence inside a cell (cells are coarse)
        belief_tb = belief + 1e-6 * conf_te
        per_dec["opinion_belief"]["aurc"].append(M.aurc(belief_tb, wrong_te))
        per_dec["opinion_belief"]["capture10"].append(M.error_capture(belief_tb, wrong_te, 0.10))
        per_dec["msp"]["aurc"].append(M.aurc(conf_te, wrong_te))
        per_dec["msp"]["capture10"].append(M.error_capture(conf_te, wrong_te, 0.10))
        print(f"  S4 {ds['name']} seed {seed}: " + ", ".join(
            f"{n}: d={acc_collect[n][-1]['opinion']['d']:.3f} ece15={acc_collect[n][-1]['ece15_test']:.3f}"
            for n in acc_collect))

    for name, rows in acc_collect.items():
        out["scorers"][name] = {
            "d": agg([r["opinion"]["d"] for r in rows]), "b": agg([r["opinion"]["b"] for r in rows]),
            "u": agg([r["opinion"]["u"] for r in rows]),
            "binned_ece_calB": agg([r["binned_ece_calB"] for r in rows]),
            "ece15_test": agg([r["ece15_test"] for r in rows]), "brier_test": agg([r["brier_test"] for r in rows]),
            "realized_test_d": agg([r["realized_test_d"] for r in rows]), "gap_test_d": agg([r["gap_test_d"] for r in rows]),
            "thin_cells": agg([r["thin_cells"] for r in rows]),
        }
        segs = sorted({g for r in rows for g in r["segments"]})
        out["segments"][name] = {g: {"d": agg([r["segments"][g]["opinion"]["d"] for r in rows if g in r["segments"]]),
                                     "n": agg([r["segments"][g]["n"] for r in rows if g in r["segments"]]),
                                     "thin_cells": agg([r["segments"][g]["thin_cells"] for r in rows if g in r["segments"]])}
                                 for g in segs}
    out["per_decision"] = {k: {m: agg(v) for m, v in d.items()} for k, d in per_dec.items()}
    out["isotonic"] = {"n_blocks": agg([r["n_blocks"] for r in iso]), "d": agg([r["d"] for r in iso])}
    out["knob"] = {k: agg(v) for k, v in knob.items()}
    out["cells_seed0_venn_abers"] = cells_dump
    return out


# --------------------------------------------------------------------------- #
def s5_shift(ds):
    sys.path.insert(0, str(BASE / "eval"))
    import run_shift_bench as SB
    X, y, names = ds["X"], ds["y"], ds["feature_names"]
    regs = SB.regimes(X, y, names)
    out = {}
    for reg in ("iid", "pay0", "util"):
        ref_mask, test_mask, desc = regs[reg]
        rows = []
        for seed in range(3 if FAST else 5):
            rng = np.random.default_rng(seed)
            idx_tr, idx_cal, idx_te = SB.draw(rng, ref_mask, test_mask, disjoint=(reg != "iid"))
            model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X[idx_tr], y[idx_tr])
            p_cal, p_te = model.predict_proba(X[idx_cal])[:, 1], model.predict_proba(X[idx_te])[:, 1]
            ct = CalibrationTrust(M=10, binning="quantile", representative="mean", debias="l2", W=W).fit(p_cal, y[idx_cal])
            real = ct.realized(p_te, y[idx_te])[MARGINAL]
            rows.append({"claimed_d": ct.global_opinion["d"], "realized_d": real["opinion"]["d"],
                         "gap_d": real["gap_d"], "auc_test": float(roc_auc_score(y[idx_te], p_te)),
                         "ece15_ref": M.ece(p_cal, y[idx_cal]), "ece15_test": M.ece(p_te, y[idx_te])})
        out[reg] = {"description": desc, **{k: agg([r[k] for r in rows]) for k in rows[0]}}
        print(f"  S5 {reg}: claimed d={out[reg]['claimed_d']['mean']:.3f} realized d={out[reg]['realized_d']['mean']:.3f}")
    return out


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    res = {"meta": {"W": W, "fast": FAST, "M_grid": M_GRID, "estimators": ESTIMATORS,
                    "debias_constants": {"floor_ratio_theory": 0.3024, "l2_ratio_theory": 0.4294}}}
    print("S1 M sweep (synthetic, known ECE)")
    res["s1_msweep"] = s1_msweep()
    print("S2 consistency")
    res["s2_consistency"] = s2_consistency()
    print("S3 finite-sample bound")
    res["s3_bound"] = s3_bound()
    print("S4 real credit models")
    taiwan, german = D.load_taiwan(), D.load_german()
    res["s4_taiwan"] = s4_real(taiwan, 2 if FAST else 5)
    res["s4_german"] = s4_real(german, 3 if FAST else 10)
    print("S5 shift (Taiwan)")
    res["s5_shift_taiwan"] = s5_shift(taiwan)
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    (RESULTS / "calibration_trust.json").write_text(json.dumps(res, indent=1))
    print(f"wrote results/calibration_trust.json in {res['meta']['seconds']} s")


if __name__ == "__main__":
    main()
