"""Calibration trust on real credit-risk models: does the opinion fitted on the
calibration split predict the miscalibration realised on held-out applicants?

For each dataset (Taiwan default, German credit), seed, scorer (raw model,
temperature-scaled, Venn-Abers) and clustering (chapter: fixed-width bins with
midpoint representatives; quantile bins with mean representatives; isotonic
blocks), the opinion is fitted on half of the calibration split and then
re-scored on the test split with the SAME cells and stated probabilities.

Metrics (all out of sample):
  claimed_d / realized_d   global disbelief the calibration data implied vs the
                           disbelief the test outcomes realise
  envelope_held            |binned ECE (cal) - binned ECE (test)| within the sum
                           of the two Hoeffding half-widths (delta = 0.05 each)
  cell_mae                 mean |claimed cell deviation - realised cell deviation|,
                           weighted by test cell size: the out-of-sample error of
                           the calibration map (the clustering comparison metric)
  cell_spearman            rank agreement between claimed and realised |deviation|
  worst_cell_hit           is the cell with the largest realised |deviation| among
                           the three cells the opinion flagged as worst?
  segment                  claimed vs realised disbelief per age band

Usage: .venv/bin/python eval/run_calibration_credit.py
Writes results/calibration_credit.json and results/fig7_ct_credit.png
"""
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from reliax_core.calibration_trust import CalibrationTrust, MARGINAL, hoeffding_bound  # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator                                # noqa: E402

import data as D                                                                       # noqa: E402
import methods as M                                                                    # noqa: E402

RESULTS = BASE / "results"
W = 2.0
SCORERS = ("raw", "temperature-scaled", "venn-abers")
CLUSTERINGS = {
    "chapter: fixed bins, midpoint": dict(M=10, binning="fixed", representative="mid"),
    "quantile bins, mean rep":       dict(M=10, binning="quantile", representative="mean"),
    "isotonic blocks, mean rep":     dict(binning="isotonic", representative="mean"),
}


def agg(v):
    a = np.asarray(v, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


def scores_for(name, p_raw, logit, t, va):
    if name == "raw":
        return p_raw
    if name == "temperature-scaled":
        return 1 / (1 + np.exp(-logit / t))
    return np.array([va.interval(float(s))["point"] for s in p_raw])


def run(ds, seeds):
    rows = {sc: {cl: [] for cl in CLUSTERINGS} for sc in SCORERS}
    segs = {sc: [] for sc in SCORERS}
    scatter = None
    for seed in range(seeds):
        X, y = ds["X"], ds["y"]
        idx = np.arange(len(y))
        idx_tr, idx_rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
        idx_cal, idx_te = train_test_split(idx_rest, test_size=0.5, random_state=seed, stratify=y[idx_rest])
        model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X[idx_tr], y[idx_tr])
        probs_cal, probs_te = model.predict_proba(X[idx_cal]), model.predict_proba(X[idx_te])
        logit_cal, logit_te = model.decision_function(X[idx_cal]), model.decision_function(X[idx_te])
        y_cal, y_te = y[idx_cal], y[idx_te]
        seg_cal, seg_te = ds["segments"]["age_band"][idx_cal], ds["segments"]["age_band"][idx_te]
        a, b = train_test_split(np.arange(len(y_cal)), test_size=0.5, random_state=0, stratify=y_cal)
        t = M.fit_temperature(logit_cal[a], y_cal[a])
        va = VennAbersCalibrator(probs_cal[a, 1], y_cal[a])
        for sc in SCORERS:
            s_b = scores_for(sc, probs_cal[b, 1], logit_cal[b], t, va)
            s_te = scores_for(sc, probs_te[:, 1], logit_te, t, va)
            for cl, cfg in CLUSTERINGS.items():
                ct = CalibrationTrust(debias="l2", W=W, **cfg).fit(s_b, y_cal[b], segments=seg_cal[b])
                real = ct.realized(s_te, y_te, segments=seg_te)
                cells = ct.realized_cells(s_te, y_te)
                n_te = np.array([c["n"] for c in cells], float)
                claimed = np.array([c["claimed_dev"] for c in cells])
                realised = np.array([c["dev"] for c in cells])
                mae = float(np.sum(n_te * np.abs(claimed - realised)) / n_te.sum())
                rho = float(spearmanr(np.abs(claimed), np.abs(realised)).correlation) if len(cells) > 2 else np.nan
                worst_real = int(np.argmax(np.abs(realised)))
                top3_claimed = set(np.argsort(-np.abs(claimed))[:3].tolist())
                ece_cal, ece_te = ct.binned_ece(), float(np.sum(n_te * np.abs(realised)) / n_te.sum())
                bound = hoeffding_bound([c["n"] for (g, _), c in ct.cells.items() if g == MARGINAL]) + \
                    hoeffding_bound(n_te)
                rows[sc][cl].append({
                    "claimed_d": ct.global_opinion["d"], "realized_d": real[MARGINAL]["opinion"]["d"],
                    "gap_d": real[MARGINAL]["gap_d"], "ece_cal": ece_cal, "ece_test": ece_te,
                    "envelope_half_width": bound, "envelope_held": abs(ece_cal - ece_te) <= bound,
                    "cell_mae": mae, "cell_spearman": rho, "worst_cell_hit": worst_real in top3_claimed,
                    "n_cells": len(cells), "thin_cells": sum(1 for (g, _), c in ct.cells.items() if g == MARGINAL and c["insufficient_evidence"]),
                })
                if cl == "quantile bins, mean rep":
                    segs[sc].append({g: (v["claimed_d"], v["opinion"]["d"]) for g, v in real.items() if g != MARGINAL})
                if seed == 0 and sc == "venn-abers":
                    scatter = scatter or {}
                    scatter[cl] = {"claimed": claimed.tolist(), "realised": realised.tolist(), "n": n_te.tolist()}
        print(f"  {ds['name']} seed {seed}: " + ", ".join(
            f"{sc}: d {rows[sc]['quantile bins, mean rep'][-1]['claimed_d']:.3f}->{rows[sc]['quantile bins, mean rep'][-1]['realized_d']:.3f}"
            for sc in SCORERS))
    out = {"n_seeds": seeds, "scorers": {}, "segments": {}, "scatter_seed0_venn_abers": scatter}
    for sc in SCORERS:
        out["scorers"][sc] = {}
        for cl, rs in rows[sc].items():
            out["scorers"][sc][cl] = {k: agg([r[k] for r in rs]) for k in rs[0] if k not in ("envelope_held", "worst_cell_hit")}
            out["scorers"][sc][cl]["envelope_held"] = f"{sum(r['envelope_held'] for r in rs)}/{len(rs)}"
            out["scorers"][sc][cl]["worst_cell_hit"] = f"{sum(r['worst_cell_hit'] for r in rs)}/{len(rs)}"
        groups = sorted({g for s in segs[sc] for g in s})
        out["segments"][sc] = {g: {"claimed_d": agg([s[g][0] for s in segs[sc] if g in s]),
                                   "realized_d": agg([s[g][1] for s in segs[sc] if g in s])} for g in groups}
    return out


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc = res["taiwan"]["scatter_seed0_venn_abers"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, (cl, d) in zip(axes, sc.items()):
        n = np.array(d["n"])
        ax.scatter(d["claimed"], d["realised"], s=8 + 200 * n / n.max(), color="#5d7f9b", alpha=.75)
        lim = 0.12
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=1)
        ax.axhline(0, color="#bbb", lw=.6); ax.axvline(0, color="#bbb", lw=.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(cl, fontsize=9.5)
        ax.set_xlabel("claimed deviation (calibration split)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("realised deviation (test applicants)")
    fig.suptitle("Taiwan default, Venn-Abers scorer, seed 0: per-cell claimed vs realised (observed rate - stated probability); "
                 "marker size = test cell size", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig7_ct_credit.png", dpi=160)


def main():
    res = {"meta": {"W": W, "debias": "l2", "clusterings": {k: str(v) for k, v in CLUSTERINGS.items()}}}
    print("Taiwan")
    res["taiwan"] = run(D.load_taiwan(), 5)
    print("German")
    res["german"] = run(D.load_german(), 10)
    (RESULTS / "calibration_credit.json").write_text(json.dumps(res, indent=1))
    figure(res)
    print("wrote results/calibration_credit.json and fig7_ct_credit.png")


if __name__ == "__main__":
    main()
