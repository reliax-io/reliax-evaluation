"""ReliaX evaluation on real credit datasets (Taiwan default, German credit).

Runs every experiment reported in PAPER.md and writes:
  paper/results/results.json   (all numbers, mean and std over seeds)
  paper/results/fig_*.png      (figures, computed from the same runs)

Usage:
  .venv/bin/python paper/eval/run_eval.py
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

from reliax_core.auditor import ErrorAuditor                     # noqa: E402
from reliax_core.conformal import ConformalCalibrator            # noqa: E402
from reliax_core.drift import PSIMonitor                         # noqa: E402
from reliax_core.fairness import MondrianConformal, coverage_audit  # noqa: E402
from reliax_core.martingale import (ALARM_THRESHOLD, ConformalMartingale,  # noqa: E402
                                        WATCH_THRESHOLD)
from reliax_core.ood import KNNOODDetector                       # noqa: E402
from reliax_core.scoring import reliability_score                # noqa: E402
from reliax_core.sl_fusion import fuse_signals                   # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator           # noqa: E402

import data as D                                                     # noqa: E402
import methods as M                                                  # noqa: E402

RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
ALPHA = 0.05
REFERRAL_RATES = (0.10, 0.20)
STREAM_LEN = 600


def agg(values):
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


def run_seed(ds, seed, do_drift, do_latency):
    X, y = ds["X"], ds["y"]
    idx = np.arange(len(y))
    idx_tr, idx_rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    idx_cal, idx_te = train_test_split(idx_rest, test_size=0.5, random_state=seed,
                                       stratify=y[idx_rest])
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_cal, y_cal = X[idx_cal], y[idx_cal]
    X_te, y_te = X[idx_te], y[idx_te]

    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X_tr, y_tr)
    probs_cal = model.predict_proba(X_cal)
    probs_te = model.predict_proba(X_te)
    logit_cal = model.decision_function(X_cal)
    logit_te = model.decision_function(X_te)
    pred_te = (probs_te[:, 1] >= 0.5).astype(int)
    wrong = (pred_te != y_te).astype(int)
    bad_approval = ((pred_te == 0) & (y_te == 1)).astype(int)   # approved, then defaulted

    out = {"model": {
        "accuracy": float(np.mean(pred_te == y_te)),
        "auc": float(roc_auc_score(y_te, probs_te[:, 1])),
        "error_rate": float(wrong.mean()),
        "bad_approval_rate": float(bad_approval.mean()),
    }}

    # ---- E1 marginal coverage --------------------------------------------------
    conf = ConformalCalibrator(probs_cal, y_cal)
    out["coverage"] = {str(a): float(conf.empirical_coverage(probs_te, y_te, a))
                       for a in (0.05, 0.10)}
    qhat = conf.qhat(ALPHA)
    set_sizes = (1.0 - probs_te <= qhat).sum(axis=1)
    out["mean_set_size"] = float(set_sizes.mean())

    # ---- E2 per-segment coverage: marginal vs Mondrian -------------------------
    out["segments"] = {}
    for attr, seg_all in ds["segments"].items():
        names = sorted(set(seg_all))
        codes = {n: i for i, n in enumerate(names)}
        g_cal = np.array([codes[s] for s in seg_all[idx_cal]])
        g_te = np.array([codes[s] for s in seg_all[idx_te]])
        mond = MondrianConformal(probs_cal, y_cal, g_cal, names)
        audit = coverage_audit(conf, mond, probs_te, y_te, g_te, ALPHA)
        out["segments"][attr] = audit["segments"]

    # ---- E3 selective prediction ----------------------------------------------
    t_temp = M.fit_temperature(logit_cal, y_cal)
    p_t = 1.0 / (1.0 + np.exp(-logit_te / t_temp))
    msp = np.maximum(probs_te[:, 1], probs_te[:, 0])
    margin = 2 * msp - 1
    pvals = np.array([conf.p_value(float(p)) for p in msp])
    va = VennAbersCalibrator(probs_cal[:, 1], y_cal)
    t0 = time.perf_counter()
    va_iv = [va.interval(float(s)) for s in probs_te[:, 1]]
    va_seconds = time.perf_counter() - t0
    va_width = np.array([v["width"] for v in va_iv])
    va_point = np.array([v["point"] for v in va_iv])
    ood = KNNOODDetector(k=10).fit(X_cal)
    ood_pct = ood.percentiles_batch(X_te)
    auditor = ErrorAuditor(seed=seed).fit(X_cal, probs_cal, y_cal, model.predict(X_cal))
    p_err = auditor.p_error_batch(X_te, probs_te)
    composite = np.array([
        reliability_score(float(margin[i]), float(pvals[i]), int(set_sizes[i]),
                          float(ood_pct[i]), bool(ood_pct[i] > 99.0),
                          float(va_width[i]), float(p_err[i]),
                          bool(p_err[i] > auditor.flag_threshold), False)
        for i in range(len(y_te))
    ])
    sl_expected = np.array([
        fuse_signals(float(msp[i]), float(pvals[i]), int(set_sizes[i]),
                     float(va_width[i]), float(p_err[i]), float(ood_pct[i]),
                     "OK", 1.0 - auditor.base_error_rate)["expected"]
        for i in range(len(y_te))
    ])
    signals = {
        "msp_tscaled": msp,            # T-scaling is monotone: same ranking as raw MSP
        "conformal_pvalue": pvals,
        "va_width": -va_width,
        "auditor": -p_err,
        "composite": composite,
        "sl_fusion": sl_expected,
    }
    out["selective"] = {}
    for name, s in signals.items():
        entry = {"aurc": M.aurc(s, wrong)}
        for r in REFERRAL_RATES:
            entry[f"err_capture@{r}"] = M.error_capture(s, wrong, r)
            entry[f"bad_approval_capture@{r}"] = M.default_capture(s, bad_approval, r)
        out["selective"][name] = entry
    out["curves"] = {  # for figure 1 (seed 0 only is plotted)
        name: [M.error_capture(s, wrong, r) for r in np.linspace(0.02, 0.5, 25)]
        for name, s in signals.items()
    }
    out["curve_rates"] = list(np.linspace(0.02, 0.5, 25))

    # ---- E4 calibration ---------------------------------------------------------
    out["calibration"] = {
        "temperature": t_temp,
        "raw": {"ece": M.ece(probs_te[:, 1], y_te), "brier": M.brier(probs_te[:, 1], y_te)},
        "tscaled": {"ece": M.ece(p_t, y_te), "brier": M.brier(p_t, y_te)},
        "venn_abers": {"ece": M.ece(va_point, y_te), "brier": M.brier(va_point, y_te)},
        "va_mean_width": float(va_width.mean()),
    }
    dec = np.quantile(va_width, 0.9)
    wide = va_width >= dec
    out["width_decile"] = {
        "err_rate_widest_decile": float(wrong[wide].mean()),
        "err_rate_rest": float(wrong[~wide].mean()),
    }
    out["_va_seconds"] = va_seconds
    out["_reliability_bins"] = _reliability_bins(probs_te[:, 1], va_point, y_te)

    # ---- E5 drift (Taiwan only) -------------------------------------------------
    if do_drift:
        rng = np.random.default_rng(seed)
        pay0 = ds["feature_names"].index("PAY_0")
        shifted_pool = idx_te[X_te[:, pay0] >= 1]           # real delayed-payment subgroup
        mart = ConformalMartingale(ood.calib_dists, seed=seed)
        iid_traj = []
        for i in rng.choice(len(X_te), STREAM_LEN, replace=True):
            iid_traj.append(mart.update(ood.distance(X_te[i]))["log10_martingale"])
        iid_max = max(iid_traj)
        mart.reset()
        shift_traj, to_watch, to_alarm = [], None, None
        pool_local = rng.choice(shifted_pool, STREAM_LEN, replace=True)
        psi = PSIMonitor(X_cal, probs_cal[:, 1], ds["feature_names"])
        for k, gi in enumerate(pool_local, start=1):
            x = X[gi]
            st = mart.update(ood.distance(x))
            shift_traj.append(st["log10_martingale"])
            psi.observe(x, float(model.predict_proba(x.reshape(1, -1))[0, 1]))
            if to_watch is None and st["martingale"] >= WATCH_THRESHOLD:
                to_watch = k
            if to_alarm is None and st["martingale"] >= ALARM_THRESHOLD:
                to_alarm = k
                break
        rep = psi.report()
        out["drift"] = {
            "iid_max_log10": iid_max,
            "iid_false_alarm": iid_max >= np.log10(WATCH_THRESHOLD),
            "steps_to_watch": to_watch, "steps_to_alarm": to_alarm,
            "shift_pool_frac": float(len(shifted_pool) / len(X_te)),
            "psi_top": rep["features"][:3] if rep["ready"] else [],
            "psi_score": rep.get("score_psi"),
            "_iid_traj": iid_traj, "_shift_traj": shift_traj,
        }

    # ---- E6 latency (Taiwan, seed 0 only) ---------------------------------------
    if do_latency:
        lat = []
        for i in range(200):
            x = X_te[i]
            t0 = time.perf_counter()
            p = model.predict_proba(x.reshape(1, -1))[0]
            conf.prediction_set(p, ALPHA)
            conf.p_value(float(p.max()))
            va.interval(float(p[1]))
            ood.assess(x)
            auditor.assess(x, p)
            lat.append((time.perf_counter() - t0) * 1000)
        out["latency_ms"] = {"p50": float(np.percentile(lat, 50)),
                             "p95": float(np.percentile(lat, 95)),
                             "mean": float(np.mean(lat))}
    out["n"] = {"train": len(y_tr), "cal": len(y_cal), "test": len(y_te)}
    return out


def _reliability_bins(p_raw, p_va, y, n_bins=12):
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        row = {"bin": [float(lo), float(hi)]}
        for tag, p in (("raw", p_raw), ("va", p_va)):
            m = (p >= lo) & (p < hi)
            row[tag] = {"n": int(m.sum()),
                        "conf": float(p[m].mean()) if m.sum() else None,
                        "acc": float(y[m].mean()) if m.sum() else None}
        rows.append(row)
    return rows


def aggregate(per_seed, keys_scalar):
    """mean/std over seeds for a dict of nested scalar paths."""
    def get(d, path):
        for k in path:
            d = d[k]
        return d
    return {"/".join(map(str, p)): agg([get(s, p) for s in per_seed]) for p in keys_scalar}


def main():
    t_start = time.time()
    results = {"meta": {"alpha": ALPHA, "date": time.strftime("%Y-%m-%d"),
                        "split": "50/25/25 train/cal/test, stratified"}}
    datasets = [(D.load_taiwan(), list(range(5)), True),
                (D.load_german(), list(range(10)), False)]
    for ds, seeds, is_taiwan in datasets:
        print(f"=== {ds['name']} (n={len(ds['y'])}, d={ds['X'].shape[1]}) ===", flush=True)
        per_seed = []
        for s in seeds:
            t0 = time.time()
            r = run_seed(ds, s, do_drift=is_taiwan, do_latency=(is_taiwan and s == 0))
            per_seed.append(r)
            print(f"  seed {s}: acc={r['model']['accuracy']:.3f} "
                  f"cov@.05={r['coverage']['0.05']:.4f} "
                  f"({time.time()-t0:.0f}s, VA {r['_va_seconds']:.0f}s)", flush=True)

        entry = {"n": per_seed[0]["n"], "seeds": len(seeds)}
        entry["model"] = {k: agg([s["model"][k] for s in per_seed])
                          for k in per_seed[0]["model"]}
        entry["coverage"] = {k: agg([s["coverage"][k] for s in per_seed])
                             for k in ("0.05", "0.1")}
        entry["mean_set_size"] = agg([s["mean_set_size"] for s in per_seed])
        entry["segments"] = {}
        for attr in per_seed[0]["segments"]:
            segs = {}
            for row in per_seed[0]["segments"][attr]:
                name = row["segment"]
                segs[name] = {
                    "n": row["n"],
                    "marginal": agg([next(r2["marginal_coverage"] for r2 in s["segments"][attr]
                                          if r2["segment"] == name) for s in per_seed]),
                    "mondrian": agg([next(r2["mondrian_coverage"] for r2 in s["segments"][attr]
                                          if r2["segment"] == name) for s in per_seed]),
                }
            entry["segments"][attr] = segs
        entry["selective"] = {}
        for sig in per_seed[0]["selective"]:
            entry["selective"][sig] = {k: agg([s["selective"][sig][k] for s in per_seed])
                                       for k in per_seed[0]["selective"][sig]}
        entry["calibration"] = {
            "temperature": agg([s["calibration"]["temperature"] for s in per_seed]),
            "va_mean_width": agg([s["calibration"]["va_mean_width"] for s in per_seed]),
        }
        for m in ("raw", "tscaled", "venn_abers"):
            entry["calibration"][m] = {k: agg([s["calibration"][m][k] for s in per_seed])
                                       for k in ("ece", "brier")}
        entry["width_decile"] = {k: agg([s["width_decile"][k] for s in per_seed])
                                 for k in per_seed[0]["width_decile"]}
        if is_taiwan:
            entry["drift"] = {
                "iid_max_log10": agg([s["drift"]["iid_max_log10"] for s in per_seed]),
                "false_alarms": int(sum(s["drift"]["iid_false_alarm"] for s in per_seed)),
                "steps_to_watch": agg([s["drift"]["steps_to_watch"]
                                       if s["drift"]["steps_to_watch"] is not None else np.nan
                                       for s in per_seed]),
                "steps_to_alarm": agg([s["drift"]["steps_to_alarm"]
                                       if s["drift"]["steps_to_alarm"] is not None else np.nan
                                       for s in per_seed]),
                "alarms": int(sum(s["drift"]["steps_to_alarm"] is not None for s in per_seed)),
                "shift_pool_frac": agg([s["drift"]["shift_pool_frac"] for s in per_seed]),
                "psi_top_seed0": per_seed[0]["drift"]["psi_top"],
                "psi_score_seed0": per_seed[0]["drift"]["psi_score"],
            }
            entry["latency_ms"] = per_seed[0]["latency_ms"]
        results[ds["name"]] = entry
        # stash raw per-seed material needed for figures
        if is_taiwan:
            results["_fig"] = {
                "curves": per_seed[0]["curves"],
                "curve_rates": per_seed[0]["curve_rates"],
                "reliability_bins": per_seed[0]["_reliability_bins"],
                "iid_trajs": [s["drift"]["_iid_traj"] for s in per_seed[:3]],
                "shift_trajs": [s["drift"]["_shift_traj"] for s in per_seed[:3]],
                "width_decile": per_seed[0]["width_decile"],
            }

    results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    with open(RESULTS / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", RESULTS / "results.json", f"({results['meta']['runtime_s']}s)")


if __name__ == "__main__":
    main()
