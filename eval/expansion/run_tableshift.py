"""Workstream A: the reliability envelope on TableShift tasks (exploratory).

Protocol fixed in PREREGISTRATION_AMENDMENT.md, Amendment 2, section 2.4,
before any task was fetched. For every task exported by
`data_fetch/fetch_tableshift.py`, per seed:

  model      HistGradientBoostingClassifier(max_iter=300) on TableShift `train`
  calibrate  on TableShift `validation`
  test       on `id_test` and `ood_test` (TableShift's own domain split)
  caps       seeded uniform subsample: train 200k, calibration 20k, test 20k

Measured on each test split: accuracy and AUC (context), marginal coverage at
alpha 0.05 and 0.10, REVIEW rate, per-segment coverage (marginal and
Mondrian) for every sensitive attribute TableShift returns, calibration-trust
disbelief claimed vs realised, Venn-Abers ECE, and one 600-row martingale
stream (kNN distance to calibration) per split.

Writes results/expansion/tableshift.json and results/expansion/TABLESHIFT.md.
Every task with an export is run and reported; nothing is dropped.

Usage:
  .venv/bin/python eval/expansion/run_tableshift.py                # all exported tasks
  .venv/bin/python eval/expansion/run_tableshift.py acsincome      # one task
"""
import json
import pathlib
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "eval"))

from reliax_core.calibration_trust import CalibrationTrust, MARGINAL  # noqa: E402
from reliax_core.conformal import ConformalCalibrator                # noqa: E402
from reliax_core.fairness import MondrianConformal, coverage_audit   # noqa: E402
from reliax_core.martingale import (ALARM_THRESHOLD, ConformalMartingale,  # noqa: E402
                                    WATCH_THRESHOLD)
from reliax_core.ood import KNNOODDetector                           # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator               # noqa: E402
import methods as M                                                  # noqa: E402

CACHE = BASE / "data_fetch" / "cache" / "tableshift"
OUT = BASE / "results" / "expansion"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.05, 0.10)
ALPHA_MAIN = 0.05
CAP = {"train": 200_000, "validation": 20_000, "id_test": 20_000, "ood_test": 20_000}
STREAM_LEN = 600
# Supplementary, added 13 Sep 2026 after the first run of diabetes_readmission
# showed 3/5 ID streams at WATCH: 20 extra ID streams per seed give a less
# noisy false-alarm estimate. The pre-specified 5-stream figure is still the
# one reported against H-A3; this one is reported next to it, labelled.
EXTRA_ID_STREAMS = 20
VA_ROWS = 2_000          # Venn-Abers refits isotonic per query; ECE on a seeded 2k subsample
MIN_SEGMENT_CAL = 50     # a segment needs this many calibration rows for its own calibrator
THIN_TEST = 100          # segments under this many test rows are reported but flagged
# TableShift README table, baseline OOD accuracy change per task. Tasks at or
# below -5.0 are the "shift present" tasks of H-A2/H-A3. Recorded here so the
# hypothesis set is fixed by the benchmark authors' number, not ours.
TABLESHIFT_BASELINE_DROP = {
    "assistments": -34.5, "college_scorecard": -11.2, "mimic_extract_mort_hosp": -6.3,
    "diabetes_readmission": -5.9, "brfss_diabetes": -4.5, "mimic_extract_los_3": -3.4,
    "anes": -2.6, "acsfoodstamps": -2.4, "acsunemployment": -1.3, "acsincome": -1.3,
}
SHIFT_PRESENT = {t for t, d in TABLESHIFT_BASELINE_DROP.items() if d <= -5.0}
ORDER = ["acsincome", "acsfoodstamps", "acsunemployment", "brfss_diabetes",
         "diabetes_readmission", "college_scorecard", "assistments", "anes",
         "mimic_extract_mort_hosp", "mimic_extract_los_3"]


def agg(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if len(a) == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "n": int(len(a))}


def load_split(npz, meta, split, rng, cap):
    n = meta["splits"][split]["n"]
    idx = np.sort(rng.choice(n, cap, replace=False)) if n > cap else np.arange(n)
    X = np.concatenate([npz[f"{split}/X_int"][idx].astype(np.float32),
                        npz[f"{split}/X_float"][idx]], axis=1)
    y = npz[f"{split}/y"][idx].astype(int)
    G = {col: npz[f"{split}/G/{col}"][idx] for col in meta["segment_attributes"]}
    return X, y, G


def stream(mart, ood, X, rng):
    """One 600-row stream through the martingale; returns the state summary."""
    mart.reset()
    rows = rng.choice(len(X), STREAM_LEN, replace=True)
    to_watch = to_alarm = None
    for k, i in enumerate(rows, start=1):
        st = mart.update(ood.distance(X[i]))
        if to_watch is None and st["martingale"] >= WATCH_THRESHOLD:
            to_watch = k
        if to_alarm is None and st["martingale"] >= ALARM_THRESHOLD:
            to_alarm = k
            break
    return {"max_log10": float(mart.max_log10), "steps_to_watch": to_watch,
            "steps_to_alarm": to_alarm, "watch": to_watch is not None,
            "alarm": to_alarm is not None}


def segment_audit(conf, probs_cal, y_cal, g_cal, probs_te, y_te, g_te, alpha):
    names = sorted(n for n in set(g_cal) if (g_cal == n).sum() >= MIN_SEGMENT_CAL)
    if len(names) < 2:
        return None
    codes = {n: i for i, n in enumerate(names)}
    c_cal = np.array([codes[s] for s in g_cal if s in codes])
    keep_cal = np.array([s in codes for s in g_cal])
    c_te = np.array([codes.get(s, -1) for s in g_te])
    keep_te = c_te >= 0
    mond = MondrianConformal(probs_cal[keep_cal], y_cal[keep_cal], c_cal, names)
    audit = coverage_audit(conf, mond, probs_te[keep_te], y_te[keep_te], c_te[keep_te], alpha)
    for row in audit["segments"]:
        row["thin"] = row["n"] < THIN_TEST
        row["cal_n"] = mond.calibration_n(row["segment"])
    return audit


def run_seed(task, npz, meta, seed):
    rng = np.random.default_rng(seed)
    X_tr, y_tr, _ = load_split(npz, meta, "train", rng, CAP["train"])
    X_cal, y_cal, G_cal = load_split(npz, meta, "validation", rng, CAP["validation"])
    tests = {s: load_split(npz, meta, s, rng, CAP[s]) for s in ("id_test", "ood_test")}
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_cal)) < 2:
        return {"error": "single-class split"}

    t0 = time.time()
    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X_tr, y_tr)
    probs_cal = model.predict_proba(X_cal)
    conf = ConformalCalibrator(probs_cal, y_cal)
    va = VennAbersCalibrator(probs_cal[:, 1], y_cal)
    ct = CalibrationTrust(M=10, binning="quantile").fit(probs_cal[:, 1], y_cal)
    claimed_d = ct.report()["global"]["d"]
    ood = KNNOODDetector(k=10).fit(X_cal)
    mart = ConformalMartingale(ood.calib_dists, seed=seed)
    qhat = conf.qhat(ALPHA_MAIN)

    out = {"n": {"train": len(y_tr), "cal": len(y_cal),
                 **{s: len(tests[s][1]) for s in tests}},
           "claimed_d": claimed_d, "splits": {}}
    for split, (X_te, y_te, G_te) in tests.items():
        probs = model.predict_proba(X_te)
        pred = (probs[:, 1] >= 0.5).astype(int)
        set_sizes = (1.0 - probs <= qhat).sum(axis=1)
        r = {
            "accuracy": float(np.mean(pred == y_te)),
            "auc": float(roc_auc_score(y_te, probs[:, 1])) if len(np.unique(y_te)) > 1 else None,
            "coverage": {str(a): float(conf.empirical_coverage(probs, y_te, a)) for a in ALPHAS},
            "review_rate": float(np.mean(set_sizes != 1)),
            "mean_set_size": float(set_sizes.mean()),
            "raw_ece": M.ece(probs[:, 1], y_te),
        }
        real = ct.realized(probs[:, 1], y_te)[MARGINAL]
        r["realized_d"] = real["opinion"]["d"]
        r["gap_d"] = real["gap_d"]
        sub = rng.choice(len(y_te), min(VA_ROWS, len(y_te)), replace=False)
        va_point = np.array([va.interval(float(s))["point"] for s in probs[sub, 1]])
        r["va_ece"] = M.ece(va_point, y_te[sub])
        r["segments"] = {}
        for attr in meta["segment_attributes"]:
            a = segment_audit(conf, probs_cal, y_cal, G_cal[attr], probs, y_te, G_te[attr],
                              ALPHA_MAIN)
            if a is not None:
                r["segments"][attr] = a
        r["martingale"] = stream(mart, ood, X_te, rng)
        if split == "id_test":
            extra = [stream(mart, ood, X_te, np.random.default_rng(1000 * seed + j))
                     for j in range(EXTRA_ID_STREAMS)]
            r["martingale_extra_id"] = {"streams": EXTRA_ID_STREAMS,
                                        "watch": int(sum(e["watch"] for e in extra)),
                                        "alarm": int(sum(e["alarm"] for e in extra))}
        out["splits"][split] = r
    out["seconds"] = round(time.time() - t0)
    return out


def summarise(task, per_seed):
    ok = [s for s in per_seed if "error" not in s]
    S = {"seeds_run": len(per_seed), "seeds_ok": len(ok),
         "errors": [s["error"] for s in per_seed if "error" in s],
         "n": ok[0]["n"] if ok else None,
         "tableshift_baseline_drop": TABLESHIFT_BASELINE_DROP.get(task),
         "shift_present": task in SHIFT_PRESENT}
    if not ok:
        return S
    S["claimed_d"] = agg([s["claimed_d"] for s in ok])
    for split in ("id_test", "ood_test"):
        rows = [s["splits"][split] for s in ok]
        S[split] = {
            "accuracy": agg([r["accuracy"] for r in rows]),
            "auc": agg([r["auc"] for r in rows]),
            "coverage": {a: agg([r["coverage"][a] for r in rows]) for a in ("0.05", "0.1")},
            "review_rate": agg([r["review_rate"] for r in rows]),
            "mean_set_size": agg([r["mean_set_size"] for r in rows]),
            "raw_ece": agg([r["raw_ece"] for r in rows]),
            "va_ece": agg([r["va_ece"] for r in rows]),
            "realized_d": agg([r["realized_d"] for r in rows]),
            "gap_d": agg([r["gap_d"] for r in rows]),
            "martingale": {
                "alarm_rate": float(np.mean([r["martingale"]["alarm"] for r in rows])),
                "watch_rate": float(np.mean([r["martingale"]["watch"] for r in rows])),
                "max_log10": agg([r["martingale"]["max_log10"] for r in rows]),
                "steps_to_alarm": agg([r["martingale"]["steps_to_alarm"] for r in rows]),
            },
        }
        if split == "id_test":
            tot = sum(r["martingale_extra_id"]["streams"] for r in rows)
            S[split]["martingale_extra_id"] = {
                "streams": tot,
                "watch_rate": sum(r["martingale_extra_id"]["watch"] for r in rows) / tot,
                "alarm_rate": sum(r["martingale_extra_id"]["alarm"] for r in rows) / tot}
        segs = {}
        for attr in rows[0]["segments"]:
            names = [x["segment"] for x in rows[0]["segments"][attr]["segments"]]
            segs[attr] = []
            for name in names:
                cells = [x for r in rows for x in r["segments"].get(attr, {}).get("segments", [])
                         if x["segment"] == name]
                if not cells:
                    continue
                segs[attr].append({
                    "segment": name,
                    "n": int(np.mean([c["n"] for c in cells])),
                    "thin": any(c["thin"] for c in cells),
                    "marginal_coverage": agg([c["marginal_coverage"] for c in cells]),
                    "mondrian_coverage": agg([c["mondrian_coverage"] for c in cells]),
                })
        S[split]["segments"] = segs
    # coverage degradation, ID minus OOD, per seed then aggregated
    S["coverage_drop_0.05"] = agg([s["splits"]["id_test"]["coverage"]["0.05"]
                                   - s["splits"]["ood_test"]["coverage"]["0.05"] for s in ok])
    S["hypotheses"] = hypotheses(task, S, ok)
    return S


def hypotheses(task, S, ok):
    cov_id = S["id_test"]["coverage"]["0.05"]["mean"]
    cov_ood = S["ood_test"]["coverage"]["0.05"]["mean"]
    drops = [s["splits"]["id_test"]["coverage"]["0.05"] - s["splits"]["ood_test"]["coverage"]["0.05"]
             for s in ok]
    shift = task in SHIFT_PRESENT
    h = {"H-A1_id_coverage_within_0.01": abs(cov_id - 0.95) <= 0.01,
         "H-A3_false_alarm_at_most_1_in_5": S["id_test"]["martingale"]["watch_rate"] <= 0.2}
    if shift:
        a2 = cov_ood < cov_id and all(d > 0 for d in drops)
        h["H-A2_ood_coverage_below_id_all_seeds"] = a2
        h["H-A3_alarm_at_least_4_of_5"] = S["ood_test"]["martingale"]["alarm_rate"] >= 0.8
        if a2:
            h["H-A4_review_rate_ood_at_least_id"] = (S["ood_test"]["review_rate"]["mean"]
                                                     >= S["id_test"]["review_rate"]["mean"])
            h["H-A5_realized_d_ood_above_claimed"] = (S["ood_test"]["realized_d"]["mean"]
                                                      > S["claimed_d"]["mean"])
    else:
        h["note"] = "TableShift baseline drop above -5.0: H-A2 to H-A5 not applicable"
    return h


def write_markdown(results):
    L = ["# Workstream A: TableShift, exploratory (Amendment 2, section 2.4)", "",
         f"Generated by `eval/expansion/run_tableshift.py` on {results['meta']['date']}. "
         "Every task with an export is listed; nothing is dropped. Coverage target 0.95 "
         "(alpha = 0.05). Mean over seeds, std in brackets. Martingale: one 600-row stream "
         "per seed per split; ALARM = wealth >= 100, false alarm = wealth >= 20 on the ID stream. "
         "The supplementary 100-stream false-alarm column (20 extra ID streams per seed) was "
         "added after the first run of diabetes_readmission showed 3/5 WATCH on 5 streams; "
         "Ville's bound is 5% at WATCH.",
         "", "## Per-task table", "",
         "| task | shift | seeds | acc ID / OOD | cov ID | cov OOD | drop | REVIEW ID / OOD | "
         "ALARM (OOD) | false alarm (ID, 5 streams) | false alarm (ID, 100 streams, suppl.) "
         "| d claimed / realised OOD |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in results["tasks"]:
        S = results["tasks"][task]
        if not S.get("seeds_ok"):
            L.append(f"| `{task}` | | 0 | not run: {'; '.join(S.get('errors', ['no export']))[:80]} "
                     "| | | | | | | |")
            continue
        I, O = S["id_test"], S["ood_test"]
        f = lambda a: f"{a['mean']:.3f} ({a['std']:.3f})"  # noqa: E731
        L.append(
            f"| `{task}` | {S['tableshift_baseline_drop']:+.1f} | {S['seeds_ok']} "
            f"| {I['accuracy']['mean']:.3f} / {O['accuracy']['mean']:.3f} "
            f"| {f(I['coverage']['0.05'])} | {f(O['coverage']['0.05'])} "
            f"| {S['coverage_drop_0.05']['mean']:+.3f} "
            f"| {I['review_rate']['mean']:.3f} / {O['review_rate']['mean']:.3f} "
            f"| {O['martingale']['alarm_rate']:.0%} | {I['martingale']['watch_rate']:.0%} "
            f"| {I['martingale_extra_id']['watch_rate']:.1%} WATCH, "
            f"{I['martingale_extra_id']['alarm_rate']:.1%} ALARM "
            f"| {S['claimed_d']['mean']:.3f} / {O['realized_d']['mean']:.3f} |")
    L += ["", "## Pre-specified expectations, per task", ""]
    for task, S in results["tasks"].items():
        if not S.get("seeds_ok"):
            continue
        L.append(f"- `{task}`: " + "; ".join(
            f"{k} = {'met' if v is True else 'not met' if v is False else v}"
            for k, v in S["hypotheses"].items()))
    L += ["", "## Per-segment coverage (worst segment per attribute, OOD split)", "",
          "| task | attribute | worst segment (n) | marginal | Mondrian | thin |", "|---|---|---|---|---|---|"]
    for task, S in results["tasks"].items():
        if not S.get("seeds_ok"):
            continue
        for attr, rows in S["ood_test"]["segments"].items():
            if not rows:
                continue
            w = min(rows, key=lambda r: r["marginal_coverage"]["mean"])
            L.append(f"| `{task}` | {attr} | {w['segment']} ({w['n']}) "
                     f"| {w['marginal_coverage']['mean']:.3f} | {w['mondrian_coverage']['mean']:.3f} "
                     f"| {'yes' if w['thin'] else ''} |")
    L += ["", "## Not run", ""]
    for task in ORDER:
        if task not in results["tasks"]:
            L.append(f"- `{task}`: no export in `data_fetch/cache/tableshift/` "
                     "(credentialed source or fetch failed; see MANIFEST.json).")
    L += ["- `heloc`, `acspubcov`, `physionet`, `nhanes_lead`, `brfss_blood_pressure`: "
          "no domain split defined by TableShift; excluded before any data was seen."]
    (OUT / "TABLESHIFT.md").write_text("\n".join(L) + "\n")


def main(tasks):
    path = OUT / "tableshift.json"
    results = json.loads(path.read_text()) if path.exists() else {"tasks": {}, "per_seed": {}}
    results["meta"] = {"date": time.strftime("%Y-%m-%d"), "seeds": list(SEEDS), "alphas": list(ALPHAS),
                       "caps": CAP, "stream_len": STREAM_LEN, "va_rows": VA_ROWS,
                       "protocol": "PREREGISTRATION_AMENDMENT.md, Amendment 2, section 2.4"}
    exported = [t for t in ORDER if (CACHE / f"{t}.npz").exists()]
    for task in (tasks or exported):
        if not (CACHE / f"{task}.npz").exists():
            print(f"=== {task}: no export, skipped", flush=True)
            continue
        meta = json.loads((CACHE / f"{task}.meta.json").read_text())
        npz = np.load(CACHE / f"{task}.npz", allow_pickle=False)
        print(f"=== {task} ({meta['n_features']} features, segments {meta['segment_attributes']})",
              flush=True)
        per_seed = []
        for seed in SEEDS:
            r = run_seed(task, npz, meta, seed)
            per_seed.append(r)
            if "error" in r:
                print(f"  seed {seed}: {r['error']}", flush=True)
                continue
            I, O = r["splits"]["id_test"], r["splits"]["ood_test"]
            print(f"  seed {seed}: acc {I['accuracy']:.3f}/{O['accuracy']:.3f} "
                  f"cov {I['coverage']['0.05']:.4f}/{O['coverage']['0.05']:.4f} "
                  f"review {I['review_rate']:.3f}/{O['review_rate']:.3f} "
                  f"mart {I['martingale']['max_log10']:.1f}/{O['martingale']['max_log10']:.1f} "
                  f"alarm={O['martingale']['alarm']} ({r['seconds']}s)", flush=True)
        results["per_seed"][task] = per_seed
        results["tasks"][task] = summarise(task, per_seed)
        results["tasks"] = {t: results["tasks"][t] for t in ORDER if t in results["tasks"]}
        path.write_text(json.dumps(results, indent=1))
        write_markdown(results)
    print(f"wrote {path} and {OUT / 'TABLESHIFT.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
