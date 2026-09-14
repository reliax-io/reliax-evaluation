"""Workstream D: native temporal drift on Home Credit, Credit Risk Model
Stability (Kaggle, 2024). Exploratory; protocol fixed in
PREREGISTRATION_AMENDMENT.md, Amendment 2, section 2.6. Licence gate cleared
14 September 2026 (competition terms read: publication in a technical
whitepaper is not prohibited; data is never redistributed from this repo).

This is NOT the sealed 2018 Home Credit Default Risk release (section 2.1).

Inputs (data_gated/cache/homecredit2024/, four parquet files): the base table
(case_id, date_decision, WEEK_NUM, MONTH, target) and the depth-0 static
tables (static_0_0, static_0_1, static_cb_0), joined on case_id. Starter
preprocessing, as in the competition's published baseline notebook and
nothing more: columns typed by their suffix letter (P, A -> float; D -> date,
converted to days before date_decision; M -> categorical; L, T -> numeric
where numeric, else categorical); columns with more than 95% missing dropped;
no feature engineering, no tuning.

Model, fixed in advance: LightGBM, num_leaves 64, learning_rate 0.05, 500
rounds, seed = run seed. Temporal protocol: sort by date_decision; train on
the first 40% of weeks, calibrate on the next 10%, evaluate on the remaining
50% in WEEK_NUM order. Five seeds (the seed changes the model and the
calibration subsample, not the split).

Measured: rolling weekly and monthly coverage at alpha 0.05; the martingale
(kNN distance to the calibration set) over the evaluation stream in calendar
order, WATCH and ALARM crossings dated; claimed vs realised disbelief per
month; and the weighted-conformal repair (Tibshirani, Barber, Candes and
Ramdas 2019): at the first ALARM, likelihood-ratio weights from a logistic
domain classifier between the calibration window and the trailing 4 weeks,
weighted quantile, coverage over the 8 weeks after the trigger with and
without recalibration. Hypotheses H-D1 to H-D3.

Writes results/expansion/homecredit_stability.json and HOMECREDIT_STABILITY.md.
"""
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from reliax_core.calibration_trust import MARGINAL, CalibrationTrust                 # noqa: E402
from reliax_core.martingale import ALARM_THRESHOLD, ConformalMartingale, WATCH_THRESHOLD  # noqa: E402
from reliax_core.ood import KNNOODDetector                                            # noqa: E402

DATA = BASE / "data_gated" / "cache" / "homecredit2024"
OUT = BASE / "results" / "expansion"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.05
TRAIN_FRAC, CAL_FRAC = 0.40, 0.10
CAL_CAP = 20_000            # calibration rows for the kNN index and the martingale (seeded subsample)
STREAM_CAP = 60_000         # evaluation rows streamed through the martingale, seeded uniform subsample in time order
REPAIR_WEEKS, TRAILING_WEEKS = 8, 4
LGB = dict(objective="binary", num_leaves=64, learning_rate=0.05, n_estimators=500, verbose=-1)


def load() -> pd.DataFrame:
    base = pd.read_parquet(DATA / "train_base.parquet")
    parts = [pd.read_parquet(DATA / f) for f in ("train_static_0_0.parquet", "train_static_0_1.parquet")]
    static = pd.concat(parts, ignore_index=True)
    cb = pd.read_parquet(DATA / "train_static_cb_0.parquet")
    df = base.merge(static, on="case_id", how="left").merge(cb, on="case_id", how="left")
    df["date_decision"] = pd.to_datetime(df["date_decision"])
    df = df.sort_values(["date_decision", "case_id"]).reset_index(drop=True)
    return df


def starter_preprocess(df: pd.DataFrame):
    """Suffix typing as in the competition's baseline notebook; nothing else."""
    feats, cats = [], []
    for c in df.columns:
        if c in ("case_id", "date_decision", "MONTH", "WEEK_NUM", "target"):
            continue
        if df[c].isna().mean() > 0.95:
            continue
        suf = c[-1]
        if suf in ("P", "A"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        elif suf == "D":
            d = pd.to_datetime(df[c], errors="coerce")
            df[c] = ((df["date_decision"] - d).dt.days).astype("float32")
        elif suf == "M":
            df[c] = df[c].astype("category"); cats.append(c)
        else:                                            # L, T and untyped
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().sum() >= 0.5 * df[c].notna().sum():
                df[c] = num.astype("float32")
            else:
                df[c] = df[c].astype("category"); cats.append(c)
        feats.append(c)
    return feats, cats


def numeric_matrix(df, feats, cats, med=None):
    """Float matrix for the kNN index: numeric columns as is, categoricals as
    codes. Missing values are filled with medians FITTED ON THE CALIBRATION
    WINDOW and reused for every later window (pass `med`); fitting per window
    manufactures a shift on columns that are mostly missing."""
    X = np.empty((len(df), len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        X[:, j] = df[c].cat.codes.to_numpy() if c in cats else df[c].to_numpy()
    if med is None:
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X, med


def weighted_qhat(scores_cal, w_cal, alpha):
    """Weighted conformal quantile (Tibshirani et al. 2019), test weight taken as the mean weight."""
    order = np.argsort(scores_cal)
    s, w = scores_cal[order], w_cal[order]
    w_test = float(np.mean(w_cal))
    cum = np.cumsum(w) / (w.sum() + w_test)
    k = int(np.searchsorted(cum, 1 - alpha, side="left"))
    return float(s[min(k, len(s) - 1)])


def run_seed(df, feats, cats, weeks, seed):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    n_w = len(weeks)
    w_tr = weeks[: int(TRAIN_FRAC * n_w)]
    w_cal = weeks[int(TRAIN_FRAC * n_w): int((TRAIN_FRAC + CAL_FRAC) * n_w)]
    w_ev = weeks[int((TRAIN_FRAC + CAL_FRAC) * n_w):]
    tr = df[df["WEEK_NUM"].isin(w_tr)]
    cal = df[df["WEEK_NUM"].isin(w_cal)]
    ev = df[df["WEEK_NUM"].isin(w_ev)]
    t0 = time.time()
    model = lgb.LGBMClassifier(random_state=seed, **LGB).fit(tr[feats], tr["target"], categorical_feature=cats)
    p_cal = model.predict_proba(cal[feats])[:, 1]
    p_ev = model.predict_proba(ev[feats])[:, 1]
    y_cal, y_ev = cal["target"].to_numpy(), ev["target"].to_numpy()
    s_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)            # nonconformity 1 - p_hat_y
    s_ev = np.where(y_ev == 1, 1 - p_ev, p_ev)
    n = len(s_cal)
    qhat = float(np.quantile(np.sort(s_cal), min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher"))
    covered = s_ev <= qhat
    sizes = (1 - p_ev <= qhat).astype(int) + (p_ev <= qhat).astype(int)

    # rolling coverage per week and per month
    ev_w = ev["WEEK_NUM"].to_numpy(); ev_m = ev["MONTH"].to_numpy()
    weekly = {int(w): {"n": int((ev_w == w).sum()), "coverage": float(covered[ev_w == w].mean()),
                       "review_rate": float((sizes[ev_w == w] != 1).mean())} for w in np.unique(ev_w)}
    monthly = {int(m): {"n": int((ev_m == m).sum()), "coverage": float(covered[ev_m == m].mean()),
                        "first_week": int(ev_w[ev_m == m].min())} for m in np.unique(ev_m)}

    # calibration opinion: claimed on the calibration window, realised per month
    ct = CalibrationTrust(M=10, binning="quantile").fit(p_cal, y_cal)
    claimed_d = ct.report()["global"]["d"]
    for m in monthly:
        mask = ev_m == m
        monthly[m]["realized_d"] = ct.realized(p_ev[mask], y_ev[mask])[MARGINAL]["opinion"]["d"]

    # martingale over the evaluation stream in calendar order
    X_cal, med = numeric_matrix(cal, feats, cats)
    X_ev, _ = numeric_matrix(ev, feats, cats, med)
    sub_cal = rng.choice(len(X_cal), min(CAL_CAP, len(X_cal)), replace=False)
    ood = KNNOODDetector(k=10).fit(X_cal[sub_cal])
    mart = ConformalMartingale(ood.calib_dists, seed=seed)
    stream_idx = np.sort(rng.choice(len(X_ev), min(STREAM_CAP, len(X_ev)), replace=False))
    first_watch = first_alarm = None
    traj = []
    for i in stream_idx:
        st = mart.update(ood.distance(X_ev[i]))
        traj.append((int(ev_w[i]), st["log10_martingale"]))
        if first_watch is None and st["martingale"] >= WATCH_THRESHOLD:
            first_watch = int(ev_w[i])
        if first_alarm is None and st["martingale"] >= ALARM_THRESHOLD:
            first_alarm = int(ev_w[i])
            break
    wealth_by_week = {}
    for w, lw in traj:
        wealth_by_week[w] = max(wealth_by_week.get(w, -9), lw)

    # weighted-conformal repair (Tibshirani et al. 2019), run at a trigger week
    def repair_at(trigger, label):
        if trigger is None:
            return None
        trail = df[(df["WEEK_NUM"] < trigger) & (df["WEEK_NUM"] >= trigger - TRAILING_WEEKS)]
        after = (ev_w >= trigger) & (ev_w < trigger + REPAIR_WEEKS)
        if len(trail) < 200 or after.sum() < 200:
            return {"trigger": label, "trigger_week": int(trigger), "skipped": "too few rows"}
        X_trail, _ = numeric_matrix(trail, feats, cats, med)
        dom = LogisticRegression(max_iter=500, C=0.1).fit(
            np.vstack([X_cal, X_trail]), np.r_[np.zeros(len(X_cal)), np.ones(len(X_trail))])
        pr = np.clip(dom.predict_proba(X_cal)[:, 1], 1e-3, 1 - 1e-3)
        w = pr / (1 - pr) * (len(X_cal) / len(X_trail))            # likelihood-ratio weights
        q_w = weighted_qhat(s_cal, w, ALPHA)
        sizes_w = (1 - p_ev[after] <= q_w).astype(int) + (p_ev[after] <= q_w).astype(int)
        r = {"trigger": label, "trigger_week": int(trigger), "n_after": int(after.sum()),
             "coverage_without": float(covered[after].mean()), "coverage_with": float((s_ev[after] <= q_w).mean()),
             "review_without": float((sizes[after] != 1).mean()), "review_with": float((sizes_w != 1).mean()),
             "qhat": qhat, "qhat_weighted": q_w, "weight_ess": float(w.sum() ** 2 / (w ** 2).sum())}
        # second arm: recalibrate the quantile on the trailing window's own outcomes
        # (what a recalibration ticket does once labels land; the label-delay ceiling)
        trail_ev = (ev_w < trigger) & (ev_w >= trigger - TRAILING_WEEKS)
        if trail_ev.sum() >= 200:
            s_tr = np.sort(s_ev[trail_ev]); n_tr = len(s_tr)
            q_l = float(np.quantile(s_tr, min(1.0, np.ceil((n_tr + 1) * (1 - ALPHA)) / n_tr), method="higher"))
            sizes_l = (1 - p_ev[after] <= q_l).astype(int) + (p_ev[after] <= q_l).astype(int)
            r["coverage_with_outcome_recal"] = float((s_ev[after] <= q_l).mean())
            r["review_with_outcome_recal"] = float((sizes_l != 1).mean())
            r["qhat_outcome_recal"] = q_l
        gap = (1 - ALPHA) - r["coverage_without"]
        r["gap_recovered_fraction"] = float((r["coverage_with"] - r["coverage_without"]) / gap) if gap > 0 else None
        if gap > 0 and "coverage_with_outcome_recal" in r:
            r["gap_recovered_fraction_outcome_recal"] = float((r["coverage_with_outcome_recal"] - r["coverage_without"]) / gap)
        return r

    # pre-registered trigger: the first ALARM of the input martingale
    repair = repair_at(first_alarm, "first ALARM (pre-registered)")
    # supplementary trigger, added after the first run showed the ALARM at the
    # first evaluation week (see HOMECREDIT_STABILITY.md): the outcome verdict,
    # i.e. the first week whose trailing 4-week coverage falls below 0.94
    wk_sorted = sorted(weekly)
    outcome_trigger = None
    for i in range(TRAILING_WEEKS, len(wk_sorted)):
        win = wk_sorted[i - TRAILING_WEEKS:i]
        n_win = sum(weekly[w]["n"] for w in win)
        cov_win = sum(weekly[w]["coverage"] * weekly[w]["n"] for w in win) / n_win
        if cov_win < 0.94:
            outcome_trigger = wk_sorted[i]
            break
    repair_outcome = repair_at(outcome_trigger, "outcome breach (supplementary)")
    return {"n": {"train": len(tr), "cal": len(cal), "eval": len(ev)}, "weeks": {"train": [int(w_tr[0]), int(w_tr[-1])],
            "cal": [int(w_cal[0]), int(w_cal[-1])], "eval": [int(w_ev[0]), int(w_ev[-1])]},
            "auc_eval": float(roc_auc_score(y_ev, p_ev)), "qhat": qhat,
            "coverage_eval": float(covered.mean()), "review_eval": float((sizes != 1).mean()),
            "claimed_d": claimed_d, "weekly": weekly, "monthly": monthly,
            "martingale": {"first_watch_week": first_watch, "first_alarm_week": first_alarm,
                           "stream_rows": int(len(stream_idx)), "wealth_by_week": wealth_by_week},
            "repair": repair, "repair_outcome": repair_outcome, "outcome_trigger_week": outcome_trigger,
            "seconds": round(time.time() - t0)}


def summarise(per_seed):
    S = {"seeds": len(per_seed), "n": per_seed[0]["n"], "weeks": per_seed[0]["weeks"],
         "auc_eval": float(np.mean([s["auc_eval"] for s in per_seed])),
         "coverage_eval": float(np.mean([s["coverage_eval"] for s in per_seed])),
         "claimed_d": float(np.mean([s["claimed_d"] for s in per_seed]))}
    months = sorted(set(int(m) for s in per_seed for m in s["monthly"]))
    S["monthly"] = [{"month": m, "n": per_seed[0]["monthly"][m]["n"],
                     "coverage": float(np.mean([s["monthly"][m]["coverage"] for s in per_seed])),
                     "realized_d": float(np.mean([s["monthly"][m]["realized_d"] for s in per_seed]))} for m in months]
    S["min_month_coverage"] = min(r["coverage"] for r in S["monthly"])
    below = [r["month"] for r in S["monthly"] if r["coverage"] < 0.94]
    S["first_month_below_0.94"] = below[0] if below else None
    S["first_alarm_weeks"] = [s["martingale"]["first_alarm_week"] for s in per_seed]
    S["first_watch_weeks"] = [s["martingale"]["first_watch_week"] for s in per_seed]
    def agg_rep(key):
        reps = [s[key] for s in per_seed if s[key] and "skipped" not in s[key]]
        return {"seeds": len(reps), "trigger_weeks": [s[key]["trigger_week"] if s[key] else None for s in per_seed],
                "coverage_without": float(np.mean([r["coverage_without"] for r in reps])) if reps else None,
                "coverage_with": float(np.mean([r["coverage_with"] for r in reps])) if reps else None,
                "review_without": float(np.mean([r["review_without"] for r in reps])) if reps else None,
                "review_with": float(np.mean([r["review_with"] for r in reps])) if reps else None,
                "gap_recovered_fraction": [r["gap_recovered_fraction"] for r in reps],
                "coverage_with_outcome_recal": float(np.mean([r["coverage_with_outcome_recal"] for r in reps if "coverage_with_outcome_recal" in r])) if reps else None,
                "review_with_outcome_recal": float(np.mean([r["review_with_outcome_recal"] for r in reps if "review_with_outcome_recal" in r])) if reps else None,
                "gap_recovered_fraction_outcome_recal": [r.get("gap_recovered_fraction_outcome_recal") for r in reps]}
    S["repair_outcome"] = agg_rep("repair_outcome")
    reps = [s["repair"] for s in per_seed if s["repair"] and "skipped" not in s["repair"]]
    S["repair"] = {"seeds_with_alarm": len(reps),
                   "coverage_without": float(np.mean([r["coverage_without"] for r in reps])) if reps else None,
                   "coverage_with": float(np.mean([r["coverage_with"] for r in reps])) if reps else None,
                   "review_without": float(np.mean([r["review_without"] for r in reps])) if reps else None,
                   "review_with": float(np.mean([r["review_with"] for r in reps])) if reps else None,
                   "gap_recovered_fraction": [r["gap_recovered_fraction"] for r in reps]}
    h = {"H-D1_some_month_below_0.94": S["first_month_below_0.94"] is not None}
    if h["H-D1_some_month_below_0.94"]:
        first_bad = S["first_month_below_0.94"]
        early = 0
        for s in per_seed:
            fw = s["martingale"]["first_watch_week"]
            first_week_bad = s["monthly"][first_bad]["first_week"] if first_bad in s["monthly"] else s["monthly"][str(first_bad)]["first_week"]
            early += int(fw is not None and fw < first_week_bad)
        h["H-D2_watch_before_first_bad_month_seeds"] = early
        h["H-D2_met_3_of_5"] = early >= 3
        rec = [g for g in S["repair"]["gap_recovered_fraction"] if g is not None]
        h["H-D3_recovered_fraction_mean_at_ALARM_trigger"] = float(np.mean(rec)) if rec else None
        h["H-D3_repair_recovers_half_the_gap_at_ALARM_trigger"] = (float(np.mean(rec)) >= 0.5) if rec else "no gap at the trigger: coverage was above target"
        rec2 = [g for g in S["repair_outcome"]["gap_recovered_fraction"] if g is not None]
        h["supplementary_recovered_fraction_at_outcome_trigger"] = float(np.mean(rec2)) if rec2 else None
        rec3 = [g for g in S["repair_outcome"]["gap_recovered_fraction_outcome_recal"] if g is not None]
        h["supplementary_recovered_fraction_outcome_recalibration"] = float(np.mean(rec3)) if rec3 else None
    S["hypotheses"] = h
    return S


def write_markdown(R):
    S = R["summary"]
    L = ["# Workstream D: Home Credit Credit Risk Model Stability 2024, native temporal drift (Amendment 2, section 2.6)", "",
         f"Generated by `eval/expansion/run_homecredit_stability.py` on {R['meta']['date']}. Not the sealed 2018 release. "
         f"Base table plus depth-0 static tables, starter preprocessing only, LightGBM fixed in advance. "
         f"Weeks {S['weeks']['train']} train, {S['weeks']['cal']} calibration, {S['weeks']['eval']} evaluation in order; "
         f"{S['n']['train']:,} / {S['n']['cal']:,} / {S['n']['eval']:,} rows; {S['seeds']} seeds. Evaluation AUC {S['auc_eval']:.3f}; "
         f"coverage over the whole evaluation period {S['coverage_eval']:.4f} at target 0.95; claimed disbelief {S['claimed_d']:.3f}.", "",
         "## Monthly coverage and realised disbelief", "", "| month | n | coverage | realised d |", "|---|---|---|---|"]
    for r in S["monthly"]:
        L.append(f"| {r['month']} | {r['n']:,} | {r['coverage']:.4f} | {r['realized_d']:.3f} |")
    L += ["", f"Lowest month: {S['min_month_coverage']:.4f}; first month below 0.94: {S['first_month_below_0.94']}.",
          f"First WATCH week per seed: {S['first_watch_weeks']}; first ALARM week per seed: {S['first_alarm_weeks']}.", "",
          "## Weighted-conformal repair at the first ALARM (8 weeks after the trigger)", ""]
    rp = S["repair"]
    if rp["seeds_with_alarm"]:
        L += [f"Seeds with an ALARM: {rp['seeds_with_alarm']}. Coverage without / with recalibration: {rp['coverage_without']:.4f} / {rp['coverage_with']:.4f}; "
              f"REVIEW rate without / with: {rp['review_without']:.3f} / {rp['review_with']:.3f}; fraction of the coverage gap recovered per seed: "
              + ", ".join("n/a" if g is None else f"{g:.2f}" for g in rp["gap_recovered_fraction"]) + "."]
    else:
        L += ["No seed reached ALARM within the streamed evaluation rows; the repair was not triggered."]
    ro = S["repair_outcome"]
    L += ["", "## Supplementary: the same repair triggered by the outcome verdict", "",
          "Added after the first run: the pre-registered trigger (first ALARM) fired in the first evaluation week on every seed, "
          "where coverage was above target and there was nothing to repair. The supplementary trigger is the first week whose "
          "trailing 4-week coverage falls below 0.94; everything else is unchanged.", ""]
    if ro["seeds"]:
        L += [f"Trigger weeks per seed: {ro['trigger_weeks']}. Coverage in the 8 weeks after, without / with recalibration: "
              f"{ro['coverage_without']:.4f} / {ro['coverage_with']:.4f}; REVIEW rate without / with: {ro['review_without']:.3f} / {ro['review_with']:.3f}; "
              "fraction of the coverage gap recovered per seed: " + ", ".join("n/a" if g is None else f"{g:.2f}" for g in ro["gap_recovered_fraction"]) + ".",
              f"Second arm, recalibrating the quantile on the trailing 4 weeks' own outcomes (the label-delay ceiling): coverage {ro['coverage_with_outcome_recal']:.4f}, "
              f"REVIEW rate {ro['review_with_outcome_recal']:.3f}, gap recovered per seed: " + ", ".join("n/a" if g is None else f"{g:.2f}" for g in ro["gap_recovered_fraction_outcome_recal"]) + "."]
    else:
        L += ["No seed reached the outcome trigger."]
    L += ["", "## Pre-specified expectations", ""] + [f"- {k}: {v}" for k, v in S["hypotheses"].items()]
    (OUT / "HOMECREDIT_STABILITY.md").write_text("\n".join(L) + "\n")


def main():
    t0 = time.time()
    df = load()
    feats, cats = starter_preprocess(df)
    weeks = np.sort(df["WEEK_NUM"].unique())
    print(f"rows {len(df):,}, features {len(feats)} ({len(cats)} categorical), weeks {weeks[0]}..{weeks[-1]}, "
          f"target rate {df['target'].mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    per_seed = []
    for seed in SEEDS:
        r = run_seed(df, feats, cats, weeks, seed)
        per_seed.append(r)
        print(f"  seed {seed}: auc {r['auc_eval']:.3f} cov {r['coverage_eval']:.4f} review {r['review_eval']:.3f} "
              f"watch wk {r['martingale']['first_watch_week']} alarm wk {r['martingale']['first_alarm_week']} "
              f"repair@alarm {None if not r['repair'] or 'skipped' in r['repair'] else (round(r['repair']['coverage_without'],4), round(r['repair']['coverage_with'],4))} "
              f"repair@outcome wk {r['outcome_trigger_week']} {None if not r['repair_outcome'] or 'skipped' in r['repair_outcome'] else (round(r['repair_outcome']['coverage_without'],4), round(r['repair_outcome']['coverage_with'],4))} ({r['seconds']}s)", flush=True)
        R = {"meta": {"date": time.strftime("%Y-%m-%d"), "alpha": ALPHA, "seeds": list(SEEDS), "lgb": LGB,
                      "caps": {"cal_knn": CAL_CAP, "stream": STREAM_CAP}, "protocol": "Amendment 2, section 2.6",
                      "licence_gate": "cleared 14 Sep 2026; data not redistributed"},
             "summary": summarise(per_seed), "per_seed": per_seed}
        (OUT / "homecredit_stability.json").write_text(json.dumps(R, indent=1, default=str))
        write_markdown(R)
    print(f"wrote {OUT / 'homecredit_stability.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
