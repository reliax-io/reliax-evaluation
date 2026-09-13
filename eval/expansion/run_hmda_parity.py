"""Workstream C: coverage parity on reported protected attributes, HMDA 2025
(exploratory; Amendment 2, section 2.5).

The HMDA outcome is a lender's decision (originate or deny), not a repayment
outcome. This measures the envelope's coverage, set size and REVIEW rate per
reported protected group over a decision-prediction model, and repeats the
audit with a geography-only proxy (census-tract modal group, ACS B03002) in
place of the reported attribute. It says nothing about PD calibration.

Population (Amendment 2): action_taken in {1, 3}; loan_purpose in {1, 31, 32};
first lien; site-built; 1 to 4 units; principal residence; not a reverse
mortgage; not primarily business. Label y = 1 if denied. Seeded stratified
subsample: 400,000 rows for model plus calibration (split 300,000 / 100,000)
and 200,000 for test. 5 seeds.

Deviation from the pre-specified feature list, recorded here: "interest rate
presence" is dropped. HMDA reports interest_rate only for originated loans,
so its presence is the label itself. Found in the field documentation before
any model was fitted.

Writes results/expansion/hmda_parity.json and results/expansion/HMDA_PARITY.md.
"""
import gzip
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from reliax_core.conformal import ConformalCalibrator                # noqa: E402
from reliax_core.fairness import MondrianConformal                   # noqa: E402

HMDA = BASE / "data_fetch" / "cache" / "hmda" / "2025"
ACS = BASE / "data_fetch" / "cache" / "acs" / "tracts_b03002_2023.csv"
OUT = BASE / "results" / "expansion"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.05
N_MODEL, N_CAL, N_TEST = 300_000, 100_000, 200_000
THIN_TEST, THIN_CAL = 100, 500
USECOLS = ["state_code", "census_tract", "derived_ethnicity", "derived_race", "derived_sex",
           "action_taken", "loan_type", "loan_purpose", "lien_status", "reverse_mortgage",
           "business_or_commercial_purpose", "loan_amount", "loan_to_value_ratio", "loan_term",
           "property_value", "construction_method", "occupancy_type", "total_units", "income",
           "debt_to_income_ratio", "applicant_age", "co-applicant_age", "aus-1"]
NUMERIC = ["loan_amount", "loan_to_value_ratio", "loan_term", "property_value", "income"]
CATEGORICAL = ["loan_type", "loan_purpose", "debt_to_income_ratio", "applicant_age",
               "co_applicant", "aus-1", "state_code"]
PROXY_GROUPS = {"white_nh": "White", "black_nh": "Black or African American",
                "asian_nh": "Asian", "aian_nh": "American Indian or Alaska Native",
                "nhpi_nh": "Native Hawaiian or Other Pacific Islander", "hispanic": "Hispanic or Latino"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 4), round(float(c + h), 4))


def load_population() -> pd.DataFrame:
    frames, n0 = [], 0
    for f in sorted(HMDA.glob("*.csv.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            # everything as category: 6.9M rows of str would not fit in memory
            df = pd.read_csv(fh, usecols=USECOLS, dtype="category", low_memory=False)
        n0 += len(df)
        keep = ((df["action_taken"].isin(["1", "3"])) & (df["loan_purpose"].isin(["1", "31", "32"]))
                & (df["lien_status"] == "1") & (df["construction_method"] == "1")
                & (df["total_units"].isin(["1", "2", "3", "4"])) & (df["occupancy_type"] == "1")
                & (df["reverse_mortgage"] == "2") & (df["business_or_commercial_purpose"] == "2"))
        df = df[keep.to_numpy()]
        for c in NUMERIC:                                   # NA / Exempt -> NaN, via the categories
            vals = pd.to_numeric(pd.Index(df[c].cat.categories), errors="coerce").to_numpy(dtype=float)
            codes = df[c].cat.codes.to_numpy()
            df[c] = np.where(codes >= 0, vals[np.clip(codes, 0, None)], np.nan)
        for c in df.columns:
            if str(df[c].dtype) == "category":
                df[c] = df[c].astype(str)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["y"] = (df["action_taken"] == "3").astype(int)
    df["co_applicant"] = np.where(df["co-applicant_age"] == "9999", "none", "present")
    df["combined_group"] = np.where(df["derived_ethnicity"] == "Hispanic or Latino",
                                    "Hispanic or Latino", df["derived_race"])
    # geography-only proxy: modal group of the applicant's census tract
    acs = pd.read_csv(ACS, dtype={"tract": str})
    shares = acs.set_index("tract")[list(PROXY_GROUPS)]
    modal = shares.idxmax(axis=1).map(PROXY_GROUPS)
    modal[shares.sum(axis=1) == 0] = np.nan
    df["proxy_group"] = df["census_tract"].map(modal).fillna("tract not matched")
    return df, {"rows_fetched": n0, "rows_qualifying": int(len(df)),
                "denial_rate": round(float(df["y"].mean()), 4),
                "tract_match_rate": round(float((df["proxy_group"] != "tract not matched").mean()), 4)}


def design(df: pd.DataFrame):
    X = df[NUMERIC].to_numpy(dtype=float)
    cats = []
    for c in CATEGORICAL:
        codes, _ = pd.factorize(df[c].fillna("NA"))
        cats.append(codes.astype(float))
    X = np.column_stack([X] + cats)
    cat_mask = np.array([False] * len(NUMERIC) + [True] * len(CATEGORICAL))
    return X, cat_mask


def audit(conf, probs_cal, y_cal, g_cal, probs_te, y_te, g_te):
    """Per-group coverage (marginal and Mondrian), set sizes, REVIEW rate."""
    q = conf.qhat(ALPHA)
    names = sorted(set(g_te))
    cal_names = [n for n in names if (g_cal == n).sum() >= THIN_CAL]
    codes = {n: i for i, n in enumerate(cal_names)}
    c_cal = np.array([codes.get(s, -1) for s in g_cal])
    mond = MondrianConformal(probs_cal[c_cal >= 0], y_cal[c_cal >= 0], c_cal[c_cal >= 0], cal_names)
    rows = []
    for name in names:
        m = g_te == name
        n = int(m.sum())
        p, y = probs_te[m], y_te[m]
        s_true = 1.0 - p[np.arange(n), y]
        sizes = (1.0 - p <= q).sum(axis=1)
        review = int((sizes != 1).sum())
        row = {"group": name, "n": n, "cal_n": int((g_cal == name).sum()),
               "thin": n < THIN_TEST or name not in codes,
               "marginal_coverage": round(float((s_true <= q).mean()), 4) if n else None,
               "review_rate": round(review / n, 4) if n else None,
               "review_ci": wilson(review, n),
               "set_size_dist": {str(k): round(float((sizes == k).mean()), 4) for k in (0, 1, 2)} if n else None,
               "denial_rate": round(float(y.mean()), 4) if n else None}
        if name in codes:
            qg = mond.qhat(ALPHA, name)
            row["mondrian_coverage"] = round(float((s_true <= qg).mean()), 4)
            row["mondrian_qhat"] = round(float(qg), 4)
        rows.append(row)
    ok = [r for r in rows if not r["thin"]]
    rates = [r["review_rate"] for r in ok]
    return {"groups": rows,
            "four_fifths_review": round(min(rates) / max(rates), 4) if len(rates) >= 2 and max(rates) > 0 else None,
            "worst_marginal": min(ok, key=lambda r: r["marginal_coverage"])["group"] if ok else None,
            "marginal_spread": round(max(r["marginal_coverage"] for r in ok)
                                     - min(r["marginal_coverage"] for r in ok), 4) if ok else None}


def run_seed(df, X, cat_mask, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), N_MODEL + N_CAL + N_TEST, replace=False)
    y = df["y"].to_numpy()
    idx_fit, idx_te = train_test_split(idx, test_size=N_TEST, random_state=seed, stratify=y[idx])
    idx_tr, idx_cal = train_test_split(idx_fit, test_size=N_CAL, random_state=seed, stratify=y[idx_fit])
    t0 = time.time()
    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed,
                                           categorical_features=cat_mask).fit(X[idx_tr], y[idx_tr])
    p_cal, p_te = model.predict_proba(X[idx_cal]), model.predict_proba(X[idx_te])
    y_cal, y_te = y[idx_cal], y[idx_te]
    conf = ConformalCalibrator(p_cal, y_cal)
    q = conf.qhat(ALPHA)
    sizes = (1.0 - p_te <= q).sum(axis=1)
    out = {"model": {"auc": round(float(roc_auc_score(y_te, p_te[:, 1])), 4),
                     "accuracy": round(float(np.mean((p_te[:, 1] >= 0.5) == y_te)), 4)},
           "marginal_coverage": round(float(conf.empirical_coverage(p_te, y_te, ALPHA)), 4),
           "review_rate": round(float(np.mean(sizes != 1)), 4),
           "attributes": {}, "seconds": None}
    for attr in ("derived_race", "derived_ethnicity", "derived_sex", "combined_group"):
        g = df[attr].to_numpy()
        out["attributes"][attr] = audit(conf, p_cal, y_cal, g[idx_cal], p_te, y_te, g[idx_te])
    g = df["proxy_group"].to_numpy()
    out["attributes"]["proxy_group"] = audit(conf, p_cal, y_cal, g[idx_cal], p_te, y_te, g[idx_te])
    rep, prox = df["combined_group"].to_numpy()[idx_te], g[idx_te]
    in_scope = np.isin(rep, list(PROXY_GROUPS.values())) & (prox != "tract not matched")
    out["proxy_vs_reported"] = {
        "test_rows_in_scope": int(in_scope.sum()),
        "share_in_scope": round(float(in_scope.mean()), 4),
        "mismatch_rate": round(float((rep[in_scope] != prox[in_scope]).mean()), 4),
        "per_group": []}
    A = {r["group"]: r for r in out["attributes"]["combined_group"]["groups"]}
    B = {r["group"]: r for r in out["attributes"]["proxy_group"]["groups"]}
    for name in PROXY_GROUPS.values():
        a, b = A.get(name), B.get(name)
        if a is None or b is None:
            continue
        lo, hi = a["review_ci"]
        out["proxy_vs_reported"]["per_group"].append({
            "group": name, "n_reported": a["n"], "n_proxy": b["n"],
            "review_reported": a["review_rate"], "review_proxy": b["review_rate"],
            "review_delta": round(b["review_rate"] - a["review_rate"], 4),
            "proxy_outside_reported_ci": not (lo <= b["review_rate"] <= hi) if lo is not None else None,
            "coverage_reported": a["marginal_coverage"], "coverage_proxy": b["marginal_coverage"],
            "thin": a["thin"] or b["thin"]})
    out["seconds"] = round(time.time() - t0)
    return out


def agg(v):
    a = np.asarray([x for x in v if x is not None], dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "n": int(len(a))} if len(a) else None


def summarise(per_seed):
    S = {"seeds": len(per_seed),
         "auc": agg([s["model"]["auc"] for s in per_seed]),
         "marginal_coverage": agg([s["marginal_coverage"] for s in per_seed]),
         "review_rate": agg([s["review_rate"] for s in per_seed]), "attributes": {}}
    for attr in per_seed[0]["attributes"]:
        names = [r["group"] for r in per_seed[0]["attributes"][attr]["groups"]]
        rows = []
        for name in names:
            cells = [r for s in per_seed for r in s["attributes"][attr]["groups"] if r["group"] == name]
            rows.append({"group": name, "n": int(np.mean([c["n"] for c in cells])),
                         "cal_n": int(np.mean([c["cal_n"] for c in cells])),
                         "thin": any(c["thin"] for c in cells),
                         "denial_rate": agg([c["denial_rate"] for c in cells]),
                         "marginal_coverage": agg([c["marginal_coverage"] for c in cells]),
                         "mondrian_coverage": agg([c.get("mondrian_coverage") for c in cells]),
                         "review_rate": agg([c["review_rate"] for c in cells]),
                         "review_ci_seed0": cells[0]["review_ci"],
                         "set_size_2": agg([c["set_size_dist"]["2"] for c in cells if c["set_size_dist"]]),
                         "set_size_0": agg([c["set_size_dist"]["0"] for c in cells if c["set_size_dist"]])})
        S["attributes"][attr] = {
            "groups": rows,
            "four_fifths_review": agg([s["attributes"][attr]["four_fifths_review"] for s in per_seed]),
            "marginal_spread": agg([s["attributes"][attr]["marginal_spread"] for s in per_seed])}
    pv = [s["proxy_vs_reported"] for s in per_seed]
    S["proxy_vs_reported"] = {"share_in_scope": agg([p["share_in_scope"] for p in pv]),
                              "mismatch_rate": agg([p["mismatch_rate"] for p in pv]), "per_group": []}
    for name in PROXY_GROUPS.values():
        cells = [g for p in pv for g in p["per_group"] if g["group"] == name]
        if not cells:
            continue
        S["proxy_vs_reported"]["per_group"].append({
            "group": name, "n_reported": int(np.mean([c["n_reported"] for c in cells])),
            "n_proxy": int(np.mean([c["n_proxy"] for c in cells])),
            "review_reported": agg([c["review_reported"] for c in cells]),
            "review_proxy": agg([c["review_proxy"] for c in cells]),
            "review_delta": agg([c["review_delta"] for c in cells]),
            "coverage_delta": agg([c["coverage_proxy"] - c["coverage_reported"] for c in cells]),
            "proxy_outside_reported_ci_seeds": int(sum(bool(c["proxy_outside_reported_ci"]) for c in cells)),
            "thin": any(c["thin"] for c in cells)})
    S["hypotheses"] = hypotheses(S)
    return S


def hypotheses(S):
    h = {}
    non_thin = [r for a in ("derived_race", "derived_ethnicity", "derived_sex")
                for r in S["attributes"][a]["groups"] if not r["thin"] and r["mondrian_coverage"]]
    h["H-C1_mondrian_within_0.01_every_non_thin_group"] = all(
        abs(r["mondrian_coverage"]["mean"] - 0.95) <= 0.01 for r in non_thin)
    h["H-C1_worst_mondrian"] = min(r["mondrian_coverage"]["mean"] for r in non_thin) if non_thin else None
    spreads = [S["attributes"][a]["marginal_spread"]["mean"] for a in ("derived_race", "derived_ethnicity", "derived_sex")]
    h["H-C2_marginal_spread_over_0.01_some_attribute"] = max(spreads) > 0.01
    h["H-C2_max_marginal_spread"] = max(spreads)
    pg = [g for g in S["proxy_vs_reported"]["per_group"] if not g["thin"]]
    h["H-C3_proxy_misstates_some_group_beyond_ci"] = any(g["proxy_outside_reported_ci_seeds"] >= 3 for g in pg)
    return h


def write_markdown(R):
    S, P = R["summary"], R["population"]
    f = lambda a: "" if a is None else f"{a['mean']:.3f} ({a['std']:.3f})"  # noqa: E731
    L = ["# Workstream C: HMDA 2025 coverage parity on reported protected class (Amendment 2, section 2.5)", "",
         f"Generated by `eval/expansion/run_hmda_parity.py` on {R['meta']['date']}. "
         "**The label is the lender's decision (denied = 1), not a repayment outcome.** "
         f"Population after the pre-specified filters: {P['rows_qualifying']:,} applications "
         f"(from {P['rows_fetched']:,} fetched rows across {P['states']} states), denial rate "
         f"{P['denial_rate']:.3f}. Per seed: {R['meta']['sizes']['model']:,} model rows, "
         f"{R['meta']['sizes']['cal']:,} calibration rows, {R['meta']['sizes']['test']:,} test rows; "
         f"{S['seeds']} seeds. Model AUC {f(S['auc'])}. Marginal coverage "
         f"{f(S['marginal_coverage'])} at target 0.95; REVIEW rate {f(S['review_rate'])}. "
         "Deviation recorded: the pre-specified feature 'interest rate presence' was dropped "
         "because HMDA reports interest_rate only on originated loans (it is the label).", ""]
    for attr, title in (("derived_race", "Race (reported)"), ("derived_ethnicity", "Ethnicity (reported)"),
                        ("derived_sex", "Sex (reported)"), ("combined_group", "Combined race/ethnicity (reported)"),
                        ("proxy_group", "Geography-only proxy (tract modal group)")):
        A = S["attributes"][attr]
        L += [f"## {title}", "", "| group | n test | n cal | denial rate | marginal cov | Mondrian cov "
              "| REVIEW rate | set size 2 | set size 0 | thin |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in A["groups"]:
            L.append(f"| {r['group']} | {r['n']:,} | {r['cal_n']:,} | {f(r['denial_rate'])} "
                     f"| {f(r['marginal_coverage'])} | {f(r['mondrian_coverage'])} | {f(r['review_rate'])} "
                     f"| {f(r['set_size_2'])} | {f(r['set_size_0'])} | {'yes' if r['thin'] else ''} |")
        L += ["", f"Four-fifths ratio of REVIEW rates (non-thin groups): {f(A['four_fifths_review'])}; "
              f"spread of marginal coverage across non-thin groups: {f(A['marginal_spread'])}.", ""]
    PV = S["proxy_vs_reported"]
    L += ["## Proxy vs reported", "",
          f"Rows in scope (reported group is one the proxy can produce, tract matched): "
          f"{f(PV['share_in_scope'])} of test rows. Proxy assigns a different group than reported "
          f"for {f(PV['mismatch_rate'])} of them. This is the geocoding component only (no surnames "
          "in the modified LAR), so it is a BIG proxy, not BISG.", "",
          "| group | n reported | n proxy | REVIEW reported | REVIEW proxy | delta | coverage delta "
          "| proxy outside reported 95% CI (seeds of 5) | thin |", "|---|---|---|---|---|---|---|---|---|"]
    for g in PV["per_group"]:
        L.append(f"| {g['group']} | {g['n_reported']:,} | {g['n_proxy']:,} | {f(g['review_reported'])} "
                 f"| {f(g['review_proxy'])} | {f(g['review_delta'])} | {f(g['coverage_delta'])} "
                 f"| {g['proxy_outside_reported_ci_seeds']} | {'yes' if g['thin'] else ''} |")
    L += ["", "## Pre-specified expectations", ""]
    for k, v in S["hypotheses"].items():
        L.append(f"- {k}: {'met' if v is True else 'not met' if v is False else v}")
    (OUT / "HMDA_PARITY.md").write_text("\n".join(L) + "\n")


def main():
    t0 = time.time()
    df, pop = load_population()
    pop["states"] = len(list(HMDA.glob("*.csv.gz")))
    print(f"population: {pop} ({time.time() - t0:.0f}s)", flush=True)
    X, cat_mask = design(df)
    per_seed = []
    for seed in SEEDS:
        r = run_seed(df, X, cat_mask, seed)
        per_seed.append(r)
        print(f"  seed {seed}: auc {r['model']['auc']:.3f} cov {r['marginal_coverage']:.4f} "
              f"review {r['review_rate']:.3f} ({r['seconds']}s)", flush=True)
    R = {"meta": {"date": time.strftime("%Y-%m-%d"), "alpha": ALPHA, "seeds": list(SEEDS),
                  "sizes": {"model": N_MODEL, "cal": N_CAL, "test": N_TEST},
                  "features": NUMERIC + CATEGORICAL,
                  "deviation": "interest rate presence dropped: reported only on originated loans",
                  "protocol": "PREREGISTRATION_AMENDMENT.md, Amendment 2, section 2.5"},
         "population": pop, "summary": summarise(per_seed), "per_seed": per_seed}
    (OUT / "hmda_parity.json").write_text(json.dumps(R, indent=1))
    write_markdown(R)
    print(f"wrote {OUT / 'hmda_parity.json'} and HMDA_PARITY.md ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
