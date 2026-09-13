"""Does the calibration stage avoid mistakes? A decision-level evaluation on real
credit data (Taiwan default, German credit; the whitepaper's models).

A lender auto-approves an applicant when the model's stated default
probability is below a cut-off tau. A bad approval is an auto-approved
applicant who then defaults. Policies compared on held-out applicants:

  P0  model alone            approve if stated PD < tau
  P1  calibrated bracket     approve if the Venn-Abers UPPER bound p1 < tau
                             (the theorem half of the calibration stage)
  P2  opinion referral       P0, then send to REVIEW every approval whose
                             (segment x score-bin) cell the calibration opinion
                             marks as untrustworthy: disbelief above d_max with
                             the default rate UNDER-stated, or insufficient
                             evidence. Referral rate r is whatever that yields.
  P3  both                   P1, then the P2 referral on top.

For P2 the honest comparators at the SAME referral rate r are
  random referral       refer r of the approvals at random (removes r of the bad approvals)
  confidence referral   refer the r of approvals with the highest stated PD
                        (what a lender without Reliax would do)
Metrics: bad approvals per 1,000 applicants, approvals kept, bad-approval
capture of the referred set relative to random and to confidence referral.

Usage: .venv/bin/python eval/run_calibration_decisions.py
Writes results/calibration_decisions.json
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from reliax_core.calibration_trust import CalibrationTrust  # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator      # noqa: E402

import data as D                                             # noqa: E402

RESULTS = BASE / "results"
TAUS = (0.3, 0.5)
D_MAX = 0.05          # cell disbelief above which the stated PD is not trusted
MIN_CELL_N = 30


def agg(v):
    a = np.asarray(v, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


def per_1000(mask, n):
    return 1000.0 * mask.sum() / n


def evaluate(p_raw, p1, y, cell_flag, tau):
    """cell_flag: per test applicant, True if the opinion says its cell's stated PD is untrustworthy."""
    n = len(y)
    approve0 = p_raw < tau
    bad0 = approve0 & (y == 1)
    approve1 = p1 < tau
    bad1 = approve1 & (y == 1)
    # P2: refer flagged approvals
    refer2 = approve0 & cell_flag
    r = refer2.sum() / max(approve0.sum(), 1)
    auto2 = approve0 & ~cell_flag
    bad2 = auto2 & (y == 1)
    caught2 = (refer2 & (y == 1)).sum()
    # comparators at the same referral rate among P0 approvals
    k = int(round(r * approve0.sum()))
    idx_appr = np.flatnonzero(approve0)
    rng = np.random.default_rng(0)
    caught_rand = []
    for _ in range(200):
        sel = rng.choice(idx_appr, k, replace=False) if k > 0 else np.array([], int)
        caught_rand.append((y[sel] == 1).sum())
    caught_rand = float(np.mean(caught_rand))
    order = idx_appr[np.argsort(-p_raw[idx_appr])]          # highest stated PD first
    caught_conf = float((y[order[:k]] == 1).sum())
    # P3: bracket then referral
    auto3 = approve1 & ~cell_flag
    bad3 = auto3 & (y == 1)
    return {
        "P0_bad_per_1000": per_1000(bad0, n), "P0_approvals_per_1000": per_1000(approve0, n),
        "P1_bad_per_1000": per_1000(bad1, n), "P1_approvals_per_1000": per_1000(approve1, n),
        "P2_bad_per_1000": per_1000(bad2, n), "P2_auto_approvals_per_1000": per_1000(auto2, n),
        "P2_referral_rate": float(r), "P2_referred_per_1000": per_1000(refer2, n),
        "P2_bad_caught": float(caught2), "random_bad_caught": caught_rand, "confidence_bad_caught": caught_conf,
        "P2_vs_random": float(caught2 / caught_rand) if caught_rand > 0 else np.nan,
        "P2_vs_confidence": float(caught2 / caught_conf) if caught_conf > 0 else np.nan,
        "P3_bad_per_1000": per_1000(bad3, n), "P3_auto_approvals_per_1000": per_1000(auto3, n),
        "bad_rate_among_P0_approvals": float(bad0.sum() / max(approve0.sum(), 1)),
        "bad_rate_among_P3_approvals": float(bad3.sum() / max(auto3.sum(), 1)),
    }


def run(ds, seeds):
    rows = {str(t): [] for t in TAUS}
    for seed in range(seeds):
        X, y = ds["X"], ds["y"]
        idx = np.arange(len(y))
        idx_tr, idx_rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
        idx_cal, idx_te = train_test_split(idx_rest, test_size=0.5, random_state=seed, stratify=y[idx_rest])
        model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X[idx_tr], y[idx_tr])
        p_cal, p_te = model.predict_proba(X[idx_cal])[:, 1], model.predict_proba(X[idx_te])[:, 1]
        y_cal, y_te = y[idx_cal], y[idx_te]
        seg_cal, seg_te = ds["segments"]["age_band"][idx_cal], ds["segments"]["age_band"][idx_te]
        a, b = train_test_split(np.arange(len(y_cal)), test_size=0.5, random_state=0, stratify=y_cal)
        va = VennAbersCalibrator(p_cal[a], y_cal[a])
        p1_te = np.array([va.interval(float(s))["p1"] for s in p_te])
        ct = CalibrationTrust(M=10, binning="quantile", representative="mean", evidence="rate",
                              debias="l2", min_cell_n=MIN_CELL_N).fit(p_cal[b], y_cal[b], segments=seg_cal[b])
        flag = np.zeros(len(y_te), bool)
        for i in range(len(y_te)):
            c = ct.lookup(float(p_te[i]), seg_te[i])
            understated = c["rate"] is not None and c["rp"] is not None and c["rate"] > c["rp"]
            flag[i] = c["insufficient_evidence"] or (c["opinion"]["d"] > D_MAX and understated)
        for t in TAUS:
            rows[str(t)].append(evaluate(p_te, p1_te, y_te, flag, t))
        r = rows["0.5"][-1]
        print(f"  {ds['name']} seed {seed}: tau=0.5 bad/1000 P0 {r['P0_bad_per_1000']:.0f} P1 {r['P1_bad_per_1000']:.0f} "
              f"P2 {r['P2_bad_per_1000']:.0f} (refer {r['P2_referral_rate']:.0%}, vs rand {r['P2_vs_random']:.2f}x, "
              f"vs conf {r['P2_vs_confidence']:.2f}x) P3 {r['P3_bad_per_1000']:.0f}")
    return {t: {k: agg([r[k] for r in rs]) for k in rs[0]} for t, rs in rows.items()}


def main():
    res = {"meta": {"taus": TAUS, "d_max": D_MAX, "min_cell_n": MIN_CELL_N,
                    "opinion": "quantile M=10, mean rep, rate evidence, l2, age-band cells"}}
    print("Taiwan")
    res["taiwan"] = run(D.load_taiwan(), 5)
    print("German")
    res["german"] = run(D.load_german(), 10)
    (RESULTS / "calibration_decisions.json").write_text(json.dumps(res, indent=1))
    print("wrote results/calibration_decisions.json")


if __name__ == "__main__":
    main()
