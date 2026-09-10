"""Shifted-data selective-prediction pilot on Taiwan default (exploratory).

Answers exactly the deck's falsifiable question, but in the SHIFTED regime:

  Flag the 10% least reliable predictions. How many bad approvals (approved,
  then defaulted) does that 10% catch, relative to what the model's own
  confidence catches at the same referral rate?

  ratio = bad_approval_capture@10%(signal) / bad_approval_capture@10%(confidence)

Protocol
--------
Train and calibrate on a REFERENCE subpopulation; evaluate on a DISJOINT,
genuinely different subpopulation. No synthetic noise: every shift is a real
split of real applicants. An i.i.d. control runs the identical pipeline on a
random split so the shifted ratio is comparable to the calm-regime ratio.

Regimes
  iid      random split (control)
  pay0     train PAY_0 <= 0 (up to date)    -> test PAY_0 >= 1 (delayed)
  limit    train LIMIT_BAL > median (prime) -> test LIMIT_BAL <= median
  util     train utilisation <= median      -> test utilisation > median

Baselines: model confidence (MSP), 5-member bagged ensemble (mean confidence
and disagreement), off-the-shelf split-conformal p-value. Two reference points
are also reported: `random` (uninformative referral) and `oracle` (perfect
ranking, i.e. the ceiling imposed by the referral rate and the bad-approval
base rate).

Usage: .venv/bin/python eval/run_shift_bench.py
Writes results/shift_bench.json
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from reliax_core.auditor import ErrorAuditor                     # noqa: E402
from reliax_core.conformal import ConformalCalibrator            # noqa: E402
from reliax_core.ood import KNNOODDetector                       # noqa: E402
from reliax_core.scoring import reliability_score                # noqa: E402
from reliax_core.sl_fusion import fuse_signals                   # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator           # noqa: E402

import data as D                                                     # noqa: E402
import methods as M                                                  # noqa: E402

RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
ALPHA = 0.05
REFERRAL = 0.10
SEEDS = range(5)
N_TRAIN, N_CAL, N_TEST = 8000, 4000, 4000
N_BOOT = 2000
BASELINE = "msp"          # "the model's own confidence" in the deck's claim


def regimes(X, y, feature_names):
    p0 = feature_names.index("PAY_0")
    lb = feature_names.index("LIMIT_BAL")
    b1 = feature_names.index("BILL_AMT1")
    util = X[:, b1] / np.maximum(X[:, lb], 1.0)
    lb_med, util_med = np.median(X[:, lb]), np.median(util)
    n = len(y)
    return {
        "iid":   (np.ones(n, bool), np.ones(n, bool), "random split (control)"),
        "pay0":  (X[:, p0] <= 0, X[:, p0] >= 1, "PAY_0<=0 -> PAY_0>=1 (payment delay)"),
        "limit": (X[:, lb] > lb_med, X[:, lb] <= lb_med, "high limit -> low limit"),
        "util":  (util <= util_med, util > util_med, "low utilisation -> high utilisation"),
    }


def draw(rng, ref_mask, test_mask, disjoint):
    """Sample train/cal from the reference pool and test from the shifted pool."""
    ref_idx = np.flatnonzero(ref_mask)
    test_idx = np.flatnonzero(test_mask)
    if not disjoint:                                   # i.i.d. control: one pool, no overlap
        pool = rng.permutation(ref_idx)
        need = N_TRAIN + N_CAL + N_TEST
        pool = pool[:need]
        return pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_CAL], pool[N_TRAIN + N_CAL:]
    ref = rng.permutation(ref_idx)
    te = rng.permutation(test_idx)[:min(N_TEST, len(test_idx))]
    return ref[:N_TRAIN], ref[N_TRAIN:N_TRAIN + N_CAL], te


def signals_for(X_tr, y_tr, X_cal, y_cal, X_te, seed):
    model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X_tr, y_tr)
    probs_cal, probs_te = model.predict_proba(X_cal), model.predict_proba(X_te)
    logit_cal, logit_te = model.decision_function(X_cal), model.decision_function(X_te)

    msp = np.maximum(probs_te[:, 1], probs_te[:, 0])
    margin = 2 * msp - 1
    conf = ConformalCalibrator(probs_cal, y_cal)
    qhat = conf.qhat(ALPHA)
    set_sizes = (1.0 - probs_te <= qhat).sum(axis=1)
    pvals = np.array([conf.p_value(float(p)) for p in msp])
    va = VennAbersCalibrator(probs_cal[:, 1], y_cal)
    va_iv = [va.interval(float(s)) for s in probs_te[:, 1]]
    va_width = np.array([v["width"] for v in va_iv])
    ood = KNNOODDetector(k=10).fit(X_cal)
    ood_pct = ood.percentiles_batch(X_te)
    auditor = ErrorAuditor(seed=seed).fit(X_cal, probs_cal, y_cal, model.predict(X_cal))
    p_err = auditor.p_error_batch(X_te, probs_te)

    composite = np.array([
        reliability_score(float(margin[i]), float(pvals[i]), int(set_sizes[i]),
                          float(ood_pct[i]), bool(ood_pct[i] > 99.0), float(va_width[i]),
                          float(p_err[i]), bool(p_err[i] > auditor.flag_threshold), False)
        for i in range(len(msp))])
    sl = np.array([
        fuse_signals(float(msp[i]), float(pvals[i]), int(set_sizes[i]), float(va_width[i]),
                     float(p_err[i]), float(ood_pct[i]), "OK",
                     1.0 - auditor.base_error_rate)["expected"]
        for i in range(len(msp))])

    # baseline: 5-member bagged ensemble, used only to rank the deployed model's calls
    members = []
    rs = np.random.default_rng(seed)
    for m in range(5):
        sub = rs.choice(len(y_tr), int(0.8 * len(y_tr)), replace=False)
        members.append(HistGradientBoostingClassifier(max_iter=300, random_state=1000 + m)
                       .fit(X_tr[sub], y_tr[sub]).predict_proba(X_te)[:, 1])
    ens = np.vstack(members)
    ens_mean = ens.mean(axis=0)
    ens_conf = np.maximum(ens_mean, 1.0 - ens_mean)
    ens_disagree = -ens.std(axis=0)

    sig = {
        "msp": msp,
        "ensemble_conf": ens_conf,
        "ensemble_disagreement": ens_disagree,
        "conformal_pvalue": pvals,
        "va_width": -va_width,
        "ood_knn": -ood_pct,
        "auditor": -p_err,
        "composite": composite,
        "sl_fusion": sl,
    }
    return model, probs_te, sig


def boot_ratio(rng, sig, base, bad, rate, n_boot):
    """Bootstrap the capture RATIO over test points (paired: same resample both arms)."""
    n = len(bad)
    out = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        d = M.default_capture(base[b], bad[b], rate)
        if d > 0:
            out.append(M.default_capture(sig[b], bad[b], rate) / d)
    a = np.asarray(out)
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))] if len(a) else [None, None]


def main():
    ds = D.load_taiwan()
    X, y, fn = ds["X"], ds["y"], ds["feature_names"]
    regs = regimes(X, y, fn)
    out = {"dataset": "taiwan_default_uci350", "referral_rate": REFERRAL,
           "baseline": BASELINE, "alpha": ALPHA, "seeds": list(SEEDS),
           "n_train": N_TRAIN, "n_cal": N_CAL, "n_test": N_TEST, "regimes": {}}

    for reg, (ref_mask, test_mask, desc) in regs.items():
        disjoint = reg != "iid"
        per_seed, ratios = [], {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            i_tr, i_cal, i_te = draw(rng, ref_mask, test_mask, disjoint)
            model, probs_te, sig = signals_for(X[i_tr], y[i_tr], X[i_cal], y[i_cal],
                                               X[i_te], seed)
            y_te = y[i_te]
            pred = (probs_te[:, 1] >= 0.5).astype(int)
            wrong = (pred != y_te).astype(int)
            bad = ((pred == 0) & (y_te == 1)).astype(int)     # approved, then defaulted
            # reference points: random referral, and the perfect-ranking ceiling
            sig["random"] = np.random.default_rng(10_000 + seed).random(len(y_te))
            sig["oracle"] = -bad.astype(float)
            row = {"accuracy": float((pred == y_te).mean()),
                   "auc": float(roc_auc_score(y_te, probs_te[:, 1])),
                   "default_rate": float(y_te.mean()),
                   "bad_approval_rate": float(bad.mean()),
                   "coverage_at_95": float(
                       ConformalCalibrator(model.predict_proba(X[i_cal]), y[i_cal])
                       .empirical_coverage(probs_te, y_te, ALPHA)),
                   "signals": {}}
            base_cap = M.default_capture(sig[BASELINE], bad, REFERRAL)
            for name, s in sig.items():
                cap = M.default_capture(s, bad, REFERRAL)
                row["signals"][name] = {
                    "bad_approval_capture@0.1": float(cap),
                    "err_capture@0.1": float(M.error_capture(s, wrong, REFERRAL)),
                    "aurc": float(M.aurc(s, wrong)),
                    "ratio_vs_baseline": float(cap / base_cap) if base_cap > 0 else None,
                }
                ratios.setdefault(name, []).append(cap / base_cap if base_cap > 0 else np.nan)
            if seed == 0:      # bootstrap CI on the ratio, seed 0
                bs = np.random.default_rng(12345)
                row["ratio_ci95"] = {n: boot_ratio(bs, s, sig[BASELINE], bad, REFERRAL, N_BOOT)
                                     for n, s in sig.items()}
            per_seed.append(row)

        agg = {"description": desc,
               "n_test": int(min(N_TEST, test_mask.sum())) if disjoint else N_TEST,
               "accuracy": _ms([r["accuracy"] for r in per_seed]),
               "auc": _ms([r["auc"] for r in per_seed]),
               "default_rate": _ms([r["default_rate"] for r in per_seed]),
               "bad_approval_rate": _ms([r["bad_approval_rate"] for r in per_seed]),
               "coverage_at_95": _ms([r["coverage_at_95"] for r in per_seed]),
               "signals": {}}
        for name in per_seed[0]["signals"]:
            agg["signals"][name] = {
                "bad_approval_capture@0.1": _ms(
                    [r["signals"][name]["bad_approval_capture@0.1"] for r in per_seed]),
                "err_capture@0.1": _ms(
                    [r["signals"][name]["err_capture@0.1"] for r in per_seed]),
                "aurc": _ms([r["signals"][name]["aurc"] for r in per_seed]),
                "ratio_vs_baseline": _ms(ratios[name]),
                "ratio_ci95_seed0": per_seed[0]["ratio_ci95"][name],
            }
        out["regimes"][reg] = agg
        print(f"[done] {reg}", flush=True)

    (RESULTS / "shift_bench.json").write_text(json.dumps(out, indent=2))
    print("wrote results/shift_bench.json")


def _ms(v):
    a = np.asarray(v, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


if __name__ == "__main__":
    main()
