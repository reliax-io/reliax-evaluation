"""Label noise in the TRAINING data, clean calibration data (EXPLORATORY, Taiwan).

Question (raised by a reviewer of the whitepaper, 28 Sep 2026): when the base
model was trained on partly wrong labels but the calibration outcomes are
right, (a) does the conformal guarantee still hold, (b) does routing on the
certificate catch more mistakes than a confidence cut at the same review rate,
and (c) does the error auditor, which learns from the clean calibration
outcomes, catch more?

Design: the run_eval.py protocol (stratified 50/25/25, HistGradientBoosting
300 iterations, 5 seeds) with the TRAINING labels corrupted and the calibration
and test labels left clean.
  symmetric_p   : each training label flipped with probability p
  missed_def_p  : each training DEFAULT relabelled "repay" with probability p
                  (the realistic case: defaults that were never recorded)
Reported per condition: model accuracy / AUC / ECE on the clean test split;
coverage and REVIEW rate of the certificate at alpha 0.05 and 0.10; bad
approvals and wrong rejections per 1,000 applications, all vs auto-decided;
share of each caught by REVIEW, by a random referral and by a confidence cut
at the same rate; the auditor's capture at the same rate and at a fixed 10%.
Not pre-registered. Writes results/label_noise.json.
Usage: python eval/run_label_noise.py [--fast]
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
from reliax_core.conformal import ConformalCalibrator      # noqa: E402
from reliax_core.auditor import ErrorAuditor                # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator      # noqa: E402
import data as D                                            # noqa: E402
import methods as M                                         # noqa: E402
import models as MODELS                                     # noqa: E402

RESULTS = MODELS.results_dir(BASE)
ALPHAS = (0.05, 0.10)
CONDITIONS = [("clean", None, 0.0), ("symmetric_0.10", "sym", 0.10), ("symmetric_0.20", "sym", 0.20),
              ("symmetric_0.30", "sym", 0.30), ("missed_def_0.30", "miss", 0.30), ("missed_def_0.50", "miss", 0.50)]


def agg(v):
    a = np.asarray(v, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


def corrupt(y_tr, kind, p, rng):
    y = y_tr.copy()
    if kind == "sym":
        flip = rng.random(len(y)) < p
        y[flip] = 1 - y[flip]
    elif kind == "miss":
        flip = (y == 1) & (rng.random(len(y)) < p)
        y[flip] = 0
    return y, float((y != y_tr).mean())


def caught(mask, what):
    return float((mask & what).sum() / max(what.sum(), 1))


def referral(score_low_is_bad, k):
    m = np.zeros(len(score_low_is_bad), bool)
    m[np.argsort(score_low_is_bad)[:k]] = True
    return m


def run(ds, cond, kind, p, seed, fast):
    X, y = ds["X"], ds["y"]
    idx = np.arange(len(y))
    idx_tr, idx_rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=y)
    idx_cal, idx_te = train_test_split(idx_rest, test_size=0.5, random_state=seed, stratify=y[idx_rest])
    rng = np.random.default_rng(1000 + seed)
    y_tr_noisy, actual = corrupt(y[idx_tr], kind, p, rng)
    model = MODELS.make_model("taiwan", seed).fit(X[idx_tr], y_tr_noisy)
    X_cal, y_cal, X_te, y_te = X[idx_cal], y[idx_cal], X[idx_te], y[idx_te]
    probs_cal, probs_te = model.predict_proba(X_cal), model.predict_proba(X_te)
    pred = probs_te.argmax(axis=1)
    wrong = pred != y_te
    approve, reject = pred == 0, pred == 1
    bad = approve & (y_te == 1)
    badrej = reject & (y_te == 0)
    msp = probs_te.max(axis=1)
    n = len(y_te)
    out = {"noise_actual": actual,
           "model": {"accuracy": float((pred == y_te).mean()), "auc": float(roc_auc_score(y_te, probs_te[:, 1])),
                     "default_rate_predicted": float(pred.mean())}}
    # calibration: raw, temperature, Venn-Abers point
    eps = 1e-6
    logit_cal = np.log(np.clip(probs_cal[:, 1], eps, 1 - eps) / np.clip(1 - probs_cal[:, 1], eps, 1 - eps))
    logit_te = np.log(np.clip(probs_te[:, 1], eps, 1 - eps) / np.clip(1 - probs_te[:, 1], eps, 1 - eps))
    t = M.fit_temperature(logit_cal, y_cal)
    p_t = 1.0 / (1.0 + np.exp(-logit_te / t))
    cal = {"ece_raw": M.ece(probs_te[:, 1], y_te), "ece_tscaled": M.ece(p_t, y_te), "temperature": t}
    if not fast:
        va = VennAbersCalibrator(probs_cal[:, 1], y_cal)
        sub = np.arange(n) if n <= 3000 else np.random.default_rng(seed).choice(n, 3000, replace=False)
        va_point = np.array([va.interval(float(s))["point"] for s in probs_te[sub, 1]])
        cal["ece_venn_abers"] = M.ece(va_point, y_te[sub])
        cal["ece_raw_same_subsample"] = M.ece(probs_te[sub, 1], y_te[sub])
    out["calibration"] = cal
    # auditor trained on the CLEAN calibration split
    auditor = ErrorAuditor(seed=seed).fit(X_cal, probs_cal, y_cal, model.predict(X_cal))
    p_err = auditor.p_error_batch(X_te, probs_te)
    # certificate at each alpha
    conf = ConformalCalibrator(probs_cal, y_cal)
    rng_ref = np.random.default_rng(seed)
    for a in ALPHAS:
        q = conf.qhat(a)
        sizes = (1.0 - probs_te <= q).sum(axis=1)
        review = sizes != 1
        k = int(review.sum())
        conf_ref = referral(msp, k)
        aud_ref = referral(-p_err, k)
        rand_ref = np.zeros(n, bool); rand_ref[rng_ref.choice(n, k, replace=False)] = True
        cov = float(np.mean([(yy in np.flatnonzero(1.0 - probs_te[i] <= q)) for i, yy in enumerate(y_te)]))
        out[f"alpha_{a}"] = {
            "coverage": cov, "review_rate": float(review.mean()),
            "bad_per_1000_all": 1000 * bad.mean(), "bad_per_1000_auto": 1000 * (bad & ~review).mean(),
            "badrej_per_1000_all": 1000 * badrej.mean(), "badrej_per_1000_auto": 1000 * (badrej & ~review).mean(),
            "bad_caught_review": caught(review, bad), "bad_caught_confidence": caught(conf_ref, bad),
            "bad_caught_random": caught(rand_ref, bad), "bad_caught_auditor": caught(aud_ref, bad),
            "badrej_caught_review": caught(review, badrej), "badrej_caught_confidence": caught(conf_ref, badrej),
            "badrej_caught_auditor": caught(aud_ref, badrej),
            "errors_caught_review": caught(review, wrong), "errors_caught_confidence": caught(conf_ref, wrong),
            "errors_caught_random": caught(rand_ref, wrong), "errors_caught_auditor": caught(aud_ref, wrong),
            "bad_per_1000_auto_if_auditor_referral": 1000 * (bad & ~aud_ref).mean(),
        }
    # fixed 10% referral, the selective-prediction convention
    k10 = int(round(0.10 * n))
    c10, a10 = referral(msp, k10), referral(-p_err, k10)
    out["referral_10pct"] = {"bad_caught_confidence": caught(c10, bad), "bad_caught_auditor": caught(a10, bad),
                             "errors_caught_confidence": caught(c10, wrong), "errors_caught_auditor": caught(a10, wrong),
                             "badrej_caught_confidence": caught(c10, badrej), "badrej_caught_auditor": caught(a10, badrej)}
    return out


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, prefix + k + ".")
        else:
            yield prefix + k, v


def main():
    fast = "--fast" in sys.argv
    ds = D.load_taiwan()
    seeds = range(5)
    out = {"meta": {"dataset": "taiwan", "seeds": len(seeds), "alphas": ALPHAS, "status": "exploratory, not pre-registered", "base_model": MODELS.describe(),
                    "noise_applied_to": "training labels only; calibration and test labels clean",
                    "venn_abers_subsample": None if fast else 3000}}
    t0 = time.perf_counter()
    for name, kind, p in CONDITIONS:
        rows = [run(ds, name, kind, p, s, fast) for s in seeds]
        keys = [k for k, _ in flatten(rows[0])]
        out[name] = {k: agg([dict(flatten(r))[k] for r in rows]) for k in keys}
        c = out[name]
        print(f"{name:16s} acc {c['model.accuracy']['mean']:.3f} auc {c['model.auc']['mean']:.3f} ece {c['calibration.ece_raw']['mean']:.3f} | "
              f"a.05 cov {c['alpha_0.05.coverage']['mean']:.3f} review {c['alpha_0.05.review_rate']['mean']:.0%} "
              f"bad/1000 {c['alpha_0.05.bad_per_1000_all']['mean']:.0f}->{c['alpha_0.05.bad_per_1000_auto']['mean']:.0f} "
              f"caught rev {c['alpha_0.05.bad_caught_review']['mean']:.2f} conf {c['alpha_0.05.bad_caught_confidence']['mean']:.2f} "
              f"aud {c['alpha_0.05.bad_caught_auditor']['mean']:.2f} rand {c['alpha_0.05.bad_caught_random']['mean']:.2f} | "
              f"10%: conf {c['referral_10pct.bad_caught_confidence']['mean']:.3f} aud {c['referral_10pct.bad_caught_auditor']['mean']:.3f}", flush=True)
    out["meta"]["seconds"] = round(time.perf_counter() - t0, 1)
    (RESULTS / "label_noise.json").write_text(json.dumps(out, indent=1))
    print("wrote results/label_noise.json in", out["meta"]["seconds"], "s")


if __name__ == "__main__":
    main()
