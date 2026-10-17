"""Confirmatory runs of Amendment 1 (PREREGISTRATION_AMENDMENT.md), on Give Me
Some Credit (2011) and Home Credit Default Risk (2018), with Taiwan run through
the same code as the exploratory third dataset.

Protocol, as written in Amendment 1 and its acceptance; nothing here is tuned:

  populations  declared per dataset in eval/confirmatory_data.py; train and
               calibrate on the reference population only
  sizes        8,000 training rows, 4,000 calibration rows, test sets of 4,000
               rows, 5 seeds (0 to 4); the seed changes the split and the model
  model        HistGradientBoostingClassifier(max_iter=300, random_state=seed);
               missing inputs filled with the training split's medians
  tiers        calm       lambda = 0
               moderate   lambda in {0.25, 0.50} of the test rows from the shifted population
               severe     lambda in {0.75, 1.00}
               corruption reference rows, rho in {0.05, 0.10} with the monetary fields x100
  signal       "gated": rows whose kNN distance to the calibration data exceeds
               its 95th calibration percentile are referred first, ordered by
               distance; all others by the model's confidence (max class
               probability). Implemented as in run_shift_bench_v2.py, whose
               values the amendment quotes: distance as its calibration
               percentile, ties at the top broken by confidence. Secondary: "regime_switch", confidence while the
               martingale on the kNN distance is OK, distance once it is WATCH
               or ALARM, evaluated in stream order. k = 10, WATCH 20, ALARM 100,
               as frozen in reliax-core.
  bar          at a 10% referral rate, on both moderate lambdas: bad-approval
               capture >= 1.5x random referral (mean over the 5 seeds) AND the
               95% paired bootstrap CI of capture(gated) / capture(confidence)
               above 1.0 (seed 0, 2,000 resamples, as in run_shift_bench.py).
               A lambda that costs the model under 3 AUC points is "mild" and does
               not count. The loss is measured, as in run_shift_bench.py, against
               the i.i.d. control: the same model trained, calibrated and tested on
               a random split of the whole population (seeded as there). PASS
               needs both confirmatory datasets to pass.
  severe       tripwire: the martingale reaches ALARM within the stream, and the
               REVIEW rate (prediction set not a singleton) rises over the calm tier
  corruption   the gated referral holds at least 80% of the corrupted rows (mean
               over seeds) and the martingale reaches ALARM in every seed

Usage
  .venv/bin/python eval/run_confirmatory.py                     # all three datasets
  .venv/bin/python eval/run_confirmatory.py taiwan              # one dataset
  .venv/bin/python eval/run_confirmatory.py --selftest          # synthetic stand-ins, no download
Writes results/confirmatory/<dataset>.json and results/confirmatory/summary.json
"""
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from reliax_core.conformal import ConformalCalibrator             # noqa: E402
from reliax_core.martingale import ALARM_THRESHOLD, ConformalMartingale, WATCH_THRESHOLD  # noqa: E402
from reliax_core.ood import KNNOODDetector                        # noqa: E402

import confirmatory_data as CD                                     # noqa: E402
import methods as M                                                # noqa: E402

SELFTEST = "--selftest" in sys.argv
OUT = BASE / "results" / ("confirmatory_selftest" if SELFTEST else "confirmatory")
ALPHA = 0.05
REFERRAL = 0.10
SEEDS = range(2) if SELFTEST else range(5)
N_TRAIN, N_CAL, N_TEST = 8000, 4000, 4000
N_BOOT = 2000
K = 10
FLAG_PCT = 95.0
MODERATE, SEVERE = (0.25, 0.50), (0.75, 1.00)
LAMBDAS = (0.0, *MODERATE, *SEVERE)
RHOS = (0.05, 0.10)
BAR_RANDOM, BAR_AUC_LOSS, BAR_CORRUPT = 1.5, 0.03, 0.80
CONFIRMATORY = ("gmsc", "homecredit2018")


def ms(v):
    a = np.asarray(v, dtype=float)
    if not np.isfinite(a).any():
        return {"mean": None, "std": None}
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


class Reference:
    """The model, the conformal threshold and the kNN index, fitted on the reference population."""

    def __init__(self, X_tr, y_tr, X_cal, y_cal, seed):
        med = np.nanmedian(X_tr, axis=0)
        self.fill = np.where(np.isfinite(med), med, 0.0)
        X_tr, X_cal = self.impute(X_tr), self.impute(X_cal)
        self.model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X_tr, y_tr)
        self.qhat = ConformalCalibrator(self.model.predict_proba(X_cal), y_cal).qhat(ALPHA)
        self.ood = KNNOODDetector(k=K).fit(X_cal)

    def impute(self, X):
        X = np.array(X, dtype=float)
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.fill, np.nonzero(bad)[1])
        return X

    def run(self, X_raw, y, seed, corrupted=None):
        X = self.impute(X_raw)
        probs = self.model.predict_proba(X)
        msp = probs.max(axis=1)
        sets = (1.0 - probs <= self.qhat)
        d, _ = self.ood.nn.kneighbors(self.ood.scaler.transform(X), n_neighbors=K)
        dist = d.mean(axis=1)                            # what KNNOODDetector.distance returns, row by row
        pct = self.ood.percentiles_batch(X)
        flagged = pct > FLAG_PCT
        mart = ConformalMartingale(self.ood.calib_dists)
        states, first_alarm = [], None
        for i, a in enumerate(dist):
            st = mart.update(float(a))["state"]
            states.append(st)
            if st == "ALARM" and first_alarm is None:
                first_alarm = i + 1
        states = np.array(states)
        # higher = more reliable. "gated" and "regime_switch" are the v2 benchmark's code
        # (run_shift_bench_v2.py), whose values Amendment 1 quotes: flagged rows first, by
        # distance percentile, ties at the top percentile broken by confidence.
        # "gated_distance" (flagged rows by raw distance alone) is a disclosed variant, not the bar.
        sig = {"msp": msp, "ood_knn": -pct,
               "gated": np.where(flagged, msp - 1.0 - pct / 100.0, msp),
               "regime_switch": np.where(states == "OK", msp, -pct / 100.0 - 1.0),
               "gated_distance": np.where(flagged, -dist, msp)}
        pred = (probs[:, 1] >= 0.5).astype(int)
        bad = ((pred == 0) & (y == 1)).astype(int)
        sig["random"] = np.random.default_rng(10_000 + seed).random(len(y))
        sig["oracle"] = -bad.astype(float)
        k = max(1, int(round(REFERRAL * len(y))))
        row = {"auc": float(roc_auc_score(y, probs[:, 1])), "default_rate": float(y.mean()),
               "bad_approvals": int(bad.sum()), "bad_approval_rate": float(bad.mean()),
               "coverage": float(sets[np.arange(len(y)), y].mean()),
               "review_rate": float((sets.sum(axis=1) != 1).mean()),
               "ood_flag_rate": float(flagged.mean()),
               "alarm": bool(mart.max_log10 >= np.log10(ALARM_THRESHOLD)),
               "watch_or_alarm": bool(mart.max_log10 >= np.log10(WATCH_THRESHOLD)),
               "first_alarm_at": first_alarm, "max_log10_martingale": round(float(mart.max_log10), 3),
               "signals": {}}
        base = M.default_capture(sig["msp"], bad, REFERRAL)
        rand = M.default_capture(sig["random"], bad, REFERRAL)
        for name, s in sig.items():
            cap = M.default_capture(s, bad, REFERRAL)
            e = {"bad_capture": float(cap),
                 "vs_confidence": float(cap / base) if base > 0 else float("nan"),
                 "vs_random": float(cap / rand) if rand > 0 else float("nan"),
                 "vs_expected_random": float(cap / REFERRAL)}
            if corrupted is not None:
                e["corrupted_caught"] = float(corrupted[np.argsort(s, kind="stable")[:k]].sum() / max(corrupted.sum(), 1))
            row["signals"][name] = e
        if seed == 0:                                    # paired bootstrap of the ratio, as run_shift_bench.py
            bs = np.random.default_rng(12345)
            row["ratio_vs_confidence_ci95"] = {n: boot_ratio(bs, sig[n], sig["msp"], bad) for n in ("gated", "regime_switch", "gated_distance", "ood_knn")}
        return row


def boot_ratio(rng, sig, base, bad):
    n = len(bad)
    out = []
    for _ in range(N_BOOT):
        b = rng.integers(0, n, n)
        dn = M.default_capture(base[b], bad[b], REFERRAL)
        if dn > 0:
            out.append(M.default_capture(sig[b], bad[b], REFERRAL) / dn)
    a = np.asarray(out)
    return [round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4)] if len(a) else [None, None]


def aggregate(rows):
    out = {k: ms([r[k] if r[k] is not None else np.nan for r in rows]) for k in rows[0]
           if k not in ("signals", "alarm", "watch_or_alarm", "ratio_vs_confidence_ci95")}
    out["alarm_seeds"] = int(sum(r["alarm"] for r in rows))
    out["watch_or_alarm_seeds"] = int(sum(r["watch_or_alarm"] for r in rows))
    out["signals"] = {n: {k: ms([r["signals"][n][k] for r in rows]) for k in rows[0]["signals"][n]} for n in rows[0]["signals"]}
    out["ratio_vs_confidence_ci95_seed0"] = rows[0].get("ratio_vs_confidence_ci95")
    return out


def run_dataset(ds):
    t0 = time.time()
    X, y = ds["X"], ds["y"]
    ref_pool, shift_pool = np.flatnonzero(ds["reference"]), np.flatnonzero(ds["shifted"])
    need = N_TRAIN + N_CAL + N_TEST
    if len(ref_pool) < need or len(shift_pool) < N_TEST:
        raise SystemExit(f"{ds['name']}: reference {len(ref_pool)} (need {need}), shifted {len(shift_pool)} (need {N_TEST})")
    per = {"mix": {str(l): [] for l in LAMBDAS}, "corruption": {str(r): [] for r in RHOS}, "iid_auc": []}
    for seed in SEEDS:
        pool = np.random.default_rng(seed).permutation(len(y))[:need]      # the i.i.d. control of run_shift_bench.py
        c_tr, c_cal, c_te = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_CAL], pool[N_TRAIN + N_CAL:]
        C = Reference(X[c_tr], y[c_tr], X[c_cal], y[c_cal], seed)
        per["iid_auc"].append(float(roc_auc_score(y[c_te], C.model.predict_proba(C.impute(X[c_te]))[:, 1])))
        print(f"  {ds['name']} seed {seed} iid control: auc {per['iid_auc'][-1]:.3f}", flush=True)
        rng = np.random.default_rng(seed)
        ref = rng.permutation(ref_pool)
        i_tr, i_cal, rest = ref[:N_TRAIN], ref[N_TRAIN:N_TRAIN + N_CAL], ref[N_TRAIN + N_CAL:]
        sh = rng.permutation(shift_pool)
        R = Reference(X[i_tr], y[i_tr], X[i_cal], y[i_cal], seed)
        for lam in LAMBDAS:
            n_sh = int(round(lam * N_TEST))
            i_te = rng.permutation(np.concatenate([rest[:N_TEST - n_sh], sh[:n_sh]]))
            row = R.run(X[i_te], y[i_te], seed)
            per["mix"][str(lam)].append(row)
            s = row["signals"]
            print(f"  {ds['name']} seed {seed} lambda {lam:.2f}: auc {row['auc']:.3f} cov {row['coverage']:.3f} review {row['review_rate']:.1%} "
                  f"alarm {row['alarm']} | vs random: msp {s['msp']['vs_random']:.2f} gated {s['gated']['vs_random']:.2f} "
                  f"switch {s['regime_switch']['vs_random']:.2f} | gated vs conf {s['gated']['vs_confidence']:.2f}", flush=True)
        i_te = rng.permutation(rest[:N_TEST])
        for rho in RHOS:
            crng = np.random.default_rng(500 + seed)
            rows = np.sort(crng.choice(N_TEST, int(round(rho * N_TEST)), replace=False))
            mask = np.zeros(N_TEST, bool)
            mask[rows] = True
            row = R.run(ds["corrupt"](X[i_te], rows), y[i_te], seed, mask)
            per["corruption"][str(rho)].append(row)
            print(f"  {ds['name']} seed {seed} corruption {rho:.2f}: caught gated {row['signals']['gated']['corrupted_caught']:.2f} "
                  f"msp {row['signals']['msp']['corrupted_caught']:.2f} alarm {row['alarm']}", flush=True)
    res = {"meta": {"dataset": ds["name"], "title": ds["title"], "status": ds["status"], "selftest": SELFTEST,
                    "n_rows": int(len(y)), "n_features": int(X.shape[1]), "feature_names": ds["feature_names"],
                    "n_reference": int(len(ref_pool)), "n_shifted": int(len(shift_pool)),
                    "default_rate_reference": round(float(y[ref_pool].mean()), 4),
                    "default_rate_shifted": round(float(y[shift_pool].mean()), 4),
                    "missing_share": round(float(np.mean(~np.isfinite(X))), 4),
                    "monetary": ds["monetary"], "seeds": list(SEEDS), "n_train": N_TRAIN, "n_cal": N_CAL,
                    "n_test": N_TEST, "referral": REFERRAL, "alpha": ALPHA, "k": K, "flag_percentile": FLAG_PCT,
                    "n_boot": N_BOOT},
           "iid_control_auc": ms(per["iid_auc"]),
           "mix": {k: aggregate(v) for k, v in per["mix"].items()},
           "corruption": {k: aggregate(v) for k, v in per["corruption"].items()},
           "per_seed": per}
    if "n_with_bureau" in ds:
        res["meta"]["n_with_bureau"] = ds["n_with_bureau"]
    res["verdict"] = verdict(res)
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    return res


def verdict(res):
    calm = res["mix"]["0.0"]
    iid = res["iid_control_auc"]["mean"]
    v = {"iid_control_auc": iid,
         "calm": {"coverage": calm["coverage"]["mean"], "review_rate": calm["review_rate"]["mean"],
                  "auc": calm["auc"]["mean"]},
         "moderate": {}, "severe": {}, "corruption": {}}
    ok = True
    for lam in MODERATE:
        m = res["mix"][str(lam)]
        g = m["signals"]["gated"]
        loss = iid - m["auc"]["mean"]
        ci = (m["ratio_vs_confidence_ci95_seed0"] or {}).get("gated", [None, None])
        e = {"auc_loss": round(loss, 4), "auc_loss_vs_calm": round(calm["auc"]["mean"] - m["auc"]["mean"], 4),
             "mild": bool(loss < BAR_AUC_LOSS),
             "gated_vs_random": g["vs_random"]["mean"], "gated_vs_confidence": g["vs_confidence"]["mean"],
             "gated_vs_confidence_ci95_seed0": ci,
             "regime_switch_vs_random": m["signals"]["regime_switch"]["vs_random"]["mean"],
             "regime_switch_vs_confidence": m["signals"]["regime_switch"]["vs_confidence"]["mean"],
             "regime_switch_vs_confidence_ci95_seed0": (m["ratio_vs_confidence_ci95_seed0"] or {}).get("regime_switch")}
        e["meets_random_bar"] = bool(e["gated_vs_random"] >= BAR_RANDOM)
        e["beats_confidence"] = bool(ci[0] is not None and ci[0] > 1.0)
        e["pass"] = bool(e["meets_random_bar"] and e["beats_confidence"] and not e["mild"])
        ok &= e["pass"]
        v["moderate"][str(lam)] = e
    v["moderate_pass"] = bool(ok)
    for lam in SEVERE:
        m = res["mix"][str(lam)]
        v["severe"][str(lam)] = {"alarm_seeds": m["alarm_seeds"], "of": len(res["per_seed"]["mix"][str(lam)]),
                                 "first_alarm_at": m["first_alarm_at"]["mean"],
                                 "review_rate": m["review_rate"]["mean"], "review_rate_calm": calm["review_rate"]["mean"],
                                 "review_rises": bool(m["review_rate"]["mean"] > calm["review_rate"]["mean"]),
                                 "coverage": m["coverage"]["mean"], "auc": m["auc"]["mean"]}
        v["severe"][str(lam)]["holds"] = bool(v["severe"][str(lam)]["alarm_seeds"] == v["severe"][str(lam)]["of"]
                                              and v["severe"][str(lam)]["review_rises"])
    for rho in RHOS:
        c = res["corruption"][str(rho)]
        caught = c["signals"]["gated"]["corrupted_caught"]["mean"]
        n = len(res["per_seed"]["corruption"][str(rho)])
        v["corruption"][str(rho)] = {"gated_caught": caught, "confidence_caught": c["signals"]["msp"]["corrupted_caught"]["mean"],
                                     "alarm_seeds": c["alarm_seeds"], "of": n,
                                     "holds": bool(caught >= BAR_CORRUPT and c["alarm_seeds"] == n)}
    return v


def selftest_sets():
    """Synthetic frames with the real column names, to run the code path without opening the sealed sets."""
    rng = np.random.default_rng(0)
    n = 30_000
    late = rng.random(n) < 0.2
    g = pd.DataFrame({"Unnamed: 0": np.arange(n), CD.GMSC_LABEL: (rng.random(n) < np.where(late, .2, .05)).astype(int),
                      "RevolvingUtilizationOfUnsecuredLines": rng.random(n), "age": rng.integers(21, 90, n),
                      CD.GMSC_SHIFT: np.where(late, rng.integers(1, 5, n), 0), "DebtRatio": rng.random(n),
                      "MonthlyIncome": np.where(rng.random(n) < .2, np.nan, rng.lognormal(8.5, .6, n)),
                      "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n), "NumberOfTimes90DaysLate": rng.integers(0, 2, n),
                      "NumberRealEstateLoansOrLineofCredit": rng.integers(0, 4, n),
                      "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 2, n), "NumberOfDependents": rng.integers(0, 4, n)})
    ids = np.arange(100_000, 100_000 + n)
    app = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": (rng.random(n) < .08).astype(int), "CODE_GENDER": rng.choice(["M", "F"], n),
                        "DAYS_BIRTH": -rng.integers(7000, 25000, n), "NAME_FAMILY_STATUS": rng.choice(["Married", "Single"], n),
                        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
                        **{c: rng.lognormal(11, .5, n) for c in CD.HC_MONEY}, "EXT_SOURCE_1": np.where(rng.random(n) < .5, np.nan, rng.random(n)),
                        "EXT_SOURCE_2": rng.random(n), "DAYS_EMPLOYED": -rng.integers(0, 9000, n)})
    bureau = pd.DataFrame({"SK_ID_CURR": rng.choice(ids, 3 * n), "CREDIT_DAY_OVERDUE": np.where(rng.random(3 * n) < .05, 10, 0),
                           "AMT_CREDIT_MAX_OVERDUE": np.nan, "AMT_CREDIT_SUM_OVERDUE": 0.0})
    return {"gmsc": lambda: CD.gmsc_frame(g), "homecredit2018": lambda: CD.homecredit_frame(app, bureau), "taiwan": CD.load_taiwan}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaders = selftest_sets() if SELFTEST else CD.LOADERS
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(loaders)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in names:
        print(f"[{name}] loading", flush=True)
        res = run_dataset(loaders[name]())
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        summary[name] = {"status": res["meta"]["status"], **res["verdict"]}
        print(f"[{name}] moderate pass: {res['verdict']['moderate_pass']}  ({res['meta']['seconds']} s)", flush=True)
    if all(c in summary for c in CONFIRMATORY):
        summary["bar"] = {"pass": bool(all(summary[c]["moderate_pass"] for c in CONFIRMATORY)),
                          "rule": "gated passes the moderate tier on both lambdas on both confirmatory datasets"}
    summary_path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {summary_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
