"""Production-model track (Amendment 3, EXPLORATORY): the tiers of the
confirmatory runs, on a model built the way a credit team would build it.

Written after the confirmatory results of Amendment 1 were seen; it does not
change them and makes no confirmatory claim. Protocol, fixed in Amendment 3
before Give Me Some Credit or Home Credit is run through it:

  data         all of the declared reference population: 4,000 rows held out for
               testing, N_CAL for calibration (10,000; 5,000 on Taiwan), the rest
               for training. Populations, tiers, corruption, seeds 0-4, the
               signals and every Reliax parameter are those of run_confirmatory.py.
  inputs       eval/production_data.py (Home Credit adds categorical columns,
               ratio features and bureau aggregates); the kNN distance reads the
               non-categorical inputs, missing values filled with training medians.
  model        HistGradientBoostingClassifier, learning rate 0.05, up to 2,000
               iterations with early stopping (10% of the training split, 50
               rounds), categorical columns as categories. max_leaf_nodes in
               {15, 31, 63} x min_samples_leaf in {20, 200}, chosen once on seed 0
               by AUC on a 20% hold-out of the training split, separately for the
               i.i.d. control and the reference model, then refitted per seed.
  operating    besides 0.5 and "the riskiest 10%", the cut-off that maximises F1
  points       on the calibration split, applied unchanged to every test set.
  reported     model: AUC, PR-AUC, accuracy, precision, recall, F1 at each point;
               Reliax: coverage, REVIEW rate, alarm, gated and regime-switch
               referral against random and confidence (seed-0 paired bootstrap),
               corrupted rows caught. No bar: nothing here passes or fails.

Usage
  .venv/bin/python eval/run_production.py [gmsc homecredit2018 taiwan] [--selftest]
Writes results/production/<dataset>.json and results/production/summary.json
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

import run_confirmatory as RC                                      # signals, bootstrap, aggregation, constants
import production_data as PD
from reliax_core.conformal import ConformalCalibrator
from reliax_core.martingale import ALARM_THRESHOLD, ConformalMartingale, WATCH_THRESHOLD
from reliax_core.ood import KNNOODDetector
import methods as M

SELFTEST = RC.SELFTEST
OUT = RC.BASE / "results" / ("production_selftest" if SELFTEST else "production")
N_TEST = RC.N_TEST
N_CAL = {"taiwan": 5000}
GRID = [{"max_leaf_nodes": l, "min_samples_leaf": m} for l in (15, 31, 63) for m in (20, 200)]
BASE_PARAMS = dict(learning_rate=0.05, max_iter=2000, early_stopping=True, validation_fraction=0.1,
                   n_iter_no_change=50, scoring="loss")


class Model:
    """The gradient-boosted model; columns with no value at all in the training split are left out."""

    def __init__(self, X, y, cat, params, seed):
        self.keep = np.isfinite(X).any(axis=0)
        c = cat[self.keep]
        self.m = HistGradientBoostingClassifier(**BASE_PARAMS, **params, categorical_features=c if c.any() else None,
                                                random_state=seed).fit(X[:, self.keep], y)
        self.n_iter_ = self.m.n_iter_

    def predict_proba(self, X):
        return self.m.predict_proba(X[:, self.keep])


def fit(X, y, cat, params, seed):
    return Model(X, y, cat, params, seed)


def tune(X, y, cat):
    """Seed 0, 20% hold-out of the training split, AUC."""
    idx = np.random.default_rng(0).permutation(len(y))
    cut = int(0.8 * len(y))
    a, b = idx[:cut], idx[cut:]
    scores = []
    for p in GRID:
        m = fit(X[a], y[a], cat, p, 0)
        scores.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
    best = int(np.argmax(scores))
    return GRID[best], [{"params": g, "auc": round(float(s), 4)} for g, s in zip(GRID, scores)]


def model_metrics(y, p, cut_f1):
    out = {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "default_rate": float(y.mean())}
    top = np.argsort(-p, kind="stable")[:int(round(0.10 * len(y)))]
    flagged = {"at_0.5": p >= 0.5, "at_f1_cut": p >= cut_f1, "riskiest_10pct": np.isin(np.arange(len(y)), top)}
    for k, f in flagged.items():
        tp = float((f & (y == 1)).sum())
        prec = tp / f.sum() if f.any() else float("nan")
        rec = tp / max((y == 1).sum(), 1)
        out[k] = {"accuracy": float(((f.astype(int)) == y).mean()), "precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if f.any() and prec + rec > 0 else 0.0, "flagged_share": float(f.mean())}
    return out


def f1_cut(y, p):
    best, cut = -1.0, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.995, 200))):
        f = p >= t
        tp = (f & (y == 1)).sum()
        if f.any() and tp:
            pr, rc = tp / f.sum(), tp / (y == 1).sum()
            f1 = 2 * pr * rc / (pr + rc)
            if f1 > best:
                best, cut = f1, float(t)
    return cut


class Production:
    """Model, conformal threshold, kNN index and operating cut-off, fitted on one training split."""

    def __init__(self, X_tr, y_tr, X_cal, y_cal, cat, params, seed):
        self.cat = cat
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)              # an all-missing column has no median: filled with 0
            med = np.nanmedian(X_tr[:, ~cat], axis=0) if (~cat).any() else np.zeros(0)
        self.fill = np.where(np.isfinite(med), med, 0.0)
        self.model = fit(X_tr, y_tr, cat, params, seed)
        self.n_iter = int(self.model.n_iter_)
        p_cal = self.model.predict_proba(X_cal)
        self.qhat = ConformalCalibrator(p_cal, y_cal).qhat(RC.ALPHA)
        self.cut_f1 = f1_cut(y_cal, p_cal[:, 1])
        self.ood = KNNOODDetector(k=RC.K).fit(self.ood_inputs(X_cal))

    def ood_inputs(self, X):
        Z = np.array(X[:, ~self.cat], dtype=float)
        bad = ~np.isfinite(Z)
        Z[bad] = np.take(self.fill, np.nonzero(bad)[1])
        return Z

    def run(self, X, y, seed, corrupted=None):
        probs = self.model.predict_proba(X)
        msp = probs.max(axis=1)
        sets = (1.0 - probs <= self.qhat)
        Z = self.ood_inputs(X)
        d, _ = self.ood.nn.kneighbors(self.ood.scaler.transform(Z), n_neighbors=RC.K)
        dist = d.mean(axis=1)
        pct = self.ood.percentiles_batch(Z)
        flagged = pct > RC.FLAG_PCT
        mart = ConformalMartingale(self.ood.calib_dists)
        states, first_alarm = [], None
        for i, a in enumerate(dist):
            st = mart.update(float(a))["state"]
            states.append(st)
            if st == "ALARM" and first_alarm is None:
                first_alarm = i + 1
        states = np.array(states)
        sig = {"msp": msp, "ood_knn": -pct,                          # as run_confirmatory.py
               "gated": np.where(flagged, msp - 1.0 - pct / 100.0, msp),
               "regime_switch": np.where(states == "OK", msp, -pct / 100.0 - 1.0),
               "gated_distance": np.where(flagged, -dist, msp)}
        pred = (probs[:, 1] >= 0.5).astype(int)
        bad = ((pred == 0) & (y == 1)).astype(int)
        sig["random"] = np.random.default_rng(10_000 + seed).random(len(y))
        sig["oracle"] = -bad.astype(float)
        k = max(1, int(round(RC.REFERRAL * len(y))))
        row = {"auc": float(roc_auc_score(y, probs[:, 1])), "default_rate": float(y.mean()),
               "bad_approvals": int(bad.sum()), "bad_approval_rate": float(bad.mean()),
               "coverage": float(sets[np.arange(len(y)), y].mean()), "review_rate": float((sets.sum(axis=1) != 1).mean()),
               "ood_flag_rate": float(flagged.mean()),
               "alarm": bool(mart.max_log10 >= np.log10(ALARM_THRESHOLD)),
               "watch_or_alarm": bool(mart.max_log10 >= np.log10(WATCH_THRESHOLD)),
               "first_alarm_at": first_alarm, "max_log10_martingale": round(float(mart.max_log10), 3),
               "model": model_metrics(y, probs[:, 1], self.cut_f1), "signals": {}}
        base = M.default_capture(sig["msp"], bad, RC.REFERRAL)
        rand = M.default_capture(sig["random"], bad, RC.REFERRAL)
        for name, s in sig.items():
            cap = M.default_capture(s, bad, RC.REFERRAL)
            e = {"bad_capture": float(cap), "vs_confidence": float(cap / base) if base > 0 else float("nan"),
                 "vs_random": float(cap / rand) if rand > 0 else float("nan")}
            if corrupted is not None:
                e["corrupted_caught"] = float(corrupted[np.argsort(s, kind="stable")[:k]].sum() / max(corrupted.sum(), 1))
            row["signals"][name] = e
        if seed == 0:
            bs = np.random.default_rng(12345)
            row["ratio_vs_confidence_ci95"] = {n: RC.boot_ratio(bs, sig[n], sig["msp"], bad) for n in ("gated", "regime_switch", "gated_distance", "ood_knn")}
        return row


def flat_model(rows):
    keys = ["auc", "pr_auc", "default_rate"]
    out = {k: RC.ms([r["model"][k] for r in rows]) for k in keys}
    for op in ("at_0.5", "at_f1_cut", "riskiest_10pct"):
        out[op] = {k: RC.ms([r["model"][op][k] for r in rows]) for k in rows[0]["model"][op]}
    return out


def aggregate(rows):
    out = RC.aggregate([{k: v for k, v in r.items() if k != "model"} for r in rows])
    out["model"] = flat_model(rows)
    return out


def run_dataset(name, ds):
    t0 = time.time()
    X, y, cat = ds["X"], ds["y"], ds["categorical"]
    n_cal = N_CAL.get(name, 10_000)
    ref_pool, shift_pool = np.flatnonzero(ds["reference"]), np.flatnonzero(ds["shifted"])
    # i.i.d. control and reference model: tuned once, on seed 0
    pool = np.random.default_rng(0).permutation(len(y))
    iid_params, iid_grid = tune(X[pool[n_cal + N_TEST:]], y[pool[n_cal + N_TEST:]], cat)
    ref0 = np.random.default_rng(0).permutation(ref_pool)
    ref_params, ref_grid = tune(X[ref0[n_cal + N_TEST:]], y[ref0[n_cal + N_TEST:]], cat)
    print(f"  {name} tuned: iid {iid_params}, reference {ref_params}", flush=True)
    per = {"iid": [], "mix": {str(l): [] for l in RC.LAMBDAS}, "corruption": {str(r): [] for r in RC.RHOS}, "n_iter": []}
    for seed in RC.SEEDS:
        pool = np.random.default_rng(seed).permutation(len(y))
        cal, te, tr = pool[:n_cal], pool[n_cal:n_cal + N_TEST], pool[n_cal + N_TEST:]
        C = Production(X[tr], y[tr], X[cal], y[cal], cat, iid_params, seed)
        per["iid"].append(C.run(X[te], y[te], seed))
        rng = np.random.default_rng(seed)
        ref = rng.permutation(ref_pool)
        cal, rest, tr = ref[:n_cal], ref[n_cal:n_cal + N_TEST], ref[n_cal + N_TEST:]
        sh = rng.permutation(shift_pool)
        R = Production(X[tr], y[tr], X[cal], y[cal], cat, ref_params, seed)
        per["n_iter"].append({"iid": C.n_iter, "reference": R.n_iter})
        m = per["iid"][-1]["model"]
        print(f"  {name} seed {seed} iid: auc {m['auc']:.3f} pr-auc {m['pr_auc']:.3f} | f1 cut: precision {m['at_f1_cut']['precision']:.3f} "
              f"recall {m['at_f1_cut']['recall']:.3f} | trees {C.n_iter}/{R.n_iter}", flush=True)
        for lam in RC.LAMBDAS:
            n_sh = int(round(lam * N_TEST))
            i_te = rng.permutation(np.concatenate([rest[:N_TEST - n_sh], sh[:n_sh]]))
            row = R.run(X[i_te], y[i_te], seed)
            per["mix"][str(lam)].append(row)
            s = row["signals"]
            print(f"  {name} seed {seed} lambda {lam:.2f}: auc {row['auc']:.3f} cov {row['coverage']:.3f} review {row['review_rate']:.1%} "
                  f"alarm {row['alarm']} | gated vs random {s['gated']['vs_random']:.2f} vs conf {s['gated']['vs_confidence']:.2f}", flush=True)
        i_te = rng.permutation(rest[:N_TEST])
        for rho in RC.RHOS:
            rows = np.sort(np.random.default_rng(500 + seed).choice(N_TEST, int(round(rho * N_TEST)), replace=False))
            mask = np.zeros(N_TEST, bool)
            mask[rows] = True
            row = R.run(ds["corrupt"](X[i_te], rows), y[i_te], seed, mask)
            per["corruption"][str(rho)].append(row)
    res = {"meta": {"dataset": name, "title": ds["title"], "track": "production model, exploratory (Amendment 3)",
                    "selftest": SELFTEST, "n_rows": int(len(y)), "n_features": int(X.shape[1]),
                    "n_categorical": int(cat.sum()), "n_distance_inputs": int((~cat).sum()), "feature_names": ds["feature_names"],
                    "n_reference": int(len(ref_pool)), "n_shifted": int(len(shift_pool)),
                    "n_train_reference": int(len(ref_pool) - n_cal - N_TEST), "n_train_iid": int(len(y) - n_cal - N_TEST),
                    "n_cal": n_cal, "n_test": N_TEST, "seeds": list(RC.SEEDS),
                    "params": {"iid": iid_params, "reference": ref_params, **BASE_PARAMS},
                    "tuning": {"iid": iid_grid, "reference": ref_grid}, "trees": per["n_iter"]},
           "iid": aggregate(per["iid"]),
           "mix": {k: aggregate(v) for k, v in per["mix"].items()},
           "corruption": {k: aggregate(v) for k, v in per["corruption"].items()},
           "per_seed": {k: v for k, v in per.items() if k != "n_iter"}}
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    return res


def headline(res):
    iid, calm = res["iid"], res["mix"]["0.0"]
    h = {"model_iid": {"auc": iid["model"]["auc"]["mean"], "pr_auc": iid["model"]["pr_auc"]["mean"],
                       "default_rate": iid["model"]["default_rate"]["mean"],
                       **{op: {k: iid["model"][op][k]["mean"] for k in ("accuracy", "precision", "recall", "f1", "flagged_share")}
                          for op in ("at_0.5", "at_f1_cut", "riskiest_10pct")}},
         "calm": {"auc": calm["auc"]["mean"], "coverage": calm["coverage"]["mean"], "review_rate": calm["review_rate"]["mean"]},
         "iid_coverage": iid["coverage"]["mean"], "iid_review_rate": iid["review_rate"]["mean"], "moderate": {}, "severe": {}, "corruption": {}}
    for lam in RC.MODERATE:
        m = res["mix"][str(lam)]
        h["moderate"][str(lam)] = {"auc": m["auc"]["mean"], "coverage": m["coverage"]["mean"],
                                   "gated_vs_random": m["signals"]["gated"]["vs_random"]["mean"],
                                   "gated_vs_confidence": m["signals"]["gated"]["vs_confidence"]["mean"],
                                   "gated_vs_confidence_ci95_seed0": (m["ratio_vs_confidence_ci95_seed0"] or {}).get("gated")}
    for lam in RC.SEVERE:
        m = res["mix"][str(lam)]
        h["severe"][str(lam)] = {"alarm_seeds": m["alarm_seeds"], "first_alarm_at": m["first_alarm_at"]["mean"],
                                 "coverage": m["coverage"]["mean"], "review_rate": m["review_rate"]["mean"], "auc": m["auc"]["mean"]}
    for rho in RC.RHOS:
        c = res["corruption"][str(rho)]
        h["corruption"][str(rho)] = {"gated_caught": c["signals"]["gated"]["corrupted_caught"]["mean"],
                                     "confidence_caught": c["signals"]["msp"]["corrupted_caught"]["mean"], "alarm_seeds": c["alarm_seeds"]}
    return h


def selftest_loaders():
    """Synthetic frames with the real column names; runs the code path without the Kaggle files."""
    st = RC.selftest_sets()
    rng = np.random.default_rng(1)
    n = 30_000
    ids = np.arange(100_000, 100_000 + n)
    app = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": (rng.random(n) < .08).astype(int), "CODE_GENDER": rng.choice(["M", "F"], n),
                        "DAYS_BIRTH": -rng.integers(7000, 25000, n), "NAME_FAMILY_STATUS": rng.choice(["Married", "Single"], n),
                        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
                        **{c: rng.lognormal(11, .5, n) for c in RC.CD.HC_MONEY},
                        **{f"EXT_SOURCE_{i}": np.where(rng.random(n) < .3, np.nan, rng.random(n)) for i in (1, 2, 3)},
                        "DAYS_EMPLOYED": np.where(rng.random(n) < .1, 365243, -rng.integers(0, 9000, n))})
    m = 3 * n
    bureau = pd.DataFrame({"SK_ID_CURR": rng.choice(ids, m), "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], m),
                           "DAYS_CREDIT": -rng.integers(0, 3000, m), "CREDIT_DAY_OVERDUE": np.where(rng.random(m) < .05, 10, 0),
                           "AMT_CREDIT_MAX_OVERDUE": np.nan, "AMT_CREDIT_SUM": rng.lognormal(11, 1, m),
                           "AMT_CREDIT_SUM_DEBT": rng.lognormal(10, 1, m), "AMT_CREDIT_SUM_OVERDUE": 0.0,
                           "CNT_CREDIT_PROLONG": rng.integers(0, 2, m)})
    return {"gmsc": lambda: PD._with_masks(st["gmsc"]()), "homecredit2018": lambda: PD.homecredit_frame(app, bureau),
            "taiwan": PD.load_taiwan}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaders = selftest_loaders() if SELFTEST else PD.LOADERS
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(loaders)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        print(f"[{name}] loading", flush=True)
        res = run_dataset(name, loaders[name]())
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        summary[name] = headline(res)
        print(f"[{name}] done in {res['meta']['seconds']} s", flush=True)
    path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {path.relative_to(RC.BASE)}")


if __name__ == "__main__":
    main()
