"""Shift benchmark v2 (EXPLORATORY, Taiwan only): three regimes a lender can
actually meet, and the question "which signal should route the 10% least
reliable decisions to REVIEW?" asked in each.

Regimes
  mix      the live population is a mixture of the reference population and a
           genuinely different one (delayed payers, the PAY_0 >= 1 subgroup),
           at shares lambda in {0, .25, .5, .75, 1}. lambda = 0 is the i.i.d.
           control, lambda = 1 the severe split of the v1 benchmark. This is
           what a book looks like while a shift is happening, not after.
  noise    the population is unchanged but a share rho of the incoming rows is
           corrupted upstream: a unit error (monetary fields x100: a currency
           or cents bug), missing fields (half the features zeroed by a failed
           join), or random perturbation (one standard deviation of noise on
           every feature). Labels are real; the model's inputs are wrong.
  severe   lambda = 1, reported for the tripwire, not for the ranking.

Signals (higher = more reliable): the v1 set, plus the candidates that the
theory says should work under shift: OOD distance gating confidence
("gated"), a rank fusion of confidence and OOD ("min_rank"), and the
regime-switching rule proposed on deck slide 16: confidence while the
exchangeability martingale is calm, OOD distance once it reaches WATCH or
ALARM ("regime_switch", evaluated sequentially over the stream).

Metrics: bad-approval capture at a 10% referral rate against random referral
and against plain confidence; error capture; the share of corrupted rows the
referral catches; coverage and the certified REVIEW rate.

Nothing here is pre-registered. It informs the amendment in
PREREGISTRATION_AMENDMENT.md; the confirmatory datasets stay untouched.

Usage: .venv/bin/python eval/run_shift_bench_v2.py [--fast]
Writes results/shift_bench_v2.json
"""
import json
import pathlib
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from reliax_core.auditor import ErrorAuditor                      # noqa: E402
from reliax_core.conformal import ConformalCalibrator             # noqa: E402
from reliax_core.martingale import ConformalMartingale            # noqa: E402
from reliax_core.ood import KNNOODDetector                        # noqa: E402
from reliax_core.scoring import reliability_score                 # noqa: E402
from reliax_core.sl_fusion import fuse_signals                    # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator            # noqa: E402

import data as D                                                   # noqa: E402
import methods as M                                                # noqa: E402

RESULTS = BASE / "results"
FAST = "--fast" in sys.argv
ALPHA = 0.05
REFERRAL = 0.10
SEEDS = range(2 if FAST else 3)
N_TRAIN, N_CAL, N_TEST = 8000, 4000, (2000 if FAST else 4000)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
NOISE = {"unit_error_0.10": ("unit", 0.10), "missing_0.10": ("missing", 0.10), "gaussian_0.10": ("gauss", 0.10),
         "unit_error_0.05": ("unit", 0.05), "unit_error_0.20": ("unit", 0.20)}
OOD_FLAG_PCT = 95.0


def ms(v):
    a = np.asarray(v, dtype=float)
    return {"mean": round(float(np.nanmean(a)), 4), "std": round(float(np.nanstd(a)), 4)}


class Reference:
    """Everything fitted on the reference population, once per seed."""

    def __init__(self, X_tr, y_tr, X_cal, y_cal, seed):
        self.seed = seed
        self.model = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(X_tr, y_tr)
        probs_cal = self.model.predict_proba(X_cal)
        self.conf = ConformalCalibrator(probs_cal, y_cal)
        self.qhat = self.conf.qhat(ALPHA)
        self.va = VennAbersCalibrator(probs_cal[:, 1], y_cal)
        self.ood = KNNOODDetector(k=10).fit(X_cal)
        self.auditor = ErrorAuditor(seed=seed).fit(X_cal, probs_cal, y_cal, self.model.predict(X_cal))
        rs = np.random.default_rng(seed)
        self.members = []
        for m in range(5):
            sub = rs.choice(len(y_tr), int(0.8 * len(y_tr)), replace=False)
            self.members.append(HistGradientBoostingClassifier(max_iter=300, random_state=1000 + m).fit(X_tr[sub], y_tr[sub]))
        self.X_cal, self.y_cal = X_cal, y_cal

    def signals(self, X_te):
        probs = self.model.predict_proba(X_te)
        msp = probs.max(axis=1)
        margin = 2 * msp - 1
        set_sizes = (1.0 - probs <= self.qhat).sum(axis=1)
        pvals = np.array([self.conf.p_value(float(p)) for p in msp])
        va_width = np.array([self.va.interval(float(s))["width"] for s in probs[:, 1]])
        ood_pct = self.ood.percentiles_batch(X_te)
        p_err = self.auditor.p_error_batch(X_te, probs)
        n = len(msp)
        composite = np.array([reliability_score(float(margin[i]), float(pvals[i]), int(set_sizes[i]), float(ood_pct[i]),
                                                bool(ood_pct[i] > 99.0), float(va_width[i]), float(p_err[i]),
                                                bool(p_err[i] > self.auditor.flag_threshold), False) for i in range(n)])
        # regime switch: sequential martingale over the stream on OOD distances
        mart = ConformalMartingale(self.ood.calib_dists)
        states, sl_states = [], []
        for i in range(n):
            st = mart.update(self.ood.distance(X_te[i]))["state"]
            states.append(st)
        states = np.array(states)
        sl = np.array([fuse_signals(float(msp[i]), float(pvals[i]), int(set_sizes[i]), float(va_width[i]), float(p_err[i]),
                                    float(ood_pct[i]), str(states[i]), 1.0 - self.auditor.base_error_rate)["expected"]
                       for i in range(n)])
        ens = np.vstack([m.predict_proba(X_te)[:, 1] for m in self.members])
        ens_mean = ens.mean(axis=0)
        flagged = ood_pct > OOD_FLAG_PCT
        gated = np.where(flagged, msp - 1.0 - ood_pct / 100.0, msp)
        r_msp = np.argsort(np.argsort(msp)) / n
        r_ood = np.argsort(np.argsort(-ood_pct)) / n
        min_rank = np.minimum(r_msp, r_ood)
        calm = states == "OK"
        regime_switch = np.where(calm, msp, -ood_pct / 100.0 - 1.0)
        sig = {
            "msp": msp, "ensemble_conf": np.maximum(ens_mean, 1 - ens_mean), "ensemble_disagreement": -ens.std(axis=0),
            "conformal_pvalue": pvals, "va_width": -va_width, "ood_knn": -ood_pct, "auditor": -p_err,
            "composite": composite, "sl_fusion": sl,
            "gated": gated, "min_rank": min_rank, "regime_switch": regime_switch,
        }
        extras = {"probs": probs, "set_sizes": set_sizes, "ood_flag_rate": float(flagged.mean()),
                  "martingale_final": str(states[-1]), "martingale_alarm_share": float((states == "ALARM").mean()),
                  "martingale_watch_or_alarm_share": float((states != "OK").mean())}
        return sig, extras


def corrupt(X, kind, rho, rng, names):
    X = X.copy()
    n = len(X)
    rows = rng.choice(n, int(round(rho * n)), replace=False)
    mask = np.zeros(n, bool)
    mask[rows] = True
    if kind == "unit":
        money = [i for i, f in enumerate(names) if f.startswith(("LIMIT_BAL", "BILL_AMT", "PAY_AMT"))]
        X[np.ix_(rows, money)] *= 100.0
    elif kind == "missing":
        for r in rows:
            cols = rng.choice(X.shape[1], X.shape[1] // 2, replace=False)
            X[r, cols] = 0.0
    else:
        sd = X.std(axis=0)
        X[rows] += rng.normal(0, 1, (len(rows), X.shape[1])) * sd
    return X, mask


def evaluate(ref, X_te, y_te, corrupted=None, seed=0):
    sig, ex = ref.signals(X_te)
    probs = ex["probs"]
    pred = (probs[:, 1] >= 0.5).astype(int)
    wrong = (pred != y_te).astype(int)
    bad = ((pred == 0) & (y_te == 1)).astype(int)
    sig["random"] = np.random.default_rng(10_000 + seed).random(len(y_te))
    sig["oracle"] = -bad.astype(float)
    k = max(1, int(round(REFERRAL * len(y_te))))
    row = {"auc": float(roc_auc_score(y_te, probs[:, 1])), "accuracy": float((pred == y_te).mean()),
           "default_rate": float(y_te.mean()), "bad_approval_rate": float(bad.mean()),
           "coverage_at_95": float(np.mean([y_te[i] in np.flatnonzero(1.0 - probs[i] <= ref.qhat) for i in range(len(y_te))])),
           "review_rate": float((ex["set_sizes"] > 1).mean()), "ood_flag_rate": ex["ood_flag_rate"],
           "martingale_final": ex["martingale_final"], "martingale_watch_or_alarm_share": ex["martingale_watch_or_alarm_share"],
           "signals": {}}
    base = M.default_capture(sig["msp"], bad, REFERRAL)
    rand = M.default_capture(sig["random"], bad, REFERRAL)
    for name, s in sig.items():
        cap = M.default_capture(s, bad, REFERRAL)
        flagged = np.argsort(s)[:k]
        entry = {"bad_capture": float(cap), "err_capture": float(M.error_capture(s, wrong, REFERRAL)),
                 "vs_confidence": float(cap / base) if base > 0 else np.nan, "vs_random": float(cap / rand) if rand > 0 else np.nan,
                 "aurc": float(M.aurc(s, wrong))}
        if corrupted is not None:
            entry["corrupted_caught"] = float(corrupted[flagged].sum() / max(corrupted.sum(), 1))
        row["signals"][name] = entry
    return row


def aggregate(rows):
    out = {k: ms([r[k] for r in rows]) for k in rows[0] if k not in ("signals", "martingale_final")}
    out["martingale_final"] = [r["martingale_final"] for r in rows]
    out["signals"] = {name: {k: ms([r["signals"][name][k] for r in rows]) for k in rows[0]["signals"][name]}
                      for name in rows[0]["signals"]}
    return out


def main():
    t0 = time.time()
    ds = D.load_taiwan()
    X, y, names = ds["X"], ds["y"], ds["feature_names"]
    p0 = names.index("PAY_0")
    ref_pool = np.flatnonzero(X[:, p0] <= 0)
    shift_pool = np.flatnonzero(X[:, p0] >= 1)
    res = {"meta": {"dataset": "taiwan", "seeds": list(SEEDS), "n_train": N_TRAIN, "n_cal": N_CAL, "n_test": N_TEST,
                    "referral": REFERRAL, "alpha": ALPHA, "lambdas": LAMBDAS, "noise": {k: v for k, v in NOISE.items()},
                    "status": "exploratory, not pre-registered"}, "mix": {}, "noise": {}}
    per = {"mix": {str(l): [] for l in LAMBDAS}, "noise": {k: [] for k in NOISE}}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ref = rng.permutation(ref_pool)
        i_tr, i_cal, ref_rest = ref[:N_TRAIN], ref[N_TRAIN:N_TRAIN + N_CAL], ref[N_TRAIN + N_CAL:]
        sh = rng.permutation(shift_pool)
        R = Reference(X[i_tr], y[i_tr], X[i_cal], y[i_cal], seed)
        for lam in LAMBDAS:
            n_sh = int(round(lam * N_TEST))
            i_te = np.concatenate([ref_rest[:N_TEST - n_sh], sh[:n_sh]])
            i_te = rng.permutation(i_te)
            row = evaluate(R, X[i_te], y[i_te], None, seed)
            per["mix"][str(lam)].append(row)
            s = row["signals"]
            print(f"  seed {seed} mix {lam}: auc {row['auc']:.3f} review {row['review_rate']:.0%} mart {row['martingale_final']} | "
                  f"vs random: msp {s['msp']['vs_random']:.2f} ood {s['ood_knn']['vs_random']:.2f} gated {s['gated']['vs_random']:.2f} "
                  f"minrank {s['min_rank']['vs_random']:.2f} switch {s['regime_switch']['vs_random']:.2f} sl {s['sl_fusion']['vs_random']:.2f}", flush=True)
        i_te = rng.permutation(ref_rest[:N_TEST])
        for key, (kind, rho) in NOISE.items():
            Xc, mask = corrupt(X[i_te], kind, rho, np.random.default_rng(500 + seed), names)
            row = evaluate(R, Xc, y[i_te], mask, seed)
            per["noise"][key].append(row)
            s = row["signals"]
            print(f"  seed {seed} noise {key}: auc {row['auc']:.3f} review {row['review_rate']:.0%} mart {row['martingale_final']} | "
                  f"vs random: msp {s['msp']['vs_random']:.2f} ood {s['ood_knn']['vs_random']:.2f} gated {s['gated']['vs_random']:.2f} "
                  f"minrank {s['min_rank']['vs_random']:.2f} switch {s['regime_switch']['vs_random']:.2f} | corrupted caught: msp {s['msp']['corrupted_caught']:.2f} ood {s['ood_knn']['corrupted_caught']:.2f}", flush=True)
    res["mix"] = {k: aggregate(v) for k, v in per["mix"].items()}
    res["noise"] = {k: aggregate(v) for k, v in per["noise"].items()}
    res["meta"]["seconds"] = round(time.time() - t0, 1)
    (RESULTS / "shift_bench_v2.json").write_text(json.dumps(res, indent=1))
    print(f"wrote results/shift_bench_v2.json in {res['meta']['seconds']} s")


if __name__ == "__main__":
    main()
