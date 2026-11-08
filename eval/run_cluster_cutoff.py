"""A cut-off per applicant from soft clusters, against one global cut-off (EXPLORATORY).

Question: can a cut-off that adapts to the applicant earn more than one cut-off for
everyone? Lender economics of Amendment 4's worked example: +M for an approved good
customer, -L for an approved default, M = 600, L = 5,000, break-even PD* = M / (M + L).

Each of 20 re-draws splits the rows into training (40%), calibration (30%) and test (30%).
A model is trained on the training rows. On the calibration rows, the model's inputs are
standardised and cut into K clusters (k-means). A new applicant gets a weight for every
cluster, softmax(-squared distance to its centre / T), T the median squared distance of the
calibration rows to their nearest centre: its "projection on the cluster basis".
Approve when the model's score p is below the cut-off:
  G0  the break-even PD*, the same for everyone (the lender alone)
  G1  the single cut-off that earned most on the calibration rows
  H   one cut-off per cluster, the one that earned most on that cluster's calibration rows,
      applied by nearest cluster
  S   per applicant: t(x) = sum_k w_k(x) t_k, each t_k the cut-off that earned most on the
      calibration rows weighted by their weight for cluster k
Reported on the test rows: the margin per 1,000 applications, the share approved and the
default rate among approvals; mean and spread over the re-draws, and S against G1 and G0
paired by re-draw.

Two datasets:
  synthetic  30,000 applicants in four groups; the true risk depends on the inputs and on the
             group, and the model (a logistic regression on the inputs) cannot see the group,
             so it is wrong by a different amount in each group: the case a per-cluster
             cut-off is built for. Two variants scale that group error: by 0
             (synthetic_no_group_error: the model is only wrong where the inputs cannot tell)
             and by 2 (synthetic_double_group_error)
  taiwan     UCI Taiwan credit-card default, 30,000 card holders, 20 inputs (sex, age and
             marital status are never used, for the model or the clusters); gradient
             boosting as in the platform's demo

Usage: python eval/run_cluster_cutoff.py [synthetic synthetic_no_group_error synthetic_double_group_error taiwan]
Writes results/cluster_cutoff/summary.json
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import data as D

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "cluster_cutoff"
M, L = 600.0, 5000.0
PD_STAR = M / (M + L)
RESPLITS = 20
KS = (4, 8)
SCALES = {"synthetic_no_group_error": 0.0, "synthetic": 1.0, "synthetic_double_group_error": 2.0}
GRID = np.unique(np.concatenate([np.linspace(0, 0.5, 101), [PD_STAR, 1.0]]))


def synthetic(scale=1.0, seed=7, n=30_000):
    rng = np.random.default_rng(seed)
    centres = np.array([[-2, -2, 0, 0], [2, -2, 0, 0], [-2, 2, 0, 0], [2, 2, 0, 0]], dtype=float)
    shift = scale * np.array([0.0, 0.0, 1.4, -1.0])   # each group's risk, beyond what the inputs show
    g = rng.integers(0, 4, n)
    X = centres[g] + rng.normal(size=(n, 4))
    logit = -2.2 + 0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2] + shift[g]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def load(name):
    if name.startswith("synthetic"):
        X, y = synthetic(SCALES[name])
        return X, y, lambda Xt, yt, s: LogisticRegression(max_iter=1000).fit(Xt, yt)
    ds = D.load_taiwan()
    return ds["X"].astype(float), ds["y"].astype(int), \
        lambda Xt, yt, s: HistGradientBoostingClassifier(max_iter=300, random_state=s).fit(Xt, yt)


def margin(p, y, t):
    a = p < t
    return M * ((y == 0) & a).sum() - L * ((y == 1) & a).sum()


def best_cut(p, y, w=None):
    w = np.ones(len(y)) if w is None else w
    gain = [M * (w * ((y == 0) & (p < t))).sum() - L * (w * ((y == 1) & (p < t))).sum() for t in GRID]
    return float(GRID[int(np.argmax(gain))])


def weights(Z, km, T):
    d2 = ((Z[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
    e = np.exp(-(d2 - d2.min(1, keepdims=True)) / T)
    return e / e.sum(1, keepdims=True)


def figures(p, y, t):
    a = p < t
    return {"margin_per_1000": float(1000 * margin(p, y, t) / len(y)), "approved": float(a.mean()),
            "default_among_approved": float(y[a].mean()) if a.any() else None}


def run(name):
    X, y, fit = load(name)
    rows = {f"{m}{'' if m in ('G0', 'G1') else '_K' + str(k)}": [] for k in KS for m in ("G0", "G1", "H", "S")}
    for r in range(RESPLITS):
        idx = np.random.default_rng(r).permutation(len(y))
        a, b = int(0.4 * len(y)), int(0.7 * len(y))
        tr, ca, te = idx[:a], idx[a:b], idx[b:]
        model = fit(X[tr], y[tr], r)
        pc, pt = model.predict_proba(X[ca])[:, 1], model.predict_proba(X[te])[:, 1]
        sc = StandardScaler().fit(X[ca])
        Zc, Zt = sc.transform(X[ca]), sc.transform(X[te])
        rows["G0"].append(figures(pt, y[te], np.full(len(te), PD_STAR)))
        rows["G1"].append(figures(pt, y[te], np.full(len(te), best_cut(pc, y[ca]))))
        for k in KS:
            km = KMeans(n_clusters=k, n_init=10, random_state=r).fit(Zc)
            hard_c, hard_t = km.labels_, km.predict(Zt)
            cut_h = np.array([best_cut(pc[hard_c == c], y[ca][hard_c == c]) for c in range(k)])
            rows[f"H_K{k}"].append(figures(pt, y[te], cut_h[hard_t]))
            d2 = ((Zc[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
            T = float(np.median(d2.min(1)))
            Wc, Wt = weights(Zc, km, T), weights(Zt, km, T)
            cut_s = np.array([best_cut(pc, y[ca], Wc[:, c]) for c in range(k)])
            rows[f"S_K{k}"].append(figures(pt, y[te], Wt @ cut_s))
        g0, g1 = rows["G0"][-1]["margin_per_1000"], rows["G1"][-1]["margin_per_1000"]
        print(f"  {name} re-draw {r}: margin/1000 G0 {g0:.0f} G1 {g1:.0f} " +
              " ".join(f"S_K{k} {rows[f'S_K{k}'][-1]['margin_per_1000']:.0f}" for k in KS), flush=True)
    out = {}
    for key, rs in rows.items():
        mg = np.array([x["margin_per_1000"] for x in rs])
        out[key] = {"margin_per_1000": {"mean": round(float(mg.mean()), 1), "sd": round(float(mg.std(ddof=1)), 1)},
                    "approved": round(float(np.mean([x["approved"] for x in rs])), 4),
                    "default_among_approved": round(float(np.mean([x["default_among_approved"] for x in rs
                                                                   if x["default_among_approved"] is not None])), 4)}
    paired = {}
    for k in KS:
        s = np.array([x["margin_per_1000"] for x in rows[f"S_K{k}"]])
        for base in ("G0", "G1"):
            d = s - np.array([x["margin_per_1000"] for x in rows[base]])
            ref = np.mean([x["margin_per_1000"] for x in rows[base]])
            paired[f"S_K{k}_vs_{base}"] = {"mean": round(float(d.mean()), 1), "sd": round(float(d.std(ddof=1)), 1),
                                          "relative": round(float(d.mean() / abs(ref)), 4),
                                          "redraws_better": int((d > 0).sum())}
    return {"meta": {"dataset": name, "status": "exploratory", "M": M, "L": L, "pd_star": round(PD_STAR, 4),
                     "resplits": RESPLITS, "split": "40% train, 30% calibration, 30% test", "ks": list(KS),
                     "n": int(len(y)), "default_rate": round(float(y.mean()), 4)},
            "methods": out, "paired": paired}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or [*SCALES, "taiwan"]
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        print(f"[{name}]", flush=True)
        summary[name] = run(name)
    path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
