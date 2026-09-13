"""Workstream B1: per-decision envelope latency at 7,500 / 100,000 / 1,000,000
calibration rows (exploratory; Amendment 2, section 2.7).

Latency is a property of the engine, not of a dataset. The calibration sets
are built by seeded bootstrap of the Taiwan calibration split (seed-0 split of
`eval/run_eval.py`, 7,500 rows, 23 features) with Gaussian jitter at 5% of
each feature's standard deviation, and probabilities from the same trained
model. This is stated wherever the number appears. The implementation
measured is `reliax_core/` unchanged.

Per calibration size, per component: 100 warm-up queries, then 1,000 timed
queries, 3 repeats; median and p95 in milliseconds over all timed queries.
Components: conformal (prediction set + p-value), Venn-Abers interval, kNN
distance + percentile with the exact scikit-learn index and with an HNSW
index (recall@10 against exact reported), martingale update, and the full
envelope (model + conformal + VA + exact kNN + martingale). The loopback HTTP
hop is measured separately on a minimal JSON endpoint and labelled as a
same-host round trip, not an in-VPC hop.

Venn-Abers at 1,000,000 rows is slow by construction (two isotonic fits per
query); if a single query exceeds VA_QUERY_BUDGET_S the timed query count for
that cell is reduced and the reduction is recorded in the output.

Writes results/expansion/latency_scale.json and results/expansion/LATENCY.md.
"""
import http.client
import http.server
import json
import pathlib
import sys
import threading
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

BASE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "eval"))

from reliax_core.conformal import ConformalCalibrator     # noqa: E402
from reliax_core.martingale import ConformalMartingale     # noqa: E402
from reliax_core.ood import KNNOODDetector                 # noqa: E402
from reliax_core.venn_abers import VennAbersCalibrator     # noqa: E402
import data as D                                           # noqa: E402

OUT = BASE / "results" / "expansion"
OUT.mkdir(parents=True, exist_ok=True)
SIZES = (7_500, 100_000, 1_000_000)
WARMUP, QUERIES, REPEATS = 100, 1_000, 3
ALPHA = 0.05
JITTER = 0.05
VA_QUERY_BUDGET_S = 2.0       # if one VA query takes longer, reduce the timed count for that cell
HNSW_M, HNSW_EF_C, HNSW_EF = 16, 200, 64


def build_calibration(X_cal, y_cal, model, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_cal), n, replace=True)
    X = X_cal[idx] + rng.normal(0.0, 1.0, (n, X_cal.shape[1])) * (JITTER * X_cal.std(axis=0))
    y = y_cal[idx]
    probs = model.predict_proba(X)
    return X, y, probs


def timed(fn, args_iter, n_warm, n_timed):
    it = iter(args_iter)
    for _ in range(n_warm):
        fn(next(it))
    lat = np.empty(n_timed)
    for i in range(n_timed):
        a = next(it)
        t0 = time.perf_counter()
        fn(a)
        lat[i] = (time.perf_counter() - t0) * 1000.0
    return lat


def stats(lat_ms):
    return {"p50_ms": round(float(np.percentile(lat_ms, 50)), 4),
            "p95_ms": round(float(np.percentile(lat_ms, 95)), 4),
            "mean_ms": round(float(np.mean(lat_ms)), 4), "n": int(len(lat_ms))}


def cycle(X, rng):
    while True:
        for i in rng.permutation(len(X)):
            yield X[i]


class HNSWIndex:
    """Approximate kNN with the same interface the envelope uses (distance,
    percentile) so the comparison is like for like: same scaler, same k."""

    def __init__(self, ood: KNNOODDetector, k: int):
        import hnswlib
        Z = ood.scaler.transform(ood._X_fit)
        self.k = k
        self.index = hnswlib.Index(space="l2", dim=Z.shape[1])
        self.index.init_index(max_elements=len(Z), M=HNSW_M, ef_construction=HNSW_EF_C)
        self.index.add_items(Z.astype(np.float32))
        self.index.set_ef(HNSW_EF)
        self.scaler, self.calib_dists = ood.scaler, ood.calib_dists

    def distance(self, x):
        z = self.scaler.transform(np.atleast_2d(x)).astype(np.float32)
        _, d2 = self.index.knn_query(z, k=self.k)
        return float(np.sqrt(np.maximum(d2, 0.0)).mean())

    def percentile(self, x):
        d = self.distance(x)
        return 100.0 * int(np.searchsorted(self.calib_dists, d, side="right")) / len(self.calib_dists)

    def neighbours(self, X):
        z = self.scaler.transform(X).astype(np.float32)
        labels, _ = self.index.knn_query(z, k=self.k)
        return labels


def loopback_hop(n=1_000):
    """Same-host HTTP round trip on a minimal JSON endpoint, keep-alive."""
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            out = json.dumps({"ok": True, "n": len(body)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port)
    payload = json.dumps({"features": list(np.zeros(23))}).encode()
    lat = []
    for i in range(WARMUP + n):
        t0 = time.perf_counter()
        conn.request("POST", "/decide", body=payload, headers={"Content-Type": "application/json"})
        conn.getresponse().read()
        if i >= WARMUP:
            lat.append((time.perf_counter() - t0) * 1000.0)
    conn.close()
    srv.shutdown()
    return stats(np.array(lat))


def main():
    ds = D.load_taiwan()
    X, y = ds["X"], ds["y"]
    idx = np.arange(len(y))
    idx_tr, idx_rest = train_test_split(idx, test_size=0.5, random_state=0, stratify=y)
    idx_cal, idx_te = train_test_split(idx_rest, test_size=0.5, random_state=0, stratify=y[idx_rest])
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X[idx_tr], y[idx_tr])
    X_cal0, y_cal0, X_te = X[idx_cal], y[idx_cal], X[idx_te]
    p_te = model.predict_proba(X_te)
    results = {"meta": {"date": time.strftime("%Y-%m-%d"), "sizes": list(SIZES),
                        "warmup": WARMUP, "queries": QUERIES, "repeats": REPEATS,
                        "calibration_rows": "seeded bootstrap of the Taiwan seed-0 calibration split "
                                            f"(7,500 rows, 23 features) with Gaussian jitter at {JITTER:.0%} "
                                            "of each feature's std; not real applicants beyond 7,500",
                        "hnsw": {"M": HNSW_M, "ef_construction": HNSW_EF_C, "ef": HNSW_EF},
                        "protocol": "PREREGISTRATION_AMENDMENT.md, Amendment 2, section 2.7 (B1)"},
               "sizes": {}}
    print("loopback hop ...", flush=True)
    results["loopback_http_hop"] = loopback_hop()
    print(f"  {results['loopback_http_hop']}", flush=True)

    for n in SIZES:
        print(f"=== n = {n:,}", flush=True)
        t0 = time.time()
        Xc, yc, pc = build_calibration(X_cal0, y_cal0, model, n, seed=0)
        conf = ConformalCalibrator(pc, yc)
        va = VennAbersCalibrator(pc[:, 1], yc)
        t_fit = time.time()
        ood = KNNOODDetector(k=10).fit(Xc)
        ood._X_fit = Xc
        fit_exact_s = time.time() - t_fit
        t_fit = time.time()
        hnsw = HNSWIndex(ood, k=10)
        fit_hnsw_s = time.time() - t_fit
        mart = ConformalMartingale(ood.calib_dists, seed=0)
        cell = {"build_s": {"calibration_set": round(t_fit - t0, 1),
                            "knn_exact_index": round(fit_exact_s, 2),
                            "hnsw_index": round(fit_hnsw_s, 2)},
                "components": {}}
        # recall@10 of HNSW against the exact index on 2,000 test rows
        Z = ood.scaler.transform(X_te[:2000])
        _, exact_nb = ood.nn.kneighbors(Z, n_neighbors=10)
        approx_nb = hnsw.neighbours(X_te[:2000])
        recall = np.mean([len(set(a) & set(b)) / 10.0 for a, b in zip(exact_nb, approx_nb)])
        cell["hnsw_recall_at_10"] = round(float(recall), 4)

        comps = {
            "conformal": (lambda p: (conf.prediction_set(p, ALPHA), conf.p_value(float(p.max()))),
                          lambda rng: cycle(p_te, rng)),
            "knn_exact": (lambda x: ood.percentile(x), lambda rng: cycle(X_te, rng)),
            "knn_hnsw": (lambda x: hnsw.percentile(x), lambda rng: cycle(X_te, rng)),
            "martingale": (lambda d: mart.update(float(d)), lambda rng: cycle(ood.calib_dists, rng)),
            "venn_abers": (lambda s: va.interval(float(s)), lambda rng: cycle(p_te[:, 1], rng)),
        }

        def full(x):
            p = model.predict_proba(x.reshape(1, -1))[0]
            conf.prediction_set(p, ALPHA)
            conf.p_value(float(p.max()))
            va.interval(float(p[1]))
            d = ood.distance(x)
            mart.update(d)
        comps["full_envelope"] = (full, lambda rng: cycle(X_te, rng))

        for name, (fn, gen) in comps.items():
            n_timed, n_warm = QUERIES, WARMUP
            if name in ("venn_abers", "full_envelope"):
                t1 = time.perf_counter()
                fn(next(gen(np.random.default_rng(99))))
                one = time.perf_counter() - t1
                if one > VA_QUERY_BUDGET_S:
                    n_timed = max(20, int(60.0 / one))
                    n_warm = 3
            lat = np.concatenate([timed(fn, gen(np.random.default_rng(r)), n_warm, n_timed)
                                  for r in range(REPEATS)])
            cell["components"][name] = stats(lat)
            if n_timed != QUERIES:
                cell["components"][name]["reduced_queries"] = n_timed
            print(f"  {name:14s} p50 {cell['components'][name]['p50_ms']:9.3f} ms  "
                  f"p95 {cell['components'][name]['p95_ms']:9.3f} ms  (n={len(lat)})", flush=True)
        results["sizes"][str(n)] = cell
        (OUT / "latency_scale.json").write_text(json.dumps(results, indent=1))
        write_markdown(results)
    print("done")


def write_markdown(R):
    L = ["# Workstream B1: envelope latency at scale (Amendment 2, section 2.7)", "",
         f"Generated by `eval/expansion/run_latency_scale.py` on {R['meta']['date']}. "
         f"Calibration rows: {R['meta']['calibration_rows']}. Per cell: {R['meta']['warmup']} "
         f"warm-up queries, {R['meta']['queries']} timed queries, {R['meta']['repeats']} repeats "
         "(a cell marked * had its query count reduced because one query exceeded "
         f"{VA_QUERY_BUDGET_S:.0f} s). Single process, single thread, Apple Silicon laptop; "
         "no network. Loopback HTTP hop measured separately below.", "",
         "| component | " + " | ".join(f"n = {int(n):,} p50 / p95" for n in R["sizes"]) + " |",
         "|---|" + "---|" * len(R["sizes"])]
    names = ["conformal", "venn_abers", "knn_exact", "knn_hnsw", "martingale", "full_envelope"]
    for name in names:
        cells = []
        for n in R["sizes"]:
            c = R["sizes"][n]["components"].get(name)
            if c is None:
                cells.append("")
                continue
            star = "*" if "reduced_queries" in c else ""
            cells.append(f"{c['p50_ms']:.2f} / {c['p95_ms']:.2f} ms{star}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "| index build | " + " | ".join(f"n = {int(n):,}" for n in R["sizes"]) + " |",
          "|---|" + "---|" * len(R["sizes"]),
          "| exact kNN (scikit-learn) | " + " | ".join(f"{R['sizes'][n]['build_s']['knn_exact_index']} s" for n in R["sizes"]) + " |",
          "| HNSW | " + " | ".join(f"{R['sizes'][n]['build_s']['hnsw_index']} s" for n in R["sizes"]) + " |",
          "| HNSW recall@10 vs exact | " + " | ".join(f"{R['sizes'][n]['hnsw_recall_at_10']:.3f}" for n in R["sizes"]) + " |",
          "", f"Loopback HTTP hop (same host, keep-alive, JSON body): p50 "
          f"{R['loopback_http_hop']['p50_ms']:.3f} ms, p95 {R['loopback_http_hop']['p95_ms']:.3f} ms. "
          "This is not an in-VPC hop; that is measured in the pilot deployment.", ""]
    big = R["sizes"].get("1000000")
    if big:
        fe = big["components"].get("full_envelope")
        if fe:
            L.append(f"H-B1 (full-envelope median at 1,000,000 rows above 100 ms with the frozen "
                     f"Venn-Abers): {'met' if fe['p50_ms'] > 100 else 'not met'} ({fe['p50_ms']:.1f} ms).")
        dom = max(big["components"].items(), key=lambda kv: kv[1]["p50_ms"] if kv[0] != "full_envelope" else -1)
        L.append(f"Dominant component at 1,000,000 rows: {dom[0]} ({dom[1]['p50_ms']:.1f} ms median).")
    (OUT / "LATENCY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
