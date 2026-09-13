"""Tests for reliax_core.calibration_trust. Run: .venv/bin/python tests/test_calibration_trust.py
(also collectable by pytest)."""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from reliax_core.calibration_trust import (CalibrationTrust, MARGINAL, averaging_fusion,  # noqa: E402
                                           bpq, closed_form_rate_opinion, cumulative_fusion,
                                           hoeffding_bound, noise_floor)


def synth(n, seed, T=1.0):
    """Credit-like scores. True default prob p ~ Beta(2, 6); the model STATES
    q = sigmoid(logit(p)/T): T=1 calibrated, T<1 over-confident, T>1 under-confident."""
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 6, n)
    y = (rng.random(n) < p).astype(int)
    q = 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))
    return q, y, p


def test_bpq_sums_to_one():
    o = bpq(3, 1)
    assert abs(o["b"] + o["d"] + o["u"] - 1) < 1e-12
    assert abs(o["b"] - 3 / 6) < 1e-12 and abs(o["u"] - 2 / 6) < 1e-12


def test_cumulative_fusion_is_evidence_addition():
    ops = [bpq(2, 1), bpq(5, 0), bpq(0, 4)]
    f = cumulative_fusion(ops)
    assert (f["r"], f["s"]) == (7, 5)
    a = averaging_fusion(ops)
    assert abs(a["r"] - 7 / 3) < 1e-12


def test_closed_form_rate_opinion_all_binnings():
    q, y, _ = synth(5000, 0, T=1.6)
    for binning in ("fixed", "quantile", "isotonic"):
        for rep in ("mid", "mean"):
            ct = CalibrationTrust(M=12, binning=binning, representative=rep).fit(q, y)
            cf = closed_form_rate_opinion(len(q), ct.binned_ece())
            g = ct.global_opinion
            assert abs(g["d"] - cf["d"]) < 1e-9, (binning, rep)
            assert abs(g["b"] - cf["b"]) < 1e-9
            assert abs(g["u"] - 2 / (len(q) + 2)) < 1e-12


def test_chapter_accuracy_mode_matches_hand_computation():
    q, y, _ = synth(3000, 1)
    pred = (q >= 0.5).astype(int)
    conf = np.where(pred == 1, q, 1 - q)          # confidence of the stated class
    correct = (pred == y).astype(int)
    ct = CalibrationTrust(M=10, binning="fixed", representative="mid", evidence="accuracy").fit(conf, correct)
    edges = np.linspace(0, 1, 11)
    R = S = 0.0
    for i in range(10):
        m = (conf >= edges[i]) & (conf < edges[i + 1]) if i < 9 else (conf >= edges[9])
        if m.sum() == 0:
            continue
        n_i, t_i, rp = m.sum(), correct[m].sum(), (edges[i] + edges[i + 1]) / 2
        R += t_i
        S += abs(t_i - n_i * rp)
    g = ct.global_opinion
    assert abs(g["r"] - R) < 1e-9 and abs(g["s"] - S) < 1e-6


def test_M_to_infinity_limit_is_L1_error():
    q, y, _ = synth(200, 2)            # N^2 / 2M = 0.01: bins are singletons w.h.p.
    M = 2_000_000
    ct = CalibrationTrust(M=M, binning="fixed", representative="mid").fit(q, y)
    l1 = float(np.mean(np.abs(y - q)))
    assert abs(ct.binned_ece() - l1) <= 1 / (2 * M) + 1e-9
    assert abs(ct.global_opinion["d"] - len(q) / (len(q) + 2) * l1) < 1e-6


def test_calibrated_model_bias_and_debias():
    d = {k: [] for k in ("none", "floor", "l2")}
    for seed in range(20):
        q, y, _ = synth(20000, 100 + seed)
        for k in d:
            ct = CalibrationTrust(M=200, binning="quantile", representative="mean", debias=k).fit(q, y)
            d[k].append(ct.global_opinion["d"] * (len(q) + 2) / len(q))   # undo the prior shrink
    raw, fl, l2 = (float(np.mean(v)) for v in d.values())
    # first-order prediction of the raw bias: mean over cells of sqrt(2 rp (1-rp) / (pi n))
    ct = CalibrationTrust(M=200, binning="quantile", representative="mean").fit(*synth(20000, 999)[:2])
    pred = np.mean([noise_floor(c["rp"], c["n"]) for c in ct.cells.values()])
    assert abs(raw - pred) / pred < 0.15, (raw, pred)
    assert 0.2 * raw < fl < 0.42 * raw, (raw, fl)     # theory: 0.30
    assert 0.3 * raw < l2 < 0.55 * raw, (raw, l2)     # theory: 0.43


def test_realized_matches_fit_on_same_data_and_detects_shift():
    q, y, p = synth(8000, 11, T=1.0)
    ct = CalibrationTrust(M=10, binning="quantile", representative="mean").fit(q, y)
    same = ct.realized(q, y)[MARGINAL]
    assert abs(same["opinion"]["d"] - ct.global_opinion["d"]) < 1e-9
    # shifted outcomes: defaults become much more likely than stated
    rng = np.random.default_rng(12)
    y_shift = (rng.random(8000) < np.minimum(p * 1.8, 1)).astype(int)
    shifted = ct.realized(q, y_shift)[MARGINAL]
    assert shifted["gap_d"] > 0.05


def test_hoeffding_bound_holds():
    q, _, p = synth(400000, 7, T=1.6)
    edges = np.linspace(0, 1, 11)
    hits = 0
    trials = 300
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.choice(len(q), 3000, replace=False)
        y = (rng.random(3000) < p[idx]).astype(int)
        ct = CalibrationTrust(M=10, binning="fixed", representative="mean").fit(q[idx], y)
        # population binned ECE for these bins and these representatives
        pop = 0.0
        for (g, i), c in ct.cells.items():
            m = (q >= edges[i]) & (q < edges[i + 1]) if i < 9 else (q >= edges[9])
            pop += c["n"] / 3000 * abs(p[m].mean() - c["rp"])
        bound = hoeffding_bound([c["n"] for c in ct.cells.values()], delta=0.05)
        hits += abs(ct.binned_ece() - pop) <= bound
    assert hits / trials >= 0.95, hits / trials


def test_lookup_segments_and_thin_cells():
    q, y, _ = synth(4000, 3)
    seg = np.where(np.arange(4000) % 50 == 0, "thin", "main")
    ct = CalibrationTrust(M=10, binning="quantile", min_cell_n=30).fit(q, y, segments=seg)
    c = ct.lookup(0.2, "main")
    assert c["segment"] == "main" and not c["fallback_to_marginal"]
    t = ct.lookup(0.2, "thin")
    assert t["segment"] == "thin" and t["insufficient_evidence"]
    u = ct.lookup(0.2, "unknown-segment")
    assert u["segment"] == MARGINAL and u["fallback_to_marginal"]
    line = ct.certificate_line(0.2, "thin", bracket=(0.15, 0.22))
    assert "insufficient evidence" in line and "disbelief" in line


def test_alpha_beta_knob():
    q, y, _ = synth(6000, 4, T=0.5)   # over-confident: stated q more extreme than p
    sym = CalibrationTrust(M=10, alpha=1, beta=1).fit(q, y).global_opinion["d"]
    under = CalibrationTrust(M=10, alpha=2, beta=1).fit(q, y).global_opinion["d"]
    over = CalibrationTrust(M=10, alpha=1, beta=2).fit(q, y).global_opinion["d"]
    assert under >= sym and over >= sym and under != over


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} tests passed")
