"""Calibration-trust opinions: the model-assessment stage (FUSION 2025 lineage).

Turns the calibration behaviour of a black-box scorer into subjective-logic
opinions omega = (b, d, u) on the proposition "the model's stated probability
can be trusted at this score level", per (segment x score-bin) cell, and fused
into a global opinion per segment and overall.

Two evidence modes
------------------
'accuracy' (the FUSION 2025 chapter, verbatim). Per bin i, with n_i predictions,
    t_i correct ones and representative probability RP_i:
        r_i = t_i,            s_i = |t_i - n_i * RP_i|.
    The opinion conflates accuracy and calibration by design (a calibrated but
    inaccurate model gets low belief).

'rate' (credit adaptation, this module). The scorer emits p(default); the
    outcome is default / no default. Per bin i, with k_i defaults observed:
        dev_i = k_i / n_i - RP_i                      (observed rate - predicted)
        s_i   = n_i * ( alpha * max(dev_i, 0) + beta * max(-dev_i, 0) )
        r_i   = n_i - s_i                              (clipped at 0)
    alpha weighs UNDER-estimation of the default rate (costs capital), beta
    weighs OVER-estimation (costs volume). alpha = beta = 1 is symmetric.

Quantification is the baseline-prior quantification (BPQ) of the thesis:
    b = r / (r + s + W),  d = s / (r + s + W),  u = W / (r + s + W),  W = 2.
Cumulative fusion of binomial opinions with equal base rates is evidence
addition, so the fused opinion over cells is BPQ(sum r_i, sum s_i). Cells are
disjoint (a decision lives in exactly one), which is the case cumulative
fusion is meant for.

What this module adds to the chapter (all tested in eval/run_calibration_trust.py
and stated with proofs in CALIBRATION_TRUST.md):

1. Closed form. In 'rate' mode with alpha = beta = 1 and no noise floor,
   r_i + s_i = n_i, hence for N calibration points
        d = N/(N+W) * ECE_M,   b = N/(N+W) * (1 - ECE_M),   u = W/(N+W),
   where ECE_M = sum_i (n_i/N) |k_i/n_i - RP_i| is the binned L1 calibration
   error with the chosen representatives. Disbelief IS the binned ECE, shrunk
   by the prior; uncertainty is a function of N only.

2. The M -> infinity limit is NOT calibration. With fixed N and midpoint
   representatives, as bins shrink every bin holds one point and
        ECE_M -> (1/N) sum_j |y_j - p_j|,
   whose expectation for a perfectly calibrated model is E[2p(1-p)] > 0. So
   the chapter's sweep over M converges to a sharpness-penalised quantity,
   which is why calibrated CIFAR-10 belief peaks and then declines. The
   chapter's "more clusters, lower uncertainty" is a by-product of disbelief
   inflating with M, not of more evidence.

3. Small-sample bias. Under exact calibration in a bin, E|k_i/n_i - RP_i| is
   about sqrt(2 RP_i (1-RP_i) / (pi n_i)) (half-normal mean), so the raw
   negative evidence of a perfectly calibrated model is positive and grows
   like sqrt(M/N). Two corrections are offered:
     debias='floor'  soft-threshold: count only the deviation above the null
                     expected deviation (first-order bias under calibration
                     shrinks by the factor E[(|Z|-sqrt(2/pi))_+] / E|Z| = 0.30,
                     at the price of a downward bias of at most the floor
                     when the bin really is miscalibrated);
     debias='l2'     the unbiased plug-in for (mu_i - RP_i)^2 of Kumar, Liang
                     and Ma (2019), (k/n - RP)^2 - k/n (1 - k/n) / (n - 1),
                     square-rooted at zero-floor: asymptotically unbiased for
                     large true deviations, null bias factor 0.43.
   Both estimators, and the raw one, are consistent when N/M -> infinity; the
   corrections change the constant, not the rate. Proofs and the measured
   constants are in CALIBRATION_TRUST.md.

4. Representatives and bins. 'mean' representative (mean stated probability
   in the bin) removes the midpoint approximation error the chapter flagged.
   'quantile' bins give equal-mass cells (no empty tails). 'isotonic' bins are
   the blocks of the isotonic fit of outcomes on scores: the data-adaptive
   monotone partition the chapter's limitations section proposed, and exactly
   the partition Venn-Abers works on.

5. Segment axis. Cells are (segment x bin); a thin cell (n < min_cell_n) is
   flagged insufficient_evidence at lookup. Segments are the customer's
   business segments (Mondrian groups), never protected classes at inference.

Limits (unchanged from the chapter): the opinion is per cell, not per
applicant; it is fitted on reference data and says nothing about a shifted
population until outcomes from that population are observed.
"""
import math

import numpy as np
from sklearn.isotonic import IsotonicRegression

W_DEFAULT = 2.0
BASE_RATE = 0.5
MARGINAL = "__all__"


# --------------------------------------------------------------------------- #
# Subjective-logic primitives
# --------------------------------------------------------------------------- #
def bpq(r: float, s: float, W: float = W_DEFAULT, a: float = BASE_RATE) -> dict:
    """Baseline-prior quantification of evidence (r positive, s negative)."""
    r = max(float(r), 0.0)
    s = max(float(s), 0.0)
    den = r + s + W
    b, d, u = r / den, s / den, W / den
    return {"b": b, "d": d, "u": u, "a": a, "expected": b + a * u, "r": r, "s": s, "W": W}


def cumulative_fusion(opinions: list[dict]) -> dict:
    """Cumulative fusion of BPQ opinions with a common W and base rate is
    evidence addition (Josang 2016, ch. 12; the opinions are independent
    observations of disjoint cells)."""
    if not opinions:
        return bpq(0.0, 0.0)
    W = opinions[0]["W"]
    return bpq(sum(o["r"] for o in opinions), sum(o["s"] for o in opinions), W, opinions[0]["a"])


def averaging_fusion(opinions: list[dict]) -> dict:
    """Averaging fusion for dependent sources that saw the SAME data (the
    chapter's across-class fusion): evidence is averaged, not summed."""
    if not opinions:
        return bpq(0.0, 0.0)
    n = len(opinions)
    W = opinions[0]["W"]
    return bpq(sum(o["r"] for o in opinions) / n, sum(o["s"] for o in opinions) / n, W, opinions[0]["a"])


def noise_floor(rp: float, n: int) -> float:
    """Expected |k/n - rp| when the bin is exactly calibrated at rate rp
    (half-normal mean of the binomial proportion, first order in 1/n)."""
    if n <= 0:
        return 0.0
    v = max(rp * (1.0 - rp), 0.0)
    return math.sqrt(2.0 * v / (math.pi * n))


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
def fixed_edges(M: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, M + 1)


def quantile_edges(scores: np.ndarray, M: int) -> np.ndarray:
    qs = np.quantile(scores, np.linspace(0.0, 1.0, M + 1))
    qs[0], qs[-1] = 0.0, 1.0
    edges = np.unique(qs)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    return edges


def isotonic_edges(scores: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Bin edges at the boundaries of the isotonic-regression blocks (pool-
    adjacent-violators) of outcomes on scores: the monotone, data-adaptive
    partition that Venn-Abers calibrates on. Number of bins is data-driven."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, outcomes)
    order = np.argsort(scores, kind="stable")
    fitted = iso.predict(scores[order])
    s_sorted = scores[order]
    # a new block starts wherever the fitted value changes
    change = np.flatnonzero(np.diff(fitted) > 1e-12) + 1
    cuts = [(s_sorted[i - 1] + s_sorted[i]) / 2.0 for i in change]
    edges = np.array([0.0] + cuts + [1.0])
    return np.unique(edges)


def assign_bins(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    M = len(edges) - 1
    return np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, M - 1)


# --------------------------------------------------------------------------- #
# The calibration-trust model
# --------------------------------------------------------------------------- #
class CalibrationTrust:
    """Fit once on (stated probability, outcome[, segment]) triples from the
    calibration split; query per decision in O(log M)."""

    def __init__(self, M: int = 10, binning: str = "quantile", representative: str = "mean",
                 evidence: str = "rate", alpha: float = 1.0, beta: float = 1.0,
                 debias: str = "none", W: float = W_DEFAULT, min_cell_n: int = 30):
        assert binning in ("fixed", "quantile", "isotonic")
        assert representative in ("mid", "mean")
        assert evidence in ("rate", "accuracy")
        assert debias in ("none", "floor", "l2")
        self.M, self.binning, self.representative = M, binning, representative
        self.evidence, self.alpha, self.beta = evidence, alpha, beta
        self.debias, self.W, self.min_cell_n = debias, W, min_cell_n
        self.edges = None
        self.cells: dict = {}          # (segment, bin) -> cell dict
        self.segment_opinion: dict = {}
        self.global_opinion: dict | None = None
        self.n = 0

    # ---- evidence for one cell ------------------------------------------- #
    def _cell(self, p: np.ndarray, y: np.ndarray, lo: float, hi: float) -> dict:
        n = int(len(p))
        rp = float((lo + hi) / 2.0 if self.representative == "mid" else p.mean())
        k = int(y.sum())
        rate = k / n
        dev = rate - rp
        floor = 0.0
        if self.debias == "floor":
            floor = noise_floor(rp, n)
            excess = max(abs(dev) - floor, 0.0)
        elif self.debias == "l2" and n > 1:
            excess = math.sqrt(max(dev * dev - rate * (1.0 - rate) / (n - 1), 0.0))
        else:
            excess = abs(dev)
        if self.evidence == "rate":
            weight = self.alpha if dev > 0 else self.beta
            s = n * weight * excess
            r = max(n - s, 0.0)
        else:  # 'accuracy': y is correctness of the stated top class, p its confidence
            s = n * excess
            r = float(k)
        op = bpq(r, s, self.W)
        return {"n": n, "k": k, "rate": rate, "rp": rp, "dev": dev, "floor": floor,
                "lo": float(lo), "hi": float(hi), "r": r, "s": s, "opinion": op,
                "insufficient_evidence": n < self.min_cell_n}

    # ---- fit ------------------------------------------------------------- #
    def fit(self, scores, outcomes, segments=None):
        p = np.asarray(scores, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        assert p.shape == y.shape and p.ndim == 1
        assert np.all((p >= 0) & (p <= 1)), "scores must be probabilities in [0, 1]"
        self.n = len(p)
        if self.binning == "fixed":
            self.edges = fixed_edges(self.M)
        elif self.binning == "quantile":
            self.edges = quantile_edges(p, self.M)
        else:
            self.edges = isotonic_edges(p, y)
        bins = assign_bins(p, self.edges)
        seg = np.full(len(p), MARGINAL, dtype=object) if segments is None else np.asarray(segments, dtype=object)
        groups = [MARGINAL] + (sorted(set(seg.tolist())) if segments is not None else [])
        self.cells = {}
        for g in groups:
            mask_g = np.ones(len(p), bool) if g == MARGINAL else (seg == g)
            for i in range(len(self.edges) - 1):
                m = mask_g & (bins == i)
                if m.sum() == 0:
                    continue
                self.cells[(g, i)] = self._cell(p[m], y[m], self.edges[i], self.edges[i + 1])
        self.segment_opinion = {
            g: cumulative_fusion([c["opinion"] for (gg, _), c in self.cells.items() if gg == g])
            for g in groups
        }
        self.global_opinion = self.segment_opinion[MARGINAL]
        return self

    # ---- queries --------------------------------------------------------- #
    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1

    def binned_ece(self, segment=MARGINAL) -> float:
        cells = [c for (g, _), c in self.cells.items() if g == segment]
        N = sum(c["n"] for c in cells)
        return float(sum(c["n"] * abs(c["dev"]) for c in cells) / max(N, 1))

    def lookup(self, score: float, segment=None) -> dict:
        """Per-decision opinion: the cell this score (and segment) falls in.
        Falls back to the marginal cell when the segment cell does not exist."""
        i = int(assign_bins(np.array([score], dtype=float), self.edges)[0])
        key = (segment, i) if segment is not None and (segment, i) in self.cells else (MARGINAL, i)
        c = self.cells.get(key)
        if c is None:                      # empty marginal bin (fixed binning): vacuous
            return {"segment": key[0], "bin": i, "lo": float(self.edges[i]), "hi": float(self.edges[i + 1]),
                    "n": 0, "rp": None, "rate": None, "opinion": bpq(0, 0, self.W),
                    "insufficient_evidence": True, "fallback_to_marginal": key[0] == MARGINAL and segment is not None}
        out = {"segment": key[0], "bin": i, "lo": c["lo"], "hi": c["hi"], "n": c["n"],
               "rp": c["rp"], "rate": c["rate"], "opinion": c["opinion"],
               "insufficient_evidence": c["insufficient_evidence"],
               "fallback_to_marginal": key[0] == MARGINAL and segment is not None}
        return out

    def realized(self, scores, outcomes, segments=None) -> dict:
        """Re-evaluate the FITTED partition and representatives on new
        (score, outcome) pairs: the delayed-outcome verdict for calibration.
        Returns the realized binned ECE and opinion on the new data, and the
        gap to what the fitted (reference) opinion claimed."""
        p = np.asarray(scores, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        bins = assign_bins(p, self.edges)
        seg = np.full(len(p), MARGINAL, dtype=object) if segments is None else np.asarray(segments, dtype=object)
        out = {}
        for g in self.segment_opinion:
            mask_g = np.ones(len(p), bool) if g == MARGINAL else (seg == g)
            R = S = 0.0
            N = 0
            ece = 0.0
            for i in range(self.n_bins):
                m = mask_g & (bins == i)
                n = int(m.sum())
                if n == 0:
                    continue
                ref = self.cells.get((g, i)) or self.cells.get((MARGINAL, i))
                rp = ref["rp"] if ref is not None else float(p[m].mean())
                c = self._cell(p[m], y[m], self.edges[i], self.edges[i + 1])
                c["dev"] = c["rate"] - rp                 # deviation from the STATED probability
                R += c["r"]; S += c["s"]; N += n
                ece += n * abs(c["dev"])
            if N == 0:
                continue
            op = bpq(R, S, self.W)
            out[g] = {"n": N, "binned_ece": ece / N, "opinion": op,
                      "claimed_d": self.segment_opinion[g]["d"], "gap_d": op["d"] - self.segment_opinion[g]["d"]}
        return out

    def realized_cells(self, scores, outcomes) -> list[dict]:
        """Per marginal bin: the deviation realised on new data against the
        FITTED representative (stated probability), for cell-level comparison
        of the claimed opinion with what later outcomes show."""
        p = np.asarray(scores, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        bins = assign_bins(p, self.edges)
        out = []
        for i in range(self.n_bins):
            m = bins == i
            ref = self.cells.get((MARGINAL, i))
            if m.sum() == 0 or ref is None:
                continue
            rate = float(y[m].mean())
            out.append({"bin": i, "n": int(m.sum()), "rp": ref["rp"], "rate": rate,
                        "dev": rate - ref["rp"], "claimed_dev": ref["dev"],
                        "claimed_d": ref["opinion"]["d"], "claimed_n": ref["n"]})
        return out

    def report(self, digits: int = 4) -> dict:
        def o(op):
            return {k: round(op[k], digits) for k in ("b", "d", "u", "expected")}
        segs = {g: {"opinion": o(op), "binned_ece": round(self.binned_ece(g), digits),
                    "n": sum(c["n"] for (gg, _), c in self.cells.items() if gg == g),
                    "thin_cells": sum(1 for (gg, _), c in self.cells.items() if gg == g and c["insufficient_evidence"])}
                for g, op in self.segment_opinion.items()}
        return {"config": {"M": self.n_bins, "binning": self.binning, "representative": self.representative,
                           "evidence": self.evidence, "alpha": self.alpha, "beta": self.beta,
                           "debias": self.debias, "W": self.W, "min_cell_n": self.min_cell_n},
                "n": self.n, "global": o(self.global_opinion), "binned_ece": round(self.binned_ece(), digits),
                "segments": segs}

    def certificate_line(self, score: float, segment=None, bracket=None) -> str:
        """The human-readable calibration line stored in the signed record."""
        c = self.lookup(score, segment)
        op = c["opinion"]
        line = f"model said {score:.3f}"
        if bracket is not None:
            line += f"; calibrated bracket [{bracket[0]:.3f}, {bracket[1]:.3f}] (theorem)"
        if c["n"] == 0:
            return line + "; no calibration observations at this score level (signal: vacuous)"
        line += (f"; the stated probability carries disbelief {op['d']:.2f} on {c['n']} calibration "
                 f"observations in this cell (signal)")
        if c["insufficient_evidence"]:
            line += "; insufficient evidence in this cell"
        return line


# --------------------------------------------------------------------------- #
# Population quantities for the theory checks (used by the tests and eval)
# --------------------------------------------------------------------------- #
def closed_form_rate_opinion(n: int, ece_binned: float, W: float = W_DEFAULT) -> dict:
    """Proposition 1: the fused 'rate' opinion (alpha = beta = 1, debias='none')."""
    return {"b": n / (n + W) * (1.0 - ece_binned), "d": n / (n + W) * ece_binned, "u": W / (n + W)}


def hoeffding_bound(cell_sizes, delta: float = 0.05) -> float:
    """Proposition 3: with probability >= 1 - delta, the binned ECE estimate is
    within this distance of its population value for the SAME bins
    (union bound over M bins, Hoeffding within each)."""
    n_i = np.asarray(cell_sizes, dtype=float)
    N, M = n_i.sum(), len(n_i)
    return float(np.sum((n_i / N) * np.sqrt(np.log(2.0 * M / delta) / (2.0 * n_i))))
