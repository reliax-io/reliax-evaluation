# Verbatim copy of reliax_core/cutoff.py at reliax-core commit 90303ee (0.4.0, with three_zones). The
# evaluation pins reliax-core v0.2.0 for every other runner; this copy lets eval/run_cutoff.py and
# eval/run_zones.py use the 0.4.0 procedures without moving that pin, and they check it against the
# installed package whenever that package has the module.
"""The lender's cut-off: its break-even point, a guaranteed cut-off, and the bracket against it.

A lender approves while the expected margin on a good customer covers the
expected loss on a bad one, PD x L < (1 - PD) x M, so the most it can accept is

    PD* = M / (M + L)          (break_even_pd)

The cut-off stays the lender's economic decision. This module adds two things
to it.

``guaranteed_cutoff`` answers "approve automatically up to which score, so
that the default rate among the approved stays at most a target?" with a
finite-sample guarantee. Approve when the score (the predicted PD) is at most
t. For thresholds t_1 < t_2 < ... (strictest first) it tests, in order, the
hypothesis "the default rate among approvals at t_j exceeds the target" with
the exact binomial p-value P(Binomial(n_j, target) <= d_j), where n_j
calibration cases are approved at t_j and d_j of them defaulted, and walks on
while p_j <= delta. The cut-off is the loosest threshold reached. This is
Learn-then-Test with fixed-sequence testing (Angelopoulos, Bates, Candes,
Jordan and Lei, 2021): with probability at least 1 - delta over the
calibration set, the default rate among future approvals at that cut-off is at
most the target, for future cases exchangeable with the calibration data.
Fixed-sequence testing needs no monotonicity of the default rate in t.
The sequence starts at the first candidate that approves enough cases to
reject at all, max(log(delta) / log(1 - target), 5% of the calibration set),
because the walk ends at the first test that fails; that start depends on the
scores only, never on the outcomes, so the guarantee is unchanged.

``three_zones`` guarantees both sides of the lender's own cut-off. The
calibration scores are cut into equal-count slices, 5% of the cases each by
default (score quantiles: the cut depends on the scores only). From the lowest
predicted PD up, each slice is tested for "its default rate exceeds PD*" with
the exact binomial p-value P(Binomial(n, PD*) <= d), and approved while
p <= delta / 2; from the highest PD down, each slice is tested for "its
default rate is below PD*" with P(Binomial(n, PD*) >= d), and declined while
p <= delta / 2. What neither walk reaches is reviewed by a person: the band
around the cut-off where the data cannot tell which side a case is on. With
probability at least 1 - delta, every approved slice has a default rate at
most PD* (an approval there earns the margin on average) and every declined
slice at least PD* (a decline there avoids an expected loss), for future cases
exchangeable with the calibration data (two fixed sequences at delta / 2
each). Slices are tested one by one, never pooled with safer neighbours: a
pooled window would let its riskiest cases sit above PD*. The guarantee on declines needs
outcomes for declined applicants, which a lender only has from past periods,
a random approval sample or bureau data.

``bracket_route`` compares the Venn-Abers bracket [p0, p1] with PD*: approve
when even the upper end is below it, decline when even the lower end is above
it, and send the case to a person only when the bracket straddles the
lender's cut-off.

Class of output: guaranteed_cutoff carries a guarantee (above); the others are
exact arithmetic. The guarantee holds where the calibration outcomes were
observed: if they exist only for applicants approved under an older, stricter
cut-off, it does not reach below that cut-off (the selective-labels problem).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta, binom

APPROVE, DECLINE, REVIEW = "approve", "decline", "review"


def break_even_pd(margin: float, loss: float) -> float:
    """PD* = M / (M + L): the highest probability of default at which lending still pays."""
    if margin <= 0 or loss <= 0:
        raise ValueError("margin and loss must be positive")
    return float(margin) / (float(margin) + float(loss))


def upper_bound(defaults: int, n: int, delta: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound on a default rate, at confidence 1 - delta."""
    if n <= 0:
        return 1.0
    if defaults >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, defaults + 1, n - defaults))


def lower_bound(defaults: int, n: int, delta: float = 0.05) -> float:
    """One-sided Clopper-Pearson lower bound on a default rate, at confidence 1 - delta."""
    if n <= 0 or defaults <= 0:
        return 0.0
    return float(beta.ppf(delta, defaults, n - defaults + 1))


def default_grid(scores: np.ndarray, n: int = 41) -> np.ndarray:
    """Candidate cut-offs, strictest first: the calibration scores' quantiles every 2.5%, 0.5% to 99.5%."""
    return np.unique(np.quantile(np.asarray(scores, dtype=float), np.linspace(0.005, 0.995, n)))


def guaranteed_cutoff(scores, outcomes, target: float, delta: float = 0.05, grid=None) -> dict:
    """The loosest cut-off whose default rate among approvals is at most ``target``, with probability
    at least 1 - ``delta``. ``scores`` is the predicted PD (higher is riskier), ``outcomes`` 1 for a
    default. Returns the cut-off (None when even the strictest candidate cannot be shown to meet the
    target) with what the calibration set shows at it."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if s.shape != y.shape or s.ndim != 1 or not len(s):
        raise ValueError("scores and outcomes must be 1-d arrays of the same, non-zero length")
    if not 0.0 < target < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("target and delta must lie in (0, 1)")
    grid = default_grid(s) if grid is None else np.sort(np.asarray(grid, dtype=float))
    order = np.argsort(s, kind="stable")
    s_sorted, y_cum = s[order], np.cumsum(y[order])
    n_start = max(math.ceil(math.log(delta) / math.log(1.0 - target)), math.ceil(0.05 * len(s)))
    grid = grid[np.searchsorted(s_sorted, grid, side="right") >= n_start]
    chosen, tested = None, 0
    for t in grid:
        n = int(np.searchsorted(s_sorted, t, side="right"))
        d = int(y_cum[n - 1]) if n else 0
        tested += 1
        p = float(binom.cdf(d, n, target)) if n else 1.0
        if p > delta:
            break
        chosen = (float(t), n, d)
    out = {"target": float(target), "delta": float(delta), "method": "learn-then-test, fixed sequence, binomial",
           "n_calibration": int(len(s)), "start_n": int(n_start), "candidates_tested": tested, "cutoff": None}
    if chosen is not None:
        t, n, d = chosen
        out.update({"cutoff": t, "approved_share": n / len(s), "n_approved": n, "defaults": d,
                    "default_rate": d / n, "upper_bound": upper_bound(d, n, delta)})
    return out


def bracket_route(p0: float, p1: float, pd_star: float) -> str:
    """Approve below the lender's cut-off, decline above it, review only when the bracket straddles it."""
    if not (0.0 <= p0 <= p1 <= 1.0) or math.isnan(pd_star):
        raise ValueError("need 0 <= p0 <= p1 <= 1 and a cut-off")
    if p1 < pd_star:
        return APPROVE
    if p0 > pd_star:
        return DECLINE
    return REVIEW


def three_zones(scores, outcomes, pd_star: float, delta: float = 0.05, slices: int = 20) -> dict:
    """Approve, review and decline zones around the lender's cut-off ``pd_star``, both sides guaranteed
    (see the module docstring). ``scores`` is the predicted PD, ``outcomes`` 1 for a default; the
    calibration scores are cut into ``slices`` equal-count slices. A new case is approved when its score
    is at most ``approve_to``, declined when above ``decline_from``, reviewed otherwise (``zone_of``)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if s.shape != y.shape or s.ndim != 1 or len(s) < 2 * slices:
        raise ValueError("scores and outcomes must be 1-d arrays of the same length, at least 2 per slice")
    if not 0.0 < pd_star < 1.0 or not 0.0 < delta < 1.0 or slices < 2:
        raise ValueError("pd_star and delta must lie in (0, 1), and slices be at least 2")
    edges = np.unique(np.quantile(s, np.arange(1, slices) / slices))
    k = np.searchsorted(edges, s, side="left")                     # slice index: (edge[k-1], edge[k]]
    S = len(edges) + 1
    n = np.bincount(k, minlength=S)
    d = np.bincount(k, weights=y, minlength=S).astype(int)
    half = delta / 2.0
    approve = 0                                                    # walk up while a slice is shown to earn the margin
    while approve < S and n[approve] and binom.cdf(d[approve], n[approve], pd_star) <= half:
        approve += 1
    decline = S                                                    # walk down while a slice is shown to lose money
    while decline > approve and n[decline - 1] and binom.sf(d[decline - 1] - 1, n[decline - 1], pd_star) <= half:
        decline -= 1
    lo = np.concatenate([[-np.inf], edges])
    hi = np.concatenate([edges, [np.inf]])
    table = [{"from": None if not np.isfinite(lo[i]) else float(lo[i]), "to": None if not np.isfinite(hi[i]) else float(hi[i]),
              "n": int(n[i]), "defaults": int(d[i]), "default_rate": float(d[i] / n[i]) if n[i] else None,
              "zone": APPROVE if i < approve else DECLINE if i >= decline else REVIEW} for i in range(S)]
    a, r = k < approve, k >= decline
    return {"pd_star": float(pd_star), "delta": float(delta), "slices": int(S),
            "method": "two fixed sequences of binomial tests on equal-count score slices, delta / 2 each",
            "approve_to": float(hi[approve - 1]) if approve else None,
            "decline_from": float(lo[decline]) if decline < S else None,
            "shares": {APPROVE: float(a.mean()), REVIEW: float((~a & ~r).mean()), DECLINE: float(r.mean())},
            "approved": {"n": int(a.sum()), "default_rate": float(y[a].mean()) if a.any() else None,
                         "upper_bound": upper_bound(int(y[a].sum()), int(a.sum()), half)},
            "declined": {"n": int(r.sum()), "default_rate": float(y[r].mean()) if r.any() else None,
                         "lower_bound": lower_bound(int(y[r].sum()), int(r.sum()), half)},
            "slices_table": table}


def zone_of(score: float, zones: dict) -> str:
    """The zone a new case falls in, from ``three_zones``."""
    a, dcl = zones["approve_to"], zones["decline_from"]
    if a is not None and score <= a:
        return APPROVE
    if dcl is not None and score > dcl:
        return DECLINE
    return REVIEW
