"""Baselines and metrics for the evaluation.

- Temperature scaling (Guo et al., 2017): the standard post-hoc calibration
  baseline. Note that T-scaling is monotone in the logit, so it does not change
  the RANKING of max-softmax confidence; it matters for the calibration
  comparison (ECE / Brier), not for selective-prediction ordering.
- ECE: 15-bin expected calibration error on p(default).
- Error capture @ referral rate r: flag the r fraction of test points ranked
  least reliable by a signal; report the fraction of all model errors captured.
- AURC: area under the risk-coverage curve (lower is better).
"""
import numpy as np


def fit_temperature(logits_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """1-D grid search minimising NLL of sigmoid(logit / T) on the calibration split."""
    ts = np.logspace(-1, 1, 400)
    best_t, best_nll = 1.0, np.inf
    for t in ts:
        p = 1.0 / (1.0 + np.exp(-logits_cal / t))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        nll = -np.mean(y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, t
    return float(best_t)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def error_capture(reliability: np.ndarray, wrong: np.ndarray, rate: float) -> float:
    """Fraction of all errors inside the `rate` fraction ranked least reliable."""
    k = max(1, int(round(rate * len(reliability))))
    flagged = np.argsort(reliability)[:k]         # ascending: least reliable first
    return float(wrong[flagged].sum() / max(wrong.sum(), 1))


def default_capture(reliability: np.ndarray, approved_default: np.ndarray, rate: float) -> float:
    """Fraction of would-be bad approvals (model says repay, truth is default)
    inside the `rate` fraction ranked least reliable."""
    return error_capture(reliability, approved_default, rate)


def risk_coverage(reliability: np.ndarray, wrong: np.ndarray, n_points: int = 50):
    """Risk (error rate among kept) vs coverage (fraction kept, most reliable first)."""
    order = np.argsort(reliability)[::-1]         # most reliable first
    w = wrong[order]
    covs, risks = [], []
    n = len(w)
    for c in np.linspace(0.05, 1.0, n_points):
        k = max(1, int(round(c * n)))
        covs.append(k / n)
        risks.append(float(w[:k].mean()))
    return np.array(covs), np.array(risks)


def aurc(reliability: np.ndarray, wrong: np.ndarray) -> float:
    covs, risks = risk_coverage(reliability, wrong, n_points=200)
    return float(np.trapezoid(risks, covs))
