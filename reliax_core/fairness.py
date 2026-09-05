"""Fairness layer: Mondrian (group-conditional) conformal + live bias monitoring.

Two distinct claims, kept distinct:

1. GUARANTEED - Mondrian conformal. Marginal coverage P(Y in C(X)) >= 1-alpha can
   hide undercoverage on a segment (young applicants, one region). Calibrating q_hat
   PER GROUP gives group-conditional coverage P(Y in C(X) | G=g) >= 1-alpha for every
   audited segment g - the certificate holds for each group, not just on average.
   Segment attributes are AUDIT-ONLY: the model never sees them.

2. MONITORED (a signal, not a proof) - live adverse-impact tracking. Rolling ALLOW
   rate per segment and the four-fifths ratio min_rate/max_rate; < 0.80 raises a
   DISPARATE_IMPACT_WATCH on the monitor (US EEOC 80% rule as a screening heuristic;
   EU AI Act Art. 10 bias-monitoring angle). Outcome labels arrive too late for live
   per-group calibration checks, so the startup audit reports those on held-out data.
"""
from collections import deque

import numpy as np

from .conformal import ConformalCalibrator

IMPACT_WINDOW = 300
MIN_PER_GROUP = 15
FOUR_FIFTHS = 0.80


class MondrianConformal:
    """One split-conformal calibrator per segment value (e.g. per age band)."""

    def __init__(self, calib_probs: np.ndarray, calib_labels: np.ndarray,
                 calib_groups: np.ndarray, group_names: list[str]):
        self.group_names = group_names
        self.calibrators = {}
        for g, name in enumerate(group_names):
            mask = calib_groups == g
            self.calibrators[name] = ConformalCalibrator(calib_probs[mask], calib_labels[mask])

    def prediction_set(self, probs: np.ndarray, alpha: float, group: str) -> list[int]:
        return self.calibrators[group].prediction_set(probs, alpha)

    def qhat(self, alpha: float, group: str) -> float:
        return self.calibrators[group].qhat(alpha)

    def calibration_n(self, group: str) -> int:
        return self.calibrators[group].n


def coverage_audit(marginal: ConformalCalibrator, mondrian: MondrianConformal,
                   probs_test: np.ndarray, y_test: np.ndarray,
                   groups_test: np.ndarray, alpha: float) -> dict:
    """Held-out audit: does marginal coverage hide per-segment undercoverage,
    and does Mondrian repair it? Every number computed, none scripted."""
    rows = []
    q_marg = marginal.qhat(alpha)
    for g, name in enumerate(mondrian.group_names):
        m = groups_test == g
        if m.sum() == 0:
            continue
        p, y = probs_test[m], y_test[m]
        hits_marg = 1.0 - p[np.arange(len(y)), y] <= q_marg
        q_g = mondrian.qhat(alpha, name)
        hits_mond = 1.0 - p[np.arange(len(y)), y] <= q_g
        rows.append({
            "segment": name,
            "n": int(m.sum()),
            "marginal_coverage": round(float(hits_marg.mean()), 4),
            "mondrian_coverage": round(float(hits_mond.mean()), 4),
            "marginal_qhat": round(q_marg, 4),
            "mondrian_qhat": round(q_g, 4),
        })
    worst = min(rows, key=lambda r: r["marginal_coverage"])
    return {
        "alpha": alpha,
        "target": round(1 - alpha, 3),
        "segments": rows,
        "worst_marginal_segment": worst["segment"],
        "worst_marginal_coverage": worst["marginal_coverage"],
    }


class ImpactMonitor:
    """Rolling adverse-impact tracking over live routing decisions."""

    def __init__(self, group_names: list[str]):
        self.group_names = group_names
        self.windows = {g: deque(maxlen=IMPACT_WINDOW) for g in group_names}

    def observe(self, group: str, routing: str):
        if group in self.windows:
            self.windows[group].append(1 if routing == "ALLOW" else 0)

    def report(self) -> dict:
        rows, rates = [], {}
        for g in self.group_names:
            w = self.windows[g]
            rate = round(sum(w) / len(w), 4) if len(w) >= MIN_PER_GROUP else None
            rows.append({"segment": g, "n": len(w), "allow_rate": rate})
            if rate is not None:
                rates[g] = rate
        ratio = None
        flag = False
        if len(rates) >= 2 and max(rates.values()) > 0:
            ratio = round(min(rates.values()) / max(rates.values()), 4)
            flag = ratio < FOUR_FIFTHS
        return {
            "segments": rows,
            "adverse_impact_ratio": ratio,
            "four_fifths_threshold": FOUR_FIFTHS,
            "disparate_impact_watch": flag,
            "note": "screening heuristic on rolling ALLOW rates - not a legal determination",
        }
