"""Conformal test martingale - an anytime-valid, label-free exchangeability test.

Upgrades demo-v2's heuristic rolling-OOD tripwire into a theorem-backed monitor.

For each incoming input x, compute a label-free nonconformity score a(x) (here: the
kNN distance to the calibration cloud) and its SMOOTHED conformal p-value against
the calibration scores a_1..a_n:

    p = ( #{a_i > a} + U * (#{a_i = a} + 1) ) / (n + 1),   U ~ Uniform(0,1)

Under exchangeability (calibration and production drawn from the same source, in any
order), these p-values are i.i.d. Uniform(0,1) - regardless of the data distribution.

We then bet against uniformity with a MIXTURE POWER MARTINGALE:

    M_k = (1/|E|) * sum_{eps in E}  prod_{i<=k}  eps * p_i^(eps-1)

Each factor eps*p^(eps-1) has expectation 1 under uniform p, so M is a nonnegative
martingale with M_0 = 1. Ville's inequality gives the anytime-valid guarantee:

    P( sup_k M_k >= c ) <= 1/c        under exchangeability.

So M >= 100 rejects exchangeability at level 0.01 no matter when we look - the
mathematically honest version of "the coverage certificate's assumptions no longer
hold". States: OK (M < 20) · WATCH (20 <= M < 100) · ALARM (M >= 100, envelope
validity trips). A recalibration event resets the wealth to 1.
"""
import math

import numpy as np

EPS_GRID = np.arange(0.05, 1.0, 0.05)  # mixture over betting exponents
WATCH_THRESHOLD = 20.0                  # evidence at level 0.05
ALARM_THRESHOLD = 100.0                 # evidence at level 0.01 -> valid=False
HISTORY = 400


class ConformalMartingale:
    def __init__(self, calib_scores: np.ndarray, seed: int = 7):
        self.calib = np.sort(np.asarray(calib_scores, dtype=float))
        self.n = len(self.calib)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.log_wealth_eps = np.zeros(len(EPS_GRID))  # log of each power martingale
        self.steps = 0
        self.history: list[dict] = [{"step": 0, "log10_m": 0.0, "p": None}]
        self.max_log10 = 0.0

    def p_value(self, score: float) -> float:
        n_gt = self.n - int(np.searchsorted(self.calib, score, side="right"))
        n_eq = int(np.searchsorted(self.calib, score, side="right")
                   - np.searchsorted(self.calib, score, side="left"))
        u = float(self.rng.random())
        return (n_gt + u * (n_eq + 1)) / (self.n + 1)

    def update(self, score: float) -> dict:
        p = self.p_value(score)
        p = min(max(p, 1e-12), 1.0)
        self.log_wealth_eps += np.log(EPS_GRID) + (EPS_GRID - 1.0) * math.log(p)
        self.steps += 1
        state = self.state()
        self.max_log10 = max(self.max_log10, state["log10_martingale"])
        self.history.append({"step": self.steps,
                             "log10_m": state["log10_martingale"], "p": round(p, 4)})
        if len(self.history) > HISTORY:
            self.history = self.history[-HISTORY:]
        return state

    def _wealth(self) -> float:
        # mixture = mean of exp(log_wealth); logsumexp for numerical safety
        m = float(np.max(self.log_wealth_eps))
        return math.exp(m) * float(np.mean(np.exp(self.log_wealth_eps - m)))

    def state(self) -> dict:
        w = self._wealth() if self.steps else 1.0
        label = "ALARM" if w >= ALARM_THRESHOLD else "WATCH" if w >= WATCH_THRESHOLD else "OK"
        return {
            "martingale": round(w, 4) if w < 1e6 else float(f"{w:.3e}"),
            "log10_martingale": round(math.log10(max(w, 1e-12)), 3),
            "state": label,
            "steps": self.steps,
            "thresholds": {"watch": WATCH_THRESHOLD, "alarm": ALARM_THRESHOLD},
            "ville_bound": "P(ever >= 100) <= 0.01 under exchangeability",
        }
