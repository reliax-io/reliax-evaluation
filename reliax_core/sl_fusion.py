"""Subjective-logic fusion of the envelope's reliability signals (PaTAS lineage).

Replaces "weighted average with hard caps" as the way to combine evidence about
the proposition "this prediction is correct". Each signal becomes a subjective
opinion omega = (belief, disbelief, uncertainty) with b + d + u = 1 (Josang 2016);
opinions are then merged with AVERAGING FUSION, the operator for dependent
sources observing the same event.

Why this beats a blend, conceptually:
- conflict is preserved: a confident certificate plus a loudly disagreeing
  auditor yields visible disbelief, not a bland midpoint;
- ignorance is first-class: an out-of-distribution input or a martingale alarm
  injects UNCERTAINTY mass rather than pretending to know the answer;
- the (trust, distrust, uncertainty) triple in the envelope is now computed by
  a published calculus, not formatted after the fact.

Source opinions (v1 mapping, documented so it can be criticised):
- confidence:  p = p_top of the base model. Its epistemic uncertainty comes from
               the CERTIFICATE, not from the model itself: a low conformal
               p-value (atypical score), an ambiguous coverage set, or a wide
               Venn-Abers interval all mean the confidence should abstain.
               Crucially, a confidently WRONG model has high p_top and narrow
               intervals; only the conformal typicality exposes it, so it must
               gate this source's certainty.
- auditor:     p = 1 - p_error from the error meta-model; fixed moderate
               uncertainty because the auditor is itself a heuristic source.
- context:     "a typical input should get base-rate performance": p = the
               model's held-out accuracy, moderately confident when the input
               is in-distribution. OOD distance and the exchangeability
               martingale drive this source toward VACUOUS (u -> 1): out of
               scope means "I don't know", not "wrong". Under averaging fusion
               a vacuous source contributes no belief and raises the fused
               uncertainty, which is exactly the intended semantics.

Probability-to-opinion mapping: b = (1-u)*p, d = (1-u)*(1-p), so the projected
probability is E = b + a*u with base rate a = 1/2.

STATUS: a principled composite, still a HEURISTIC in the guarantee taxonomy.
The falsifiable test is the pre-registered shifted-data benchmark: if SL fusion
does not beat the v1 blend there, it goes.
"""
import math

U_FLOOR = 0.02
U_CEIL = 0.98
BASE_RATE = 0.5
AUDITOR_U = 0.30


def _opinion(p: float, u: float) -> dict:
    p = min(max(p, 0.0), 1.0)
    u = min(max(u, U_FLOOR), U_CEIL)
    return {"b": (1 - u) * p, "d": (1 - u) * (1 - p), "u": u}


def averaging_fusion(opinions: list[dict]) -> dict:
    """N-ary averaging fusion (Josang 2016, ch. 12): weight of each source is
    the product of the OTHER sources' uncertainties, so confident sources
    dominate and ignorant ones abstain."""
    n = len(opinions)
    weights = []
    for i in range(n):
        w = 1.0
        for j, o in enumerate(opinions):
            if j != i:
                w *= o["u"]
        weights.append(w)
    den = sum(weights)
    if den <= 0:  # all sources dogmatic; degrade to plain mean
        b = sum(o["b"] for o in opinions) / n
        d = sum(o["d"] for o in opinions) / n
        return {"b": b, "d": d, "u": max(0.0, 1 - b - d)}
    b = sum(o["b"] * w for o, w in zip(opinions, weights)) / den
    d = sum(o["d"] * w for o, w in zip(opinions, weights)) / den
    u = n * math.prod(o["u"] for o in opinions) / den
    s = b + d + u
    return {"b": b / s, "d": d / s, "u": u / s}


def fuse_signals(p_top: float, conformal_p_value: float, set_size: int,
                 va_width: float, auditor_p_error: float,
                 ood_percentile: float, martingale_state: str,
                 base_accuracy: float = 0.8) -> dict:
    """Returns the fused trust triple plus per-source opinions."""
    u_conf = max(
        0.10,                                    # model confidence is never certainty
        va_width / 0.25,                         # calibration cannot pin the PD down
        1.0 - conformal_p_value / 0.5,           # atypical score vs calibration
        0.50 if set_size != 1 else 0.0,          # certificate cannot exclude the other label
    )
    conf = _opinion(p_top, min(u_conf, 0.95))
    auditor = _opinion(1.0 - auditor_p_error, AUDITOR_U)
    u_ctx = 0.25 + 0.75 * (ood_percentile / 100.0) ** 3
    if martingale_state == "ALARM":
        u_ctx = max(u_ctx, 0.97)
    elif martingale_state == "WATCH":
        u_ctx = max(u_ctx, 0.60)
    context = _opinion(base_accuracy, u_ctx)

    fused = averaging_fusion([conf, auditor, context])
    expected = fused["b"] + BASE_RATE * fused["u"]
    return {
        "belief": round(fused["b"], 4),
        "disbelief": round(fused["d"], 4),
        "uncertainty": round(fused["u"], 4),
        "expected": round(expected, 4),
        "score_sl": round(100 * expected),
        "sources": {
            "confidence": {k: round(v, 4) for k, v in conf.items()},
            "auditor": {k: round(v, 4) for k, v in auditor.items()},
            "context": {k: round(v, 4) for k, v in context.items()},
        },
        "method": "sl-averaging-fusion-v1",
        "note": "principled composite, still a heuristic; benchmarked against the v1 blend",
    }
