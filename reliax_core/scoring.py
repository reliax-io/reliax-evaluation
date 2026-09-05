"""Reliability score v3: 0-100 ordinal composite.

Calibrated heuristic, NOT a probability and NOT a theorem. v3 blends six signals:

    0.25  confidence margin           (p_top - p_second)
    0.20  conformal p-value           (typicality of the predicted class)
    0.15  coverage-set cardinality    (singleton good, ambiguous bad)
    0.15  OOD percentile              (distance to the calibration cloud)
    0.10  Venn-Abers interval width   (narrow = calibration pins the PD down)
    0.15  auditor agreement           (1 - p_error from the error meta-model)

Hard caps keep the score honest: an OOD flag caps at 40, an empty coverage set at
35, an auditor flag at 55, a martingale ALARM at 30.
"""

SET_CARDINALITY = {0: 0.0, 1: 1.0, 2: 0.45}  # >=3 -> 0.15


def reliability_score(
    margin: float,
    p_value: float,
    set_size: int,
    ood_percentile: float,
    ood_flag: bool,
    va_width: float,
    auditor_p_error: float,
    auditor_flag: bool,
    martingale_alarm: bool,
) -> int:
    set_component = SET_CARDINALITY.get(set_size, 0.15)
    pv_component = min(1.0, p_value / 0.5)          # p >= 0.5 is fully typical
    ood_component = 1.0 - ood_percentile / 100.0
    va_component = max(0.0, 1.0 - va_width / 0.25)  # width >= 0.25 scores zero
    auditor_component = 1.0 - min(1.0, auditor_p_error / 0.5)
    raw = (
        0.25 * margin
        + 0.20 * pv_component
        + 0.15 * set_component
        + 0.15 * ood_component
        + 0.10 * va_component
        + 0.15 * auditor_component
    )
    score = round(100 * raw)
    if ood_flag:
        score = min(score, 40)
    if set_size == 0:
        score = min(score, 35)
    if auditor_flag:
        score = min(score, 55)
    if martingale_alarm:
        score = min(score, 30)
    return max(0, min(100, score))
