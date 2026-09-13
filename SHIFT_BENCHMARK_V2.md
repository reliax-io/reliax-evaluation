# Shift benchmark v2 (exploratory): three regimes, one question

`eval/run_shift_bench_v2.py`, results in `results/shift_bench_v2.json`, figure
`results/fig8_bench_v2.png`. Taiwan default only, 3 seeds, 4,000 test rows per
tier, model and every calibrator fitted on the reference population
(up-to-date payers, PAY_0 <= 0). Nothing here is pre-registered; the
confirmatory datasets are untouched. The amendment that turns this into a
binding protocol is `PREREGISTRATION_AMENDMENT.md`.

The question in every tier: which signal should route the 10% least reliable
decisions to REVIEW? Metric: bad approvals (approved, then defaulted) caught
in that 10%, relative to random referral and to the model's own confidence.

## Regime 1: a shift that is happening (mixed populations)

The live book is a mixture: a share lambda of applicants comes from the
delayed-payer population, the rest from the reference population.

| lambda | model AUC | certified REVIEW rate (alpha .05) | martingale | confidence | OOD distance | confidence gated by OOD | regime switch | SL fusion v0.2 |
|---|---|---|---|---|---|---|---|---|
| 0 (calm) | 0.643 | 47% | OK, 3/3 | **1.43** | 0.84 | 1.02 | 1.43 | 1.41 |
| 0.25 | 0.687 | 55% | ALARM, 3/3 | 1.34 | 1.89 | **2.15 ± 0.09** | 1.88 | 0.94 |
| 0.50 | 0.677 | 63% | ALARM, 3/3 | 1.17 | 1.48 | **1.61 ± 0.10** | 1.48 | 0.52 |
| 0.75 | 0.646 | 71% | ALARM, 3/3 | 0.93 | 1.19 | 1.20 | 1.19 | 0.38 |
| 1.00 (severe) | 0.583 | 79% | ALARM, 3/3 | 0.80 | 1.02 | 1.03 | 1.03 | 0.33 |

Numbers are bad-approval capture at 10% referral divided by random
referral's capture, mean over seeds. Against confidence, the gated signal is
1.60x at lambda 0.25 and 1.38x at lambda 0.50; the pre-registered bar (1.5x
random and above confidence) is met at both moderate tiers on this
exploratory data, and not met by anything at 0.75 or above.

Reading. When nothing has moved, confidence is the best ranking and the
OOD distance is worse than random: distance from the calibration data says
nothing about who defaults in a stable population. As soon as a quarter of
the book has shifted, confidence loses half its edge and the OOD distance
becomes the informative signal, because the shifted applicants are both
far from the calibration data and mis-scored. Gating confidence with the
OOD flag (rows above the 95th calibration percentile of distance are
routed first, ordered by distance; everyone else by confidence) takes the
best of both and is the strongest signal in the moderate tiers. Once the
shifted share passes three quarters, no per-decision ranking is worth
having; the martingale has been in ALARM since lambda 0.25 in every seed
and the certified REVIEW rate has risen from 47% to 79% on its own. The
subjective-logic fusion of v0.2 collapses under shift for the reason
already published: its error auditor is fitted on reference data.

## Regime 2: corrupted inputs

Population unchanged; a share rho of incoming rows is corrupted upstream.
The mistake is acting on a wrong input, so the metric is the share of
corrupted rows inside the 10% referred.

| corruption | model AUC | martingale | confidence catches | OOD distance catches | gated catches |
|---|---|---|---|---|---|
| unit error x100 on monetary fields, 5% of rows | 0.632 | ALARM 3/3 | 0.01 | **1.00** | 1.00 |
| unit error, 10% of rows | 0.628 | ALARM 3/3 | 0.02 | **1.00** | 0.84 |
| unit error, 20% of rows | 0.613 | ALARM 3/3 | 0.02 | 0.50 (capacity: 10% referral) | 0.45 |
| one standard deviation of noise on every feature, 10% | 0.630 | ALARM 3/3 | 0.09 | **0.65** | 0.65 |
| half the fields zeroed (failed join), 10% | 0.635 | mixed (WATCH, OK, ALARM) | 0.20 | 0.17 | 0.18 |

Reading. A currency or cents bug on monetary fields is invisible to the
model's confidence (it is confidently wrong on every corrupted row) and
fully visible to the input-space distance: every corrupted row is inside
the referred 10% until the corrupted share exceeds the referral budget.
Random perturbation is caught two thirds of the time. Missing fields
zeroed by a failed join are NOT caught by distance, because zeros are
inside the range of every feature; that failure needs a schema and
missingness check on the input, which is a data-assessment rule rather
than a distance, and is the honest gap this benchmark exposes.

## What the benchmark claims, and does not

- Claimed: there is a regime, a shift in progress or a corrupted feed,
  where the envelope's OOD-gated routing catches 1.4x to 1.6x the bad
  approvals that the model's own confidence catches at the same referral
  rate, and where the tripwire fires in every seed. Both are plausible
  production events, and both are cases the product was built for.
- Not claimed: that any signal ranks mistakes better than confidence in a
  stable population, or that anything ranks once a shift is complete. In
  the first case the certified threshold is the product; in the second,
  the tripwire and recalibration are.
- Not yet claimed at all: any of this on the confirmatory datasets. The
  tiers, the signal (gated confidence, parameters frozen) and the bar go
  into the pre-registration amendment before Give Me Some Credit or Home
  Credit are opened.
