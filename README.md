# Reliax whitepaper evaluation

Reproducibility package for the working paper **"Per-Decision Reliability
Envelopes for Credit-Risk Models: An Evaluation on Real Data"** (v0.2,
September 2026). The paper is in [PAPER.md](PAPER.md); a formatted version is
at [reliax.io](https://reliax.io).

Every number in the paper is produced by `eval/run_eval.py` and stored in
`results/results.json`. Nothing is hand-typed without a source there.

The shifted-data regime is measured separately by `eval/run_shift_bench.py`,
which writes `results/shift_bench.json`. It is a negative result and it is
reported here in full.

## Headline results

| Claim | Measured |
|---|---|
| Conformal coverage at a 0.95 target | 0.9515 +/- 0.0040 (Taiwan), 0.9532 +/- 0.0167 (German) |
| Drift false alarms (5 x 600 i.i.d. real applications) | 0 / 5; real subpopulation shift caught in 4/5 streams |
| Calibration ECE on a miscalibrated model | 0.065 (Venn-Abers) vs 0.083 (T-scaled) vs 0.186 (raw) |
| Best selective-prediction AURC, Taiwan | 0.0955 (subjective-logic fusion) |
| Full envelope latency | 8.0 ms median, 17.5 ms p95 |
| **Bad-approval capture under severe shift** | **the envelope loses to random referral; see the pilot below** |

Honest negatives are in the paper too: the SL fusion trails plain confidence on
the small German dataset, and it trails plain confidence in-distribution on
Taiwan as well.

## Shifted-data pilot: a negative result

`eval/run_shift_bench.py` asks one question. Flag the 10% of predictions ranked
least reliable; how many bad approvals (approved, then defaulted) does that 10%
catch, against the model's own confidence, against random referral, and against
the ceiling a perfect ranking would reach?

Train and calibrate on a reference subpopulation, evaluate on a disjoint one.
No synthetic noise: every shift is a real split of real applicants. Four
constructions, five seeds, bootstrap CIs. Only the payment-delay split is
severe enough to matter (model AUC 0.756 -> 0.584); the other three cost the
model under 3 AUC points and plain confidence still works there.

On that severe split, bad-approval capture at a 10% referral rate:

| Signal | Capture | vs confidence | vs random |
|---|---|---|---|
| oracle (perfect ranking) | 0.258 | 3.32x | 2.68x |
| kNN OOD distance | 0.102 | 1.32x (CI 1.13 to 1.54) | 1.06x |
| **random referral** | 0.096 | 1.27x | 1.00x |
| **model confidence** | 0.078 | 1.00x | 0.81x |
| composite | 0.055 | 0.72x | 0.57x |
| SL fusion | 0.023 | 0.32x | 0.24x |

Two things follow, and neither flatters the method.

1. Under severe shift the model's own confidence falls below random referral.
   Referring at random catches 1.27x what confidence catches. Any threshold
   expressed as a multiple of confidence is therefore measuring against an
   anti-informative denominator, and the 2x bar stated in earlier material has
   been withdrawn for that reason.
2. The composite and SL fusion scores are worse than random here. The error
   auditor is fitted on reference data and collapses precisely under shift,
   which drags the fusion down with it. kNN distance is the only component that
   adds information, and the fusion destroys it.

What does survive shift is detection rather than per-decision ranking:
conformal coverage degrades measurably and in the right direction, from 0.9519
i.i.d. to 0.9031 to 0.9113 across the three shifted splits against a 0.95
target, and the test martingale catches real subpopulation shift in 4/5 streams
with 0/5 false alarms.

Give Me Some Credit and Home Credit are not run here and remain frozen.

## Calibration-trust stage (FUSION 2025 method, credit adaptation)

`reliax_core/calibration_trust.py` turns the calibration behaviour of the
scorer into subjective-logic opinions per (segment x score-bin) cell and a
fused global opinion, with the evidence collection revised for a credit PD
model (rate evidence, mean representatives, quantile or isotonic bins,
debiasing, segment cells, delayed-outcome re-evaluation). The theory is in
[CALIBRATION_TRUST.md](CALIBRATION_TRUST.md): the global opinion is a closed
form of the binned ECE and the sample size (Proposition 1); as the number of
bins grows it converges to E|Y - q|, a sharpness-penalised quantity, not to
calibration (Proposition 2); a Hoeffding finite-sample envelope on the
disbelief (Proposition 3, holds in 500/500 trials); consistency when M grows
like N^(1/3) (Proposition 4); and the small-sample bias constants of the two
corrections (Proposition 5). Experiments: `eval/run_calibration_trust.py`
(about 3 minutes) writes `results/calibration_trust.json`;
`eval/make_ct_figures.py` draws `results/fig5_ct_msweep.png` and
`results/fig6_ct_cells.png`. `eval/run_calibration_credit.py` is the held-out
credit-risk evaluation (claimed vs realised disbelief per scorer, clustering
and segment; `results/calibration_credit.json`, `results/fig7_ct_credit.png`).
`eval/run_calibration_decisions.py` and `eval/run_routing_decisions.py` ask the
decision-level question (bad approvals avoided per 1,000 at what referral rate,
against random and confidence referral); both are negative for per-decision
mistake avoidance and are reported in full in section 9c of the note.
Tests: `tests/test_calibration_trust.py`.

Honest results included: on Taiwan the model is already calibrated (d about
0.005); on German the opinion ranks raw, temperature-scaled and Venn-Abers
scorers the way ECE does but every cell is thin and flagged; as a
per-decision ranking signal the cell belief does not beat plain confidence;
under the payment-delay shift the reference-fitted opinion claims d = 0.061
while the realised value is 0.225, which is what the delayed-outcome verdict
is for.

## Shift benchmark v2 (exploratory) and the pre-registration amendment

`eval/run_shift_bench_v2.py` asks the routing question in three regimes a
lender can actually meet: a shift in progress (mixed populations), corrupted
inputs (unit errors, missing fields, noise) and the severe split. Confidence
wins when nothing has moved; confidence gated by the OOD distance catches
2.15x random and 1.60x confidence when a quarter of the book has shifted,
1.61x and 1.38x at half; nothing ranks once the shift is complete, where the
tripwire fires in every seed. A currency bug on monetary fields is caught
100% by distance and 1% by confidence; zeroed fields are caught by neither.
Write-up: [SHIFT_BENCHMARK_V2.md](SHIFT_BENCHMARK_V2.md). The tiers, signal
and bar go into [PREREGISTRATION_AMENDMENT.md](PREREGISTRATION_AMENDMENT.md)
(draft) before the confirmatory datasets are opened.

## Reproduce

```bash
git clone https://github.com/Ouatt-Isma/reliax-evaluation.git && cd reliax-evaluation
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python eval/run_eval.py         # ~2 minutes; rewrites results/results.json
.venv/bin/python eval/make_figures.py     # rewrites results/fig_*.png
.venv/bin/python eval/run_shift_bench.py  # ~10 minutes; rewrites results/shift_bench.json
```

Splits, seeds and every threshold are fixed in `eval/run_eval.py`.

## Layout

- `reliax_core/` - the envelope's method components (conformal, Mondrian,
  Venn-Abers, test martingale, kNN OOD, PSI, error auditor, SL fusion).
  Methods only: the Reliax product layer is not part of this release.
- `eval/` - dataset loaders, baselines/metrics, experiment runners, figures.
- `data/` - tier 1: the two real UCI datasets, verbatim, with `PROVENANCE.md` (CC BY 4.0, redistributed).
- `data_fetch/` - tier 2: fetch scripts for public sources that are not redistributed here (TableShift, HMDA, ACS).
- `data_gated/` - tier 3: instructions only for sources whose terms prohibit redistribution (Fannie Mae, Freddie Mac, Kaggle).
- `docs/` - the evaluation expansion plan; the binding protocol is `PREREGISTRATION_AMENDMENT.md`.
- `results/` - `results.json`, `shift_bench.json` and the four paper figures.

## License

Code: Apache-2.0, public for every experiment. Data is redistributed where
the licence permits (UCI, CC BY 4.0, see `data/PROVENANCE.md`), fetched by
script where the source is public (TableShift, HMDA, ACS; `data_fetch/`), and
instructions-only where the provider's terms prohibit redistribution (Fannie
Mae, Freddie Mac, Kaggle; `data_gated/`).
