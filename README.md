# Reliax whitepaper evaluation

The reproducibility package for the working paper **"Per-Decision Reliability
Envelopes for Credit-Risk Models: An Evaluation on Real Data"** (September
2026). The paper is in [PAPER.md](PAPER.md); a formatted version is at
[reliax.io](https://reliax.io). Every number in the paper is produced by a
runner script in this repository and stored under `results/`; nothing is
hand-typed without a source there. The negative results are reported here in
full, next to the positive ones. Terms used here are defined in the
[glossary](https://github.com/reliax-io#terms).

The method code is [reliax-core](https://github.com/reliax-io/reliax-core),
pinned at a tagged release in `requirements.txt`, so a clean clone installs
the same code that produced the results.

## Reproduce

```bash
git clone https://github.com/reliax-io/reliax-evaluation.git && cd reliax-evaluation
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python eval/run_eval.py         # ~2 minutes; rewrites results/results.json
.venv/bin/python eval/make_figures.py     # rewrites results/fig_*.png
.venv/bin/python eval/run_shift_bench.py  # ~10 minutes; rewrites results/shift_bench.json
```

Splits, seeds and every threshold are fixed in `eval/run_eval.py`. The
further experiments below each name their own runner.

## Headline results

Two public credit datasets: Default of Credit Card Clients (Taiwan, 30,000
applicants) and Statlog German Credit (1,000 applicants), both from the UCI
repository.

| Claim | Measured |
|---|---|
| Conformal coverage at a 0.95 target | 0.9515 ± 0.0040 (Taiwan), 0.9532 ± 0.0167 (German) |
| Drift false alarms, 5 streams of 600 unshifted applicants | 0 of 5; a real subpopulation shift caught in 4 of 5 streams |
| Calibration error on a miscalibrated model (ECE) | 0.065 (Venn-Abers) vs 0.083 (temperature scaling) vs 0.186 (raw) |
| Best selective-prediction risk-coverage trade-off (AURC), Taiwan | 0.0955 (subjective-logic fusion) |
| Full envelope latency | 8.0 ms median, 17.5 ms p95 |
| **Bad approvals caught under severe shift** | **worse than referring at random; see below** |

Two further negatives are in the paper: the subjective-logic fusion trails
plain confidence on the small German dataset, and it trails plain confidence
on unshifted Taiwan data as well.

## The negative result: a severe shift

`eval/run_shift_bench.py` asks one question. Refer the 10% of decisions ranked
least reliable; how many bad approvals (approved, then defaulted) does that 10%
contain, against the model's own confidence, against random referral, and
against the ceiling a perfect ranking would reach?

Train and calibrate on a reference subpopulation, evaluate on a disjoint one.
No synthetic noise: every shift is a real split of real applicants. Four
constructions, five seeds, bootstrap confidence intervals. Only the
payment-delay split is severe enough to matter (model AUC 0.756 to 0.584); the
other three cost the model under 3 AUC points and plain confidence still works
there.

On that severe split, the share of bad approvals caught at a 10% referral rate:

| Signal | Caught | vs confidence | vs random |
|---|---|---|---|
| oracle (perfect ranking) | 0.258 | 3.32x | 2.68x |
| kNN OOD distance | 0.102 | 1.32x (CI 1.13 to 1.54) | 1.06x |
| **random referral** | 0.096 | 1.27x | 1.00x |
| **model confidence** | 0.078 | 1.00x | 0.81x |
| composite | 0.055 | 0.72x | 0.57x |
| SL fusion | 0.023 | 0.32x | 0.24x |

Two things follow.

1. Under severe shift the model's own confidence falls below random referral.
   Any threshold expressed as a multiple of confidence is therefore measuring
   against an anti-informative denominator. The pre-registered target of twice
   the wrong-approval capture of model confidence (section 4.3 of the paper)
   has been withdrawn for that reason.
2. The composite and SL fusion scores are worse than random here. The error
   auditor is fitted on reference data and collapses precisely under shift,
   which drags the fusion down with it. kNN distance is the only component that
   adds information, and the fusion destroys it.

What does survive shift is detection rather than per-decision ranking:
conformal coverage degrades measurably and in the right direction, from 0.9519
unshifted to between 0.9031 and 0.9113 across the three shifted splits against
a 0.95 target, and the test martingale catches real subpopulation shift in 4 of
5 streams with 0 of 5 false alarms.

Give Me Some Credit and Home Credit are not run here and remain frozen for
the confirmatory study.

## Further experiments

Each has its own write-up and runner; the README keeps one paragraph per
experiment.

**Calibration trust.** `reliax_core.calibration_trust` turns the calibration
behaviour of the scorer into a subjective-logic opinion per segment and score
bin, with five finite-sample propositions in
[CALIBRATION_TRUST.md](CALIBRATION_TRUST.md). Runners:
`eval/run_calibration_trust.py` (about 3 minutes), `eval/run_calibration_credit.py`
(the held-out credit evaluation) and `eval/run_calibration_decisions.py` with
`eval/run_routing_decisions.py` (bad approvals avoided per 1,000 decisions,
against random and confidence referral). The decision-level results are
negative for per-decision mistake avoidance and are reported in full in
[section 9c of CALIBRATION_TRUST.md](CALIBRATION_TRUST.md#9c-does-it-avoid-mistakes-decision-level-results-evalrun_calibration_decisionspy-evalrun_routing_decisionspy).
Under the payment-delay shift the reference-fitted opinion claims a
calibration error of 0.061 while the realised value is 0.225, which is what the
delayed-outcome verdict is for.

**Shift benchmark v2 (exploratory).** `eval/run_shift_bench_v2.py` asks the
routing question in three regimes a lender can actually meet: a shift in
progress (mixed populations), corrupted inputs (unit errors, missing fields,
noise) and the severe split. Confidence wins when nothing has moved; confidence
gated by the OOD distance catches 2.15x random and 1.60x confidence when a
quarter of the book has shifted; nothing ranks once the shift is complete,
where the drift test fires in every seed. A currency error on monetary fields
is caught 100% by distance and 1% by confidence; zeroed fields are caught by
neither. Write-up: [SHIFT_BENCHMARK_V2.md](SHIFT_BENCHMARK_V2.md).

**Exploratory expansion.** Pre-specified in the second pre-registration
amendment ([PREREGISTRATION_AMENDMENT.md](PREREGISTRATION_AMENDMENT.md))
before any download; nothing here touches the confirmatory datasets. Runners
in `eval/expansion/`, results in `results/expansion/`, discussion in section 7
of the paper.

| Workstream | Source | In one line | Write-up |
|---|---|---|---|
| Cross-domain | TableShift, 8 non-credit tasks with the benchmark's own shifts | Coverage holds in-distribution on all 8 and falls under shift on 7 of 8; the input drift test fires on most shifted streams, and the one task where it fails (hospital readmission, sparse one-hot inputs, small calibration set) is diagnosed | [TABLESHIFT.md](results/expansion/TABLESHIFT.md) |
| Parity on reported class | HMDA 2025, 4.7M mortgage applications, label = denied | Marginal coverage 0.924 Black, 0.936 Hispanic, 0.956 White at a 0.95 target; per-segment calibration brings every non-thin group within 0.01; a geography-only proxy misassigns 27.7% of applicants | [HMDA_PARITY.md](results/expansion/HMDA_PARITY.md) |
| Latency at scale | Bootstrapped rows of Taiwan dimensionality | 8.5, 41 and 493 ms median at 7,500, 100,000 and 1,000,000 calibration rows; the Venn-Abers fit dominates | [LATENCY.md](results/expansion/LATENCY.md) |
| Temporal drift | Home Credit Stability 2024, 1.53M decisions over 92 weeks | Coverage 0.953 through December 2019, 0.936 in February and March 2020, 0.96 or more from April 2020. The pre-registered hypothesis that at least one month would fall below 0.94 is confirmed. Recalibration on 4 weeks of landed outcomes recovers 94% of the coverage gap | [HOMECREDIT_STABILITY.md](results/expansion/HOMECREDIT_STABILITY.md) |
| Macro drift | Fannie Mae and Freddie Mac loan performance data | Not run: gated on the provider's terms | |

To reproduce the expansion: `data_fetch/README.md` (TableShift environment in
`data_fetch/TABLESHIFT_ENV.md`), then `eval/expansion/run_tableshift.py`,
`run_hmda_parity.py`, `run_latency_scale.py`, `diag_martingale_sparse.py`.

## Layout

- `eval/` - dataset loaders, baselines and metrics, experiment runners, figures.
- `data/` - the two UCI datasets, verbatim, with `PROVENANCE.md` (CC BY 4.0, redistributed here).
- `data_fetch/` - fetch scripts for public sources that are not redistributed here (TableShift, HMDA, ACS).
- `data_gated/` - instructions only, for sources whose terms prohibit redistribution (Fannie Mae, Freddie Mac, Kaggle).
- `docs/` - the evaluation expansion plan; the binding protocol is `PREREGISTRATION_AMENDMENT.md`.
- `results/` - `results.json`, `shift_bench.json`, the paper figures and `expansion/`.

The method code lived in a `reliax_core/` directory here until 25 September
2026; the pre-registration names commits of this repository as the frozen
method code, and those commits remain in this history. Every runner now
imports the installed `reliax-core` package and reproduces the committed
results exactly (only timings and run dates differ).

## Licence

Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Data is
redistributed where the licence permits (UCI, CC BY 4.0, see
`data/PROVENANCE.md`), fetched by script where the source is public
(TableShift, HMDA, ACS) and instructions-only where the provider's terms
prohibit redistribution (Fannie Mae, Freddie Mac, Kaggle).

## About this documentation

The documentation in this repository was written with the help of AI and
reviewed by the Reliax team.
