# Pre-registration amendment, draft for review (September 2026)

Status: DRAFT. Not yet published to the repository as the binding protocol.
Give Me Some Credit and Home Credit remain untouched. Once accepted, this file
is committed with a date and the confirmatory runs follow it verbatim.

## Why amend

The corrected bar of the first pre-registration asks the reliability ranking
to reach 1.5x random referral, and to beat confidence with a bootstrap CI
above 1.0, on shifted splits that cost the model at least 5 AUC points. On
Taiwan the only split that clears that floor (delayed payers) is so severe
that random referral beats the model's own confidence: the population has
changed, not drifted. In that regime the correct product behaviour is to
stop auto-deciding (tripwire, suspended guarantee, recalibration), not to
rank better, and the v1 benchmark therefore tested the ranking in the one
regime where ranking is the wrong tool.

The exploratory v2 benchmark (`eval/run_shift_bench_v2.py`, Taiwan only)
shows the regime in between: when the live population is a MIXTURE of the
reference population and the shifted one, the model's confidence loses its
ordering while the shifted rows are still a minority, and signals that use
the input-space distance to the calibration data recover it. It also shows
that corrupted inputs (a unit error on monetary fields) are invisible to
confidence and visible to the same distance.

This amendment therefore keeps the original bar for the SEVERE tier as the
tripwire test, and adds a MODERATE tier and a CORRUPTION tier on which the
ranking bar is evaluated. The bar itself is unchanged.

## Tiers (per confirmatory dataset)

Reference population, shifted population and monetary fields are declared
per dataset before any model is trained (table below). Train and calibrate
on the reference population only. Test sets of 4,000 rows, 5 seeds.

| tier | construction | what is claimed |
|---|---|---|
| calm | reference population only (lambda = 0) | the certified REVIEW threshold; no ranking claim |
| moderate | lambda in {0.25, 0.50} of the test rows drawn from the shifted population | the ranking bar (below) |
| severe | lambda in {0.75, 1.00} | tripwire: martingale reaches ALARM within the stream; REVIEW rate rises; no ranking claim |
| corruption | reference population, rho in {0.05, 0.10} of rows with monetary fields x100 | the referral contains at least 80% of the corrupted rows; the martingale reaches ALARM |

Declared populations:

| dataset | reference | shifted | monetary fields |
|---|---|---|---|
| Give Me Some Credit | NumberOfTime30-59DaysPastDueNotWorse = 0 | >= 1 | MonthlyIncome, DebtRatio x income |
| Home Credit | no previous late payment in bureau data | any late payment | AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE |

## The bar (unchanged in substance, applied to the moderate tier)

At a 10% referral rate, the signal under test must catch at least 1.5x the
bad approvals that random referral catches, AND beat the model's own
confidence with a 95% paired bootstrap CI above 1.0, on both lambda values,
on at least two of the three datasets (Taiwan counts as exploratory, so in
practice on both confirmatory datasets).

The signal under test is "confidence gated by OOD", fixed here: applicants
whose kNN distance to the calibration data exceeds its 95th calibration
percentile are referred first, ordered by distance; all others are ordered
by the model's confidence. It is stateless and per decision. The
regime-switching rule (confidence while the exchangeability martingale on
OOD distance is OK, distance once it is WATCH or ALARM) is the secondary
signal, reported alongside. Parameters (k = 10 neighbours, flag at the 95th
percentile, WATCH at 20, ALARM at 100) are frozen at the values in
`reliax_core` at the commit of this amendment. Exploratory Taiwan values
(`SHIFT_BENCHMARK_V2.md`): gated 2.15x random and 1.60x confidence at
lambda 0.25, 1.61x and 1.38x at lambda 0.50.

Fail: the triage rank stays advisory and the certified envelope ships
without it. Pass: the triage rank ships as the routing rule for the
moderate regime only; the calm regime keeps the certified threshold and the
severe regime keeps the tripwire.

## What this amendment does not do

It does not change the bar, the referral rate, the seed count, the bootstrap
procedure or the AUC-floor logic (the moderate tier's AUC loss is reported;
if a dataset's moderate tier costs the model under 3 AUC points, the tier is
reported as "mild" and does not count towards the pass). It does not add
free-text or agent tasks. It is published before the confirmatory datasets
are opened, with this rationale, so that the change is visible as a
pre-specified amendment rather than an after-the-fact adjustment.

---

# Amendment 2: dataset disambiguation and the exploratory expansion (13 September 2026)

Status: written before any of the datasets named below were downloaded,
opened or inspected. Frozen method code: `reliax_core/` at commit 05db815.
The commit that adds this section is the record; the plan this amendment
implements is `docs/EVALUATION_EXPANSION_PLAN.md`.

## 2.1 Which "Home Credit" is confirmatory

The first pre-registration (PAPER.md, section 5, "Next steps") and Amendment 1
above name "Home Credit" without a year. Two different Kaggle releases carry
that name:

| release | year | identity | status |
|---|---|---|---|
| Home Credit Default Risk | 2018 | `application_train.csv` plus `bureau.csv` and related tables; no decision timestamps | CONFIRMATORY, sealed |
| Home Credit, Credit Risk Model Stability | 2024 | base table keyed by `case_id` with `date_decision`, `WEEK_NUM`, `MONTH`, `target`; feature tables in a different schema | EXPLORATORY (section 2.6) |

Reading adopted: the confirmatory set is the **2018 Home Credit Default Risk**
release. This is not a choice made now; it follows from Amendment 1, which
declared the reference population as "no previous late payment in bureau
data" and the monetary fields as `AMT_INCOME_TOTAL`, `AMT_CREDIT`,
`AMT_ANNUITY`, `AMT_GOODS_PRICE`. Those are column names of the 2018
release's `application_train.csv` and `bureau.csv`; the 2024 release uses a
different naming scheme and no table called bureau. The original wording was
ambiguous; the declared fields were not.

## 2.2 Sealed sets

**Give Me Some Credit (Kaggle, 2011) and Home Credit Default Risk (Kaggle,
2018) remain sealed.** Nothing in this amendment touches them. They are not
downloaded, not used for tuning, not used as a sanity check, and not used to
choose any parameter of the exploratory work. They are opened only for the
confirmatory runs of Amendment 1, after Amendment 1 is accepted.

## 2.3 The standing rule for everything below

Every dataset named in sections 2.4 to 2.8 enters the **exploratory** bucket.
Exploratory results can inform the product, the whitepaper's exploratory
sections and the deck's "not yet confirmed" boxes. They cannot be used to
re-tune the triage rank, move the 1.5x bar, change k, the flag percentile,
the WATCH or ALARM thresholds, or generate a confirmatory claim. Every task
that is started is reported, including the ones where a guarantee breaks.

Method parameters, frozen at the commit of this amendment, for every run
below: split conformal with nonconformity 1 - p_hat_y; Mondrian conformal per
audit segment; inductive Venn-Abers as implemented in
`reliax_core/venn_abers.py`; kNN OOD distance with k = 10 in standardised
feature space; flag at the 95th calibration percentile; mixture power
martingale over the epsilon grid 0.05 to 0.95 with WATCH at 20 and ALARM at
100; PSI as implemented. Base model: `HistGradientBoostingClassifier`
(scikit-learn, `max_iter=300`, seed = run seed) unless a section says
otherwise. Coverage targets alpha = 0.05 (primary) and 0.10 (secondary).

Definitions used throughout:

- REVIEW rate: fraction of test rows whose conformal set at alpha = 0.05 is
  not a singleton (empty or both classes).
- Martingale stream: 600 rows drawn in the stated order; the score fed to the
  martingale is the kNN distance to the calibration set. ALARM = wealth
  reaches 100 within the stream. False alarm = wealth reaches 20 (WATCH) at
  any point in an in-distribution stream (the stricter reading; identical to
  `eval/run_eval.py`).
- Calibration-trust disbelief: `CalibrationTrust` (M = 10, quantile bins)
  fitted on the calibration split (claimed) and re-evaluated on the test
  split with realised outcomes (realised), as in `eval/run_calibration_credit.py`.
- Thin cell: any segment or bin with fewer than 100 test rows is reported but
  flagged and excluded from pass/fail statements.
- Seeds: 5 per task unless stated. Seed controls the model, the calibration
  subsample and the stream order. Mean and standard deviation over seeds.

## 2.4 Workstream A: cross-domain generalisation, TableShift (exploratory)

Source: TableShift (Gardner, Popovic and Schmidt, NeurIPS 2023 Datasets and
Benchmarks; `github.com/mlfoundations/tableshift`). The domain split is the
one TableShift defines for each task and is not modified. Splits used:
TableShift `train` for the model, TableShift `validation` as the calibration
set, `id_test` and `ood_test` as the two test sets. Where a split exceeds the
cap, a seeded uniform subsample is taken: train 200,000 rows, calibration
20,000, each test split 20,000. Audit segments are the sensitive attributes
TableShift returns with each split (the third element of `get_pandas`).

Tasks, declared in advance. The benchmark has 15 tasks; 10 carry a domain
split, 5 do not.

| task identifier | domain | shift variable | access | run |
|---|---|---|---|---|
| `acsincome` | finance | geographic region | public | yes |
| `acsfoodstamps` | public policy | geographic region | public | yes |
| `acsunemployment` | labour | education level | public | yes |
| `brfss_diabetes` | health | race | public | yes |
| `diabetes_readmission` | health | admission source | public | yes |
| `college_scorecard` | education | institution type | public | yes |
| `assistments` | education | school | public (Kaggle-hosted) | yes if the file can be fetched under its terms; reported either way |
| `anes` | civic | geographic region | credentialed (ANES) | if credentials are obtained within the plan window; reported either way |
| `mimic_extract_mort_hosp` | health | insurance type | credentialed (PhysioNet) | same |
| `mimic_extract_los_3` | health | insurance type | credentialed (PhysioNet) | same |
| `heloc`, `acspubcov`, `physionet`, `nhanes_lead`, `brfss_blood_pressure` | various | none defined by TableShift | mixed | excluded from the shift analysis: no OOD split exists. Excluded before any data was seen. |

Quantities measured per task, per seed, on `id_test` and `ood_test`:

1. Model accuracy and AUC (context only; nothing is claimed about the model).
2. Marginal coverage at alpha = 0.05 and 0.10, against target.
3. Per-segment coverage, marginal calibrator and Mondrian calibrator, for
   every TableShift sensitive attribute; worst segment reported.
4. Coverage degradation, ID minus OOD, at alpha = 0.05.
5. Calibration-trust disbelief, claimed (calibration split) vs realised (each
   test split), and Venn-Abers ECE on each test split.
6. Martingale: ALARM rate over 5 OOD streams; false-alarm rate over 5 ID
   streams; steps to WATCH and ALARM where reached.
7. REVIEW rate on ID and on OOD.

Pre-specified expectations (exploratory hypotheses; each is reported as met
or not met, per task):

- H-A1. ID coverage at alpha = 0.05 lies within 0.01 of 0.95 on every task
  (finite-sample conformal guarantee on exchangeable data).
- H-A2. On tasks where TableShift's own baseline reports an OOD accuracy drop
  of 5 points or more, OOD coverage falls below ID coverage, and the sign is
  consistent across seeds. The size of the drop is reported, not predicted.
- H-A3. On the same tasks, the martingale reaches ALARM in at least 4 of 5
  OOD streams. On every task, the false-alarm rate on ID streams is at most 1
  in 5 (Ville: at most 0.05 per stream at WATCH).
- H-A4. REVIEW rate on OOD is at least the REVIEW rate on ID on every task
  where H-A2 holds (set-size routing self-adjusts).
- H-A5. Realised disbelief on OOD exceeds claimed disbelief on every task
  where H-A2 holds.

What a failure means: a task where H-A1 fails is a bug or a non-exchangeable
split and is investigated and reported as such. A task where H-A3 fails
(shift present, no ALARM) is a real limitation of the input-space tripwire
and is reported as such; it is the case slide 08 already describes.

Definition of done: one table, one row per task run, with coverage ID, coverage
OOD, target, ALARM rate, false-alarm rate, REVIEW rate ID and OOD. All tasks
started are in the table. Nothing about latency, parity on protected
attributes, or customer data is claimed from this workstream.

## 2.5 Workstream C: coverage parity on reported protected attributes, HMDA (exploratory)

Source: HMDA Modified Loan Application Register, 2025 reporting year, FFIEC
and CFPB (`ffiec.cfpb.gov/data-publication/modified-lar`), fetched by script.
Public data; no redistribution in this repository beyond the fetch script.

Stated up front: **the HMDA outcome is a lender's decision (originate or
deny), not a repayment outcome.** This workstream measures coverage parity,
set-size distribution and REVIEW-rate parity of the envelope over a
decision-prediction model. It does not and cannot measure PD calibration or
default outcomes.

Population and label, declared now: rows with `action_taken` in {1 originated,
3 denied}; single-family (1 to 4 units), site-built, first lien, principal
residence, home purchase or refinancing (`loan_purpose` in {1, 31, 32}),
non-reverse mortgage, non-business purpose. Label y = 1 if denied. Features:
loan amount, income, loan-to-value, debt-to-income band, loan term, interest
rate presence, loan type, loan purpose, occupancy, property value, applicant
age band, co-applicant presence, AUS name, state. Protected attributes,
audit only, never in the features: `derived_race`, `derived_ethnicity`,
`derived_sex`. Seeded subsample of 400,000 qualifying rows for the model plus
calibration, and 200,000 for test, stratified on the label. 5 seeds.

Quantities measured per protected group:

1. Marginal and Mondrian coverage at alpha = 0.05 and set-size distribution.
2. REVIEW rate with a 95% Wilson interval, and the four-fifths ratio of the
   lowest to the highest group REVIEW rate.
3. Thin cells flagged (under 100 test rows, or under 500 calibration rows for
   the Mondrian calibrator).

Proxy-vs-reported comparison: the same audit re-run with the protected
attribute replaced by a geography-only proxy. The modified LAR has no names,
so the surname component of BISG cannot be reproduced; the proxy is the
geocoding component only (census-tract race and ethnicity shares from the
ACS 5-year table B03002, fetched by script), with each applicant assigned the
tract's modal group. This is stated as a BIG proxy, not BISG, wherever the
result appears. Reported: per-group REVIEW rate under proxy vs reported
attribute, per-group coverage under proxy vs reported, and the fraction of
applicants whose proxy group differs from the reported group.

Pre-specified expectations:

- H-C1. Mondrian coverage at alpha = 0.05 lies within 0.01 of 0.95 for every
  non-thin group.
- H-C2. Marginal (non-Mondrian) coverage differs across groups by more than
  0.01 for at least one group pair; this is the case Mondrian exists for.
- H-C3. The proxy audit misstates at least one group's REVIEW rate by more
  than its Wilson interval; the direction and size are reported, not predicted.

Definition of done: a parity table by reported protected class, a proxy vs
reported delta table, and a rewritten slide 12 bullet citing a measurement.

## 2.6 Workstream D: native temporal drift, Home Credit Credit Risk Model Stability 2024 (exploratory)

Source: Kaggle competition "Home Credit, Credit Risk Model Stability" (2024).
This is not the sealed 2018 release (section 2.1). Access requires a Kaggle
account and acceptance of the competition rules; the rules restrict use to
non-commercial and academic purposes. **Gate:** this workstream starts only
after a written read of those terms confirms that publication in a vendor
whitepaper is permitted, or after written consent. If neither is obtained
within the plan window, the workstream is reported as not run and the reason
stated. Data is never redistributed from this repository.

Base model, fixed in advance so that the envelope, not the model, is
evaluated: LightGBM (`num_leaves=64`, `learning_rate=0.05`, 500 rounds, seed
= run seed) on the base table plus the depth-0 static tables only, with the
competition's published starter preprocessing. No feature engineering beyond
that. No tuning on AUC.

Temporal protocol: sort by `date_decision`. Train on the first 40% of weeks,
calibrate on the next 10%, evaluate on the remaining 50% in `WEEK_NUM` order.

Quantities measured:

1. Rolling weekly and monthly coverage at alpha = 0.05 against target across
   the evaluation period.
2. Martingale wealth over calendar time, with WATCH and ALARM crossings dated.
3. Claimed vs realised disbelief, month by month.
4. Weighted-conformal recalibration (Tibshirani, Barber, Candes and Ramdas,
   2019; likelihood-ratio weights from a logistic domain classifier between
   the calibration window and the trailing 4 weeks), triggered at the first
   ALARM: coverage in the 8 weeks after the trigger, with and without
   recalibration. This is the repair path slide 16 describes as a design.

Pre-specified expectations:

- H-D1. Monthly coverage at alpha = 0.05 drifts below 0.94 in at least one
  month of the evaluation period (the competition was built around
  instability; if coverage never moves, the set is uninformative for this
  purpose and that is reported).
- H-D2. Where H-D1 holds, the martingale reaches WATCH before the first month
  whose coverage falls below 0.94, on at least 3 of 5 seeds.
- H-D3. Weighted recalibration recovers at least half of the coverage gap
  (target minus pre-recalibration coverage) in the 8 weeks after the trigger.
  The recovered fraction is the headline number of this workstream.

## 2.7 Workstream B: latency at scale and macro drift, Fannie Mae and Freddie Mac (exploratory)

### B1, latency

The latency measurement is a property of the engine, not of any dataset.
Calibration sets of 7,500, 100,000 and 1,000,000 rows at the Taiwan feature
dimensionality (23 features) are built by seeded bootstrap of the Taiwan
calibration split with Gaussian jitter at 5% of each feature's standard
deviation; this is stated on every figure that reports the number. If the
Fannie Mae licence read (B2) clears, the same harness is re-run on real
acquisition rows and both numbers are reported.

Measured, per calibration size, over 1,000 queries after 100 warm-up queries,
3 repeats, median and p95: conformal quantile lookup and set construction;
Venn-Abers interval; kNN distance and percentile, exact index (scikit-learn
`NearestNeighbors`, default algorithm) and, separately, an approximate index
(HNSW via `hnswlib`) with its recall at 10 against the exact index; martingale
update; the full envelope. The component that dominates as n grows is named.
The in-VPC network hop is measured as a loopback HTTP round trip on the same
host over a minimal JSON endpoint and is labelled as such; a true in-VPC hop
is measured in the pilot deployment, not here.

The implementation measured is `reliax_core/` at the commit of this
amendment. If a faster implementation of any component is written later, its
latency is reported only after a test shows it returns identical outputs to
the frozen one on the Taiwan test split.

Pre-specified expectation: H-B1, the full-envelope median at 1,000,000 rows
exceeds 100 ms with the current Venn-Abers implementation, which refits an
isotonic regression per query. This is stated in advance so that the number
is a measurement of the frozen code and not a surprise.

### B2, macro drift (gated)

Source: Fannie Mae Single-Family Loan Performance Data (registration and
terms acceptance required) or the Freddie Mac equivalent. **Gate:** the
terms restrict redistribution and use "in support of external commercial
purposes"; this workstream starts only after written consent from the
provider or a lawyer's read confirming the intended use. Data is never
redistributed from this repository. If the gate does not clear within the
plan window, the workstream is reported as not run.

Protocol, declared now: acquisition-time features only (credit score,
loan-to-value, combined loan-to-value, debt-to-income, original rate,
original balance, loan purpose, property type, occupancy, number of
borrowers, first-time-buyer flag, channel, state). Label: 90 or more days
delinquent within the first 24 months of performance. Train on 2005 and 2006
acquisitions (seeded subsample of 400,000), calibrate on a disjoint 100,000
of the same vintages, evaluate quarterly on 2007 Q1 to 2010 Q4 acquisitions.
Live verdict: martingale on kNN distance in acquisition order. Outcome
verdict: quarterly coverage at alpha = 0.05 on realised labels. Lag: quarters
between the first ALARM and the first quarter whose coverage is below 0.93
for two consecutive quarters.

Pre-specified expectations: H-B2a, outcome coverage falls below 0.93 for at
least two consecutive quarters somewhere in 2007 to 2009. H-B2b, the input
martingale's first ALARM comes later than the first outcome breach, or not at
all, on at least 3 of 5 seeds: the macro cycle changes outcomes more than it
changes application inputs, which is why the product carries two verdicts.
Either direction is reported.

## 2.8 Repository tiers and the public-data claim

The claim "code and data public and verified reproducible" is replaced,
everywhere it appears, by a tiered statement:

> Code is public for every experiment. Data is redistributed where the
> licence permits (UCI, CC BY 4.0), fetched by script where the source is
> public (TableShift, HMDA, ACS), and instructions-only where the provider's
> terms prohibit redistribution (Fannie Mae, Freddie Mac, Kaggle).

Layout: `data/` (tier 1, shipped), `data_fetch/` (tier 2, scripts),
`data_gated/` (tier 3, instructions only). Exploratory runners live in
`eval/expansion/`; results in `results/expansion/`.

## 2.9 Order of work and what is not done

Order: this amendment, then A, then C, then D (gated), then B1, then B2
(gated), then the whitepaper's exploratory section. Nothing in this
amendment is a substitute for a shadow replay on a lender's book, and the
CRO conversations continue in parallel.

Nothing in this amendment changes Amendment 1: not the bar, not the referral
rate, not the seed count, not the bootstrap, not the AUC floor, not the
declared populations.
