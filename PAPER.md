# Per-Decision Reliability Envelopes for Credit-Risk Models: An Evaluation on Real Data

**Ismael Ouattara, Sébastien Michel** · Reliax
Working paper v0.3 draft, September 2026 (v0.2 added the subjective-logic fusion signal; v0.3 adds section 7, the exploratory expansion). Code and data: `paper/` in the Reliax repository.

## Abstract

Credit-risk models execute high-consequence decisions whose failures are silent:
a wrong approval looks identical to a right one until the loss materialises. We
evaluate the Reliax *reliability envelope*, a per-decision bundle of a conformal
coverage certificate (marginal and per-segment), a Venn-Abers calibrated
probability-of-default (PD) interval, an error-auditor signal, and an
anytime-valid exchangeability test, on two real public credit datasets: the
Taiwan credit-card default dataset (30,000 applicants) and the German credit
dataset (1,000 applicants). Findings: (1) the conformal coverage guarantee is
verified empirically (0.9515 ± 0.0040 and 0.9532 ± 0.0167 at a 0.95 target);
(2) per-segment coverage audits on real sex and age attributes show only mild
segment deviations on these datasets, and the Mondrian repair is cost-free
insurance; (3) for selective prediction on in-distribution data,
confidence-based ranking is a strong baseline, while the error auditor gives
the best capture of wrong approvals (18.3% vs 17.2% at a 10% referral rate);
(4) Venn-Abers matches temperature scaling when the base model is already
calibrated and beats it clearly when it is not (ECE 0.065 vs 0.083 on German
credit, raw 0.186), and its interval width predicts model errors (error rate
0.229 in the widest decile vs 0.174 elsewhere); (5) the conformal test
martingale raises zero false alarms on 5 x 600 i.i.d. applications and detects
a real subpopulation shift in 4 of 5 streams within 600 applications; (6) a
subjective-logic fusion of the envelope's signals achieves the best
risk-coverage trade-off of all signals on the larger dataset (AURC 0.0955)
while clearly improving on the weighted-blend composite it replaces, though it
trails plain confidence on the small dataset; (7) the full envelope computes
in 8.0 ms median per decision. We state explicitly what
these results do not show: the pre-registered shifted-data benchmark that
motivates the uncertainty-aware signals remains future work. Section 7
(v0.3, exploratory, pre-specified before any download) adds coverage parity
on reported protected class in 4.7 million HMDA 2025 decisions, latency at
up to a million calibration rows, and seven non-credit TableShift tasks under
the benchmark's own shifts, including one where the input-space tripwire
fails and why.

---

## 1. Introduction

Production credit models fail three ways, all silently: confident errors on
individual applications, calibration decay when the economy shifts, and
segment-level unreliability hidden by aggregate metrics. Existing tooling
(aggregate monitoring, post-hoc explainability) does not answer the operational
question: *should this specific decision be acted on automatically, right now,
and can that be evidenced later?*

Reliax answers with a per-decision reliability envelope. This paper is a first
evaluation of its components on real data. It is deliberately small and
deliberately honest: every number below is produced by `eval/run_eval.py` and
stored in `results/results.json`; nothing is hand-typed without a source there.

## 2. The reliability envelope

For an application x with model output p(x), the envelope carries:

| Component | Construction | Guarantee class |
|---|---|---|
| Coverage certificate | Split conformal set Ĉ(x) at level α | Proven: P(Y ∈ Ĉ(X)) ≥ 1-α, distribution-free, finite-sample (Vovk et al., 2005) |
| Segment certificate | Mondrian conformal, one q̂ per audited segment | Proven per group: P(Y ∈ Ĉ(X) \| G=g) ≥ 1-α |
| PD interval [p0, p1] | Inductive Venn-Abers over calibration scores | Proven in the Venn sense: one of the two isotonic calibrators is perfectly calibrated (Vovk & Petej, 2014) |
| Drift verdict | Mixture power martingale on smoothed conformal p-values of a label-free nonconformity (kNN distance) | Proven, anytime-valid: P(sup M ≥ c) ≤ 1/c under exchangeability (Ville, 1939; Vovk et al., 2003) |
| Auditor | Meta-model trained on the base model's own calibration-split errors | Heuristic ranking signal |
| Composite score (v1) | 0-100 weighted blend with hard caps | Heuristic, labelled as such |
| Trust triple (SL fusion) | Subjective-logic averaging fusion of certificate-gated confidence, auditor and context opinions (Josang, 2016; PaTAS lineage) | Heuristic, calculus-backed |

Protected attributes are audit-only: the base model never receives them.

## 3. Experimental setup

**Datasets.** (a) *Taiwan*: Default of Credit Card Clients (Yeh & Lien, 2009;
UCI 350), 30,000 applicants, 22.1% default rate. SEX, AGE, MARRIAGE excluded
from the 20 model features; sex and age bands (21-30 / 31-45 / 46+) used for
audits. (b) *German*: Statlog German Credit (UCI 144), 1,000 applicants, 30%
bad rate, one-hot encoded to 56 features excluding sex/personal-status and age.
Provenance and licenses in `data/PROVENANCE.md`.

**Protocol.** Stratified 50/25/25 train/calibration/test split. Base model:
histogram gradient boosting (scikit-learn, 300 iterations). 5 seeds (Taiwan),
10 seeds (German); all tables report mean ± std over seeds. α = 0.05
throughout.

**Baselines and metrics.** Temperature scaling fitted on the calibration split
(Guo et al., 2017) for calibration comparisons; max-softmax probability (MSP)
ranking for selective prediction (T-scaling is monotone in the logit, so raw
and T-scaled MSP produce the same ranking). Metrics: empirical coverage;
per-segment coverage; error capture and wrong-approval capture at fixed
referral rates plus AURC (area under the risk-coverage curve, lower is
better); 15-bin ECE and Brier score; martingale false alarms and detection
delay; wall-clock latency. Ranking signals include (v0.2) the subjective-logic
fusion's expected value: each signal becomes an opinion (belief, disbelief,
uncertainty) and opinions merge by averaging fusion; the confidence source's
uncertainty is gated by the certificate (conformal p-value, set ambiguity,
interval width), and out-of-distribution inputs drive the context source
vacuous.

**Base model quality** (context): Taiwan accuracy 0.821 ± 0.001, AUC
0.777 ± 0.006; German accuracy 0.749 ± 0.021, AUC 0.769 ± 0.027. Both are
consistent with published results on these datasets.

## 4. Results

### 4.1 The coverage guarantee holds on real data (E1)

| Dataset | Target 1-α | Empirical coverage | Mean set size |
|---|---|---|---|
| Taiwan | 0.95 | **0.9515 ± 0.0040** | 1.49 |
| Taiwan | 0.90 | 0.8997 ± 0.0012 | - |
| German | 0.95 | **0.9532 ± 0.0167** | - |
| German | 0.90 | 0.8964 ± 0.0307 | - |

The certificate does what the theorem says, on real applicants, at both levels.
The mean set size of 1.49 on Taiwan is itself informative: at 95% confidence,
roughly half of real applications cannot be assigned a single outcome, which is
precisely the population a REVIEW route exists for.

### 4.2 Per-segment coverage and bias (E2)

Marginal coverage by real demographic segment (Taiwan, α = 0.05):

| Segment | n (test) | Marginal q̂ | Mondrian q̂ |
|---|---|---|---|
| female | 4,481 | 0.9527 ± 0.0045 | 0.9505 ± 0.0051 |
| male | 3,019 | 0.9497 ± 0.0040 | 0.9538 ± 0.0051 |
| age 21-30 | 2,777 | 0.9581 ± 0.0019 | 0.9544 ± 0.0023 |
| age 31-45 | 3,552 | 0.9486 ± 0.0047 | 0.9522 ± 0.0053 |
| age 46+ | 1,171 | 0.9447 ± 0.0086 | 0.9457 ± 0.0136 |

Two honest observations. First, on this dataset marginal conformal does *not*
severely undercover any audited segment; the worst (age 46+) sits about half a
point below target, within roughly one standard deviation. Second, on German
credit the female segment shows marginal 0.9426 ± 0.0253 against 0.9744 ± 0.0250
after Mondrian repair, but with only 75 test points per seed this is indicative
at best. The conclusion we draw is deliberately modest: segment gaps are
dataset-dependent and cannot be assumed either present or absent; the audit
costs one extra calibration pass, and the Mondrian repair restores a per-group
guarantee wherever a gap does exist while costing essentially nothing where it
does not (Figure 2). We note that fully conditional coverage is impossible in
general (Barber et al., 2021); group-conditional guarantees over audited
segments are the practical middle ground.

### 4.3 Selective prediction (E3)

Ranking signals compared at a 10% referral rate (Taiwan; higher is better):

| Signal | Error capture @10% | Wrong-approval capture @10% | AURC ↓ |
|---|---|---|---|
| MSP (T-scaled) | 0.2317 ± 0.0099 | 0.1719 ± 0.0098 | 0.0958 ± 0.0032 |
| Conformal p-value | 0.2315 ± 0.0099 | 0.1719 ± 0.0098 | 0.0958 ± 0.0032 |
| Venn-Abers width | 0.1269 ± 0.0119 | 0.0782 ± 0.0108 | 0.1468 ± 0.0041 |
| Error auditor | 0.2309 ± 0.0068 | **0.1831 ± 0.0100** | 0.0983 ± 0.0020 |
| Composite score (v1) | 0.2173 ± 0.0042 | 0.1592 ± 0.0042 | 0.1028 ± 0.0016 |
| SL fusion (v0.2) | 0.2266 ± 0.0057 | 0.1686 ± 0.0058 | **0.0955 ± 0.0029** |

Random referral captures 10% by construction; all confidence-linked signals
capture roughly 2.2 to 2.3 times random. Four honest findings. First, on
in-distribution data, plain model confidence is a strong selective-prediction
baseline, and the conformal p-value ranking coincides with it by construction
(both are monotone in the top-class probability under a shared score function).
Second, the error auditor is the best signal for the economically relevant
event, wrong approvals (applications approved by the model that then default),
capturing 18.3% vs 17.2% at 10% referral, about 7% relative improvement, and it
achieves this while being trained only on calibration-split mistakes. Third,
the v1 composite underperforms pure confidence on i.i.d. data (it spends
ranking capacity on OOD and width components that are uninformative when
nothing is shifted).

Fourth (v0.2): replacing the weighted blend with a subjective-logic fusion
recovers most of that loss. On Taiwan, SL fusion attains the best AURC of all
six signals (0.0955 ± 0.0029) and improves on the v1 blend on every metric.
The mechanism matters: the confidence opinion's uncertainty is gated by the
certificate, so a confidently wrong prediction with an atypical conformal
score abstains rather than asserting, and out-of-distribution inputs
contribute uncertainty mass rather than fake confidence. The honest limit: on
German credit (calibration n = 250) SL fusion trails both confidence (capture
0.1522 vs 0.1874) and the v1 blend, because its input signals (auditor,
interval width) are themselves noisy at that scale and the fusion inherits
their noise. SL fusion helps when its evidence sources are well estimated; it
is not magic on small data.

None of this tests the regime that motivates the uncertainty-aware signals:
**this experiment does not test the shifted-data regime**. The pre-registered
benchmark (2x wrong-approval capture vs temperature-scaled confidence on
frozen shifted splits) remains future work and is not claimed here.

### 4.4 Calibrated PD intervals (E4)

| Dataset | Raw ECE | T-scaled ECE | Venn-Abers ECE | Fitted T |
|---|---|---|---|---|
| Taiwan | 0.0132 ± 0.0024 | 0.0127 ± 0.0024 | 0.0155 ± 0.0048 | 1.01 |
| German | 0.1862 ± 0.0171 | 0.0826 ± 0.0195 | **0.0649 ± 0.0181** | 3.99 |

The pattern is clean: the Taiwan model is already essentially calibrated
(fitted temperature 1.01), and no post-hoc method adds anything. The German
model is badly miscalibrated (temperature 3.99, raw ECE 0.186), and there
Venn-Abers gives the best calibration, ahead of temperature scaling. Since a
lender cannot know in advance which regime they are in, a method that is
harmless in the first and best in the second is the right default, and it comes
with a per-decision interval rather than a single number.

The interval width carries signal of its own: on Taiwan, the widest decile of
PD intervals has a model error rate of 0.229 ± 0.029 versus 0.174 ± 0.003 for
the rest, a 32% relative difference (Figure 4). Width is a per-decision warning
even though it is a weak global ranker (Table in 4.3).

### 4.5 Drift detection with controlled error (E5)

Streams of 600 real applications drawn from the held-out test split. The
i.i.d. stream is sampled uniformly; the shifted stream is sampled from the real
delayed-payment subpopulation (PAY_0 ≥ 1, which is 22.4% of applicants), a
covariate shift by conditioning, not a synthetic perturbation.

| Quantity | Value |
|---|---|
| False alarms on i.i.d. streams (5 seeds x 600 apps) | **0 / 5** (max log10 M = 0.22 ± 0.56) |
| Streams reaching ALARM under shift | 4 / 5 within 600 apps |
| Applications to ALARM (when reached) | 192 ± 110 |
| PSI on the shifted window (top features, seed 0) | PAY_0: 7.37, PAY_2: 1.11, PAY_3: 0.64 (all MAJOR) |

The anytime-validity guarantee (P(false alarm, ever) ≤ 0.01 at the ALARM
threshold) is consistent with the observed zero false alarms, and the detector
fires on a genuinely shifted stream of real applicants (Figure 3). Caveats: the
shift induced here is strong (conditioning on a highly predictive feature), and
one stream did not alarm within 600 applications; milder shifts will take
proportionally longer, which is the correct behaviour of an evidence-accumulating
test rather than a defect.

### 4.6 Latency (E6)

Full per-decision path (model inference, conformal set and p-value, Venn-Abers
interval, kNN OOD, auditor) with a calibration set of 7,500: **median 8.0 ms,
p95 17.5 ms** per application on commodity hardware. The Venn-Abers isotonic
fits dominate; a precomputed IVAP would reduce this further. This is comfortably
inside a 40 ms gate budget.

## 5. Discussion and limitations

**What this evaluation shows.** The proven components behave as proven, on real
data: coverage at target, zero drift false alarms with detection on real
subpopulation shift, calibration that is never worse and sometimes much better
than the standard baseline, an auditor that improves capture of exactly the
errors that cost money, and a subjective-logic fusion that beats the ad-hoc
blend it replaced and matches confidence-based ranking on the larger dataset. The full stack runs at single-digit millisecond median
latency.

**What it does not show.** (1) No shifted-data selective-prediction results:
the motivating regime of the uncertainty-aware signals (v1 composite, SL
fusion) is untested here; on i.i.d. data the v1 blend trails plain confidence
and SL fusion roughly matches it on Taiwan while trailing on German. (2) The datasets are one-shot cross-sections; real
credit portfolios have delayed labels and temporal drift that resampling cannot
fully emulate. (3) German credit is small and its sex attribute conflates
marital status; its segment results are indicative only. (4) No comparison yet
against deep-ensemble or MC-dropout baselines. (5) These are the vendor's own
measurements; the planned benchmark is pre-registered and code-released
precisely so that this stops being a caveat.

**Next steps.** Pre-register the shifted-data benchmark (frozen shift splits on
Taiwan, Give Me Some Credit and Home Credit; H0: wrong-approval capture at 10%
referral is less than 2x the T-scaled baseline; abandon threshold stated in
advance), add ensemble baselines, and evaluate delayed-label coverage
backtesting on a portfolio with real timestamps.

## 6. Reproducibility

`paper/eval/run_eval.py` runs every experiment end-to-end in about 2 minutes
(splits, seeds and thresholds fixed in code); `paper/eval/make_figures.py`
regenerates all figures from `paper/results/results.json`. Datasets are stored
verbatim with provenance and licenses in `paper/data/`. Environment: Python
3.14, scikit-learn 1.9, numpy 2.5.

## 7. Exploratory expansion (Amendment 2, September 2026)

Everything in this section is **exploratory**. It was pre-specified in
`PREREGISTRATION_AMENDMENT.md`, Amendment 2, before any of the datasets
were downloaded; it does not touch the confirmatory sets (Give Me Some
Credit, Home Credit Default Risk 2018), the triage-rank bar or any
threshold. Every task started is reported. The per-task numbers are in
`results/expansion/`; the runners are in `eval/expansion/`.

### 7.1 Coverage parity on reported protected class (HMDA 2025)

**The label is a lender's decision, not a default.** The Home Mortgage
Disclosure Act modified LAR reports whether an application was originated or
denied, together with the applicant's reported race, ethnicity and sex. It
carries no repayment outcome, so this experiment measures the envelope's
coverage, set size and REVIEW routing per protected group over a
decision-prediction model; it says nothing about PD calibration. The 2025
national file (52 states, fetched by script from the FFIEC data browser) gives
4,715,455 applications after the pre-specified population filters (first
lien, site-built, one to four units, principal residence, home purchase or
refinancing, not reverse, not business), with a 15.3% denial rate. Per seed:
300,000 rows for the model, 100,000 for calibration, 200,000 for test; five
seeds. The protected attributes are audit-only and never enter the model. One
pre-specified feature, "interest rate presence", was dropped before any model
was fitted: HMDA reports the interest rate only on originated loans, so its
presence is the label. Model AUC 0.864.

Marginal coverage is 0.950 overall and hides a gap: 0.924 for Black
applicants, 0.936 for Hispanic applicants, 0.941 where race is not reported,
against 0.956 for White applicants (target 0.95). The Mondrian calibrator
repairs every non-thin group to within 0.01 of target (worst group 0.945).
This is the case slide 12 describes, measured on reported rather than proxied
attributes, and it is the first dataset in this evaluation where the marginal
certificate genuinely under-covers a protected group.

| group (reported race) | n test | denial rate | marginal coverage | Mondrian coverage | REVIEW rate |
|---|---|---|---|---|---|
| White | 131,236 | 0.129 | 0.956 | 0.950 | 0.145 |
| Asian | 13,116 | 0.127 | 0.949 | 0.952 | 0.084 |
| Black or African American | 17,342 | 0.255 | **0.924** | 0.949 | **0.242** |
| American Indian or Alaska Native | 1,368 | 0.228 | 0.941 | 0.951 | 0.204 |
| Race not available | 30,755 | 0.211 | 0.941 | 0.950 | 0.186 |
| Hispanic or Latino (ethnicity) | 24,917 | 0.183 | **0.936** | 0.952 | 0.165 |

The REVIEW routing does not inherit the parity of the coverage. Set-size
routing sends 24.2% of Black applicants to REVIEW against 14.5% of White
applicants; the four-fifths ratio of REVIEW rates across non-thin race groups
is 0.35 (0.59 across ethnicity, 0.50 across sex). The REVIEW rate tracks the
denial rate: the model is least certain exactly where denials are most
common. A per-group coverage guarantee therefore does not, by itself, give
per-group parity of friction, and a fair-lending review will ask about the
second, not the first. What the envelope does provide is the audit that
makes this visible per decision and per group; whether a 24% REVIEW rate on
one group is acceptable is a policy question the audit surfaces and does not
answer.

**Proxy vs reported.** The same audit was re-run with the applicant's group
replaced by a geography-only proxy (the modal race or ethnicity of the
applicant's census tract, ACS 5-year table B03002). The modified LAR carries
no names, so only the geocoding half of BISG can be reproduced; this is a BIG
proxy, not BISG. The proxy assigns a different group than the applicant
reported for 27.7% of in-scope applicants. It overstates the Hispanic REVIEW
rate by 3.2 points (0.196 vs 0.165 reported) and the Asian rate by 2.4 points
(0.107 vs 0.083), both outside the reported-attribute 95% interval in all
five seeds; the Black and White rates it gets within noise. A proxy-based
parity audit would therefore misstate two of the six groups on this data.
Pre-specified expectations H-C1, H-C2 and H-C3 were all met.

### 7.2 Latency at 7,500, 100,000 and 1,000,000 calibration rows

Latency is a property of the engine, so this measurement uses calibration
sets built by seeded bootstrap of the Taiwan calibration split (23 features)
with 5% Gaussian jitter, not real applicants beyond the first 7,500; the
number is stated with that caveat wherever it appears. Per cell: 100 warm-up
and 1,000 timed queries, three repeats, single thread, no network. The code
measured is `reliax_core/` unchanged.

| component | 7,500 rows | 100,000 rows | 1,000,000 rows |
|---|---|---|---|
| conformal set and p-value | 0.02 / 0.02 ms | 0.11 / 0.22 ms | 1.10 / 2.56 ms |
| Venn-Abers interval | 2.19 / 3.25 ms | 31.6 / 40.9 ms | **472 / 653 ms** |
| kNN distance, exact index | 1.05 / 4.48 ms | 2.21 / 6.93 ms | 10.5 / 24.0 ms |
| kNN distance, HNSW index | 0.08 / 0.21 ms | 0.09 / 0.14 ms | 0.10 / 0.19 ms |
| martingale update | 0.01 / 0.05 ms | 0.01 / 0.01 ms | 0.01 / 0.02 ms |
| full envelope (with model, exact kNN) | 8.5 / 20.9 ms | 41.0 / 56.4 ms | 493 / 637 ms |

Median / p95. HNSW recall at 10 against the exact index: 0.999, 0.991,
0.939; index build 34 s at a million rows against 465 s for the exact
leave-one-out pass. Loopback HTTP round trip on the same host: 0.21 ms median,
0.43 ms p95; this is not an in-VPC hop, which is measured in the pilot.

The pre-stated expectation H-B1 was met: at a million rows the envelope does
not meet a per-decision budget, and the reason is one component. The
inductive Venn-Abers predictor as implemented refits two isotonic regressions
on the whole calibration set for every query, which is linear in n; it is 26%
of the envelope at 7,500 rows and 96% at a million. The conformal quantile
is also recomputed per call (1.1 ms at a million rows) and can be cached.
Neither is a limitation of the method: Vovk, Petej and Fedorova (2015) give
a precomputed IVAP with O(n log n) setup and O(log n) per query. Under
Amendment 2 that implementation is measured only after a test shows it returns
the same intervals as the frozen code on the Taiwan test split; until then the
honest 1,000,000-row number for the shipped code is 493 ms, and the number
without Venn-Abers is about 20 ms with the exact index and about 7 ms with
HNSW.

### 7.3 Cross-domain generalisation (TableShift)

TableShift (Gardner, Popovic and Schmidt, 2023) pairs each of its tasks with a
distribution shift that the benchmark authors defined. The ten tasks with a
domain split were declared in Amendment 2; seven were run (five public, two
Kaggle-hosted) and three need credentialed access (ANES, MIMIC) that was not
obtained in the plan window. The five tasks without a domain split were
excluded before any data was seen. **None of these tasks is credit.** The
model is a gradient-boosted classifier trained on TableShift's `train`
split, calibrated on `validation`, tested on `id_test` and on `ood_test`
(the benchmark's own shift); five seeds; calibration and test capped at
20,000 rows.

| task (domain) · shift | cal n | acc ID / OOD | cov ID | cov OOD | REVIEW ID / OOD | ALARM on OOD | false WATCH / ALARM (100 ID streams) | disbelief claimed / realised OOD |
|---|---|---|---|---|---|---|---|---|
| ACS income (finance) · region | 20,000 | 0.826 / 0.808 | 0.949 | 0.937 | 0.335 / 0.343 | 60% | 0% / 0% | 0.008 / 0.063 |
| ACS food stamps (public policy) · region | 20,000 | 0.848 / 0.820 | 0.949 | 0.943 | 0.290 / 0.326 | 100% | 2% / 1% | 0.007 / 0.015 |
| ACS unemployment (labour) · education | 20,000 | 0.972 / 0.962 | 0.948 | 0.926 | 0.040 / 0.059 | 100% | 2% / 0% | 0.001 / 0.002 |
| BRFSS diabetes (health) · race | 20,000 | 0.876 / 0.834 | 0.950 | 0.931 | 0.217 / 0.246 | 100% | 2% / 2% | 0.007 / 0.038 |
| Hospital readmission (health) · admission source | 4,286 | 0.666 / 0.624 | 0.949 | 0.961 | 0.740 / 0.820 | 100% | 27% / 8% | 0.014 / 0.040 |
| College Scorecard (education) · institution type | 12,320 | 0.954 / 0.871 | 0.946 | 0.848 | 0.015 / 0.039 | 100% | 4% / 1% | 0.003 / 0.020 |
| ASSISTments (education) · school | 20,000 | 0.938 / 0.583 | 0.950 | 0.590 | 0.027 / 0.010 | 40% | 1% / 0% | 0.007 / 0.308 |

What holds off credit. (1) In-distribution coverage is within 0.01 of target on all seven tasks, on data nobody at Reliax chose (H-A1 met everywhere). (2) Under the benchmark's shift, coverage falls on six of seven tasks, by 0.6 points (food stamps) to 10 points (College Scorecard) to 36 points (ASSISTments); the exception is hospital readmission, where the model's probabilities become less extreme on the new admission source, the sets widen, and coverage overshoots to 0.961 while the REVIEW rate rises from 74% to 82%. (3) The REVIEW rate rises under shift on six of seven tasks: set-size routing self-adjusts when the model becomes less certain. (4) The realised calibration disbelief exceeds the claimed value on all seven tasks under shift, by 1.5x (unemployment) to 44x (ASSISTments), which is what the delayed-outcome verdict is for. (5) The input-space martingale reaches ALARM within 600 rows on every OOD stream of five tasks and on 60% of ACS income's, with 0 to 4% false alarms at WATCH on six tasks.

**ASSISTments is the case that matters most.** The school shift is the largest in the benchmark (baseline accuracy 0.94 to 0.58), and the shipped envelope misses it on the two label-free channels. Coverage collapses from 0.950 to 0.590; the sets do not widen, because the model is confidently wrong on the new school, so the REVIEW rate falls from 2.7% to 1.0% rather than rising (H-A4 not met); and the input-space martingale, which reads distances in a 26-feature space where the shift is a change of school identity, reaches ALARM in 2 of 5 streams (H-A3 not met; 2% over 40 supplementary streams). The one channel that sees it is the outcome verdict: realised disbelief 0.308 against a claimed 0.007. This is the regime slide 08 describes, confident errors under a shift the inputs do not show, and it is why the certificate carries a delayed outcome verdict and why no label-free signal is presented as sufficient.

**What the calibration stage is worth here.** The claimed-vs-realised
disbelief behaved as an instrument on every task: in distribution the two
agree to within 0.002; under shift the realised value exceeds the claim on
every seed of every task. The Venn-Abers bracket did not: on these large,
already-calibrated gradient-boosted models its ECE is equal or worse than the
raw model's in distribution and equal under shift. The bracket's value is the
small, miscalibrated regime of section 4.4 (German credit, ECE 0.186 to
0.065); on a calibrated scorer it is a cost, and at scale (7.2) a large one.
The product conclusion is that the calibration opinion is the outcome-drift
verdict and always runs, and the bracket is an option switched on where the
scorer needs it; as a per-decision ranking signal neither beats plain
confidence (`CALIBRATION_TRUST.md`, section 9c).

| task | raw ECE ID | Venn-Abers ECE ID | raw ECE OOD | Venn-Abers ECE OOD | disbelief claimed | realised ID | realised OOD |
|---|---|---|---|---|---|---|---|
| ACS income (finance) | 0.009 | 0.021 | 0.063 | 0.062 | 0.008 | 0.009 | 0.063 |
| ACS food stamps (public policy) | 0.008 | 0.018 | 0.015 | 0.028 | 0.007 | 0.007 | 0.015 |
| ACS unemployment (labour) | 0.003 | 0.007 | 0.004 | 0.009 | 0.001 | 0.001 | 0.002 |
| BRFSS diabetes (health) | 0.007 | 0.017 | 0.038 | 0.037 | 0.007 | 0.007 | 0.038 |
| Hospital readmission (health) | 0.016 | 0.033 | 0.041 | 0.042 | 0.014 | 0.016 | 0.040 |
| College Scorecard (education) | 0.006 | 0.012 | 0.025 | 0.024 | 0.003 | 0.005 | 0.020 |
| ASSISTments (education) | 0.006 | 0.011 | 0.309 | 0.311 | 0.007 | 0.005 | 0.308 |

**Why the readmission task breaks the tripwire, and what it means.** A
supplementary diagnostic (`eval/expansion/diag_martingale_sparse.py`, not
pre-registered, added after the fact) tests whether the conformal p-values of
held-out in-distribution rows are uniform against the calibration set's
leave-one-out distances, which the martingale's guarantee requires.

| task | cal n | binary / all features | KS p, shipped | false WATCH / ALARM, shipped | KS p, no scaler | false WATCH / ALARM, no scaler | OOD ALARM shipped / no scaler |
|---|---|---|---|---|---|---|---|
| ACS income (finance) | 20,000 | 229 / 232 | 0.171 | 0% / 0% | 0.756 | 0% / 0% | 100% / 98% |
| ACS food stamps (public policy) | 20,000 | 234 / 239 | 0.218 | 0% / 0% | 0.195 | 2% / 0% | 100% / 22% |
| ACS unemployment (labour) | 20,000 | 220 / 223 | 0.262 | 0% / 0% | 0.281 | 2% / 0% | 100% / 100% |
| BRFSS diabetes (health) | 20,000 | 137 / 142 | 0.135 | 0% / 0% | 0.938 | 8% / 2% | 100% / 98% |
| Hospital readmission (health) | 4,286 | 173 / 183 | 0.001 | 22% / 8% | 0.229 | 0% / 0% | 85% / 70% |
| College Scorecard (education) | 12,320 | 1 / 118 | 0.016 | 8% / 5% | 0.011 | 5% / 2% | 100% / 12% |
| ASSISTments (education) | 20,000 | 15 / 26 | 0.654 | 0% / 0% | 0.526 | 0% / 0% | 2% / 100% |

On five of seven tasks the shipped detector's p-values are uniform and the
false-alarm rate is at or near Ville's bound. On hospital readmission they are not (KS
p = 0.001, an excess of small p-values: held-out rows sit about 1.5% farther
from the calibration cloud than calibration rows sit from each other). The
cause is the per-column standardisation: with 4,286 calibration rows and 183
mostly binary columns, rare categories get large weights, and the
leave-one-out calibration distances are no longer exchangeable with test
distances. Dropping the standardisation restores uniformity (KS p = 0.23)
and 0% false alarms on this task, but costs detection there (OOD ALARM 85%
to 70%) and much more on ACS food stamps (100% to 22%), where the scaled
distance is what makes the region shift visible. Fitting the scaler on the
training split instead does not fix it (KS p = 0.015). College Scorecard sits at the edge (KS p = 0.016, 8% / 5% false alarms on 40 streams); there, scoring a held-out calibration half the way test rows are scored restores uniformity (KS p = 0.23). On ASSISTments the scaler does the opposite damage: with it the shipped detector sees the school shift in 2% of streams, without it in 100%, because the shift lives in a few identity columns the standardisation flattens. The credit datasets of sections 4 and 5 have 23 dense numeric features and 7,500 calibration rows and do not show any of this. The conclusion is a stated limitation: the shipped
input-space tripwire is valid on dense numeric inputs and on sparse
categorical inputs with a large calibration set; it is not valid as shipped
on sparse categorical inputs with a small one, and its per-column
standardisation can hide a shift carried by a few categorical columns. A distance that is
exchangeable by construction on such inputs is a change to `reliax_core`; it
will be pre-registered and measured, not slipped in.

### 7.4 Not run, and what section 7 does not show

Pre-specified and not run in this window: Home Credit, Credit Risk Model
Stability 2024 (native temporal drift and the weighted-conformal
recalibration recovery number; gated on a read of the Kaggle competition
terms, which restrict use to non-commercial purposes); Fannie Mae or Freddie
Mac loan performance (the 2008 macro cycle and the lag between the input
verdict and the outcome verdict; gated on the provider's terms, which
prohibit use in support of external commercial purposes without consent);
and the TableShift tasks `college_scorecard`, `assistments`, `anes`,
`mimic_extract_mort_hosp`, `mimic_extract_los_3`, which need a Kaggle token,
ANES registration or PhysioNet credentials. Each stays in the exploratory
bucket with its hypotheses fixed in Amendment 2.

Section 7 does not show anything about a lender's book, about default
outcomes outside the two UCI datasets, or about the triage rank, whose bar
is unchanged and whose confirmatory sets remain sealed. The latency figure at
a million rows is on synthetic rows of credit dimensionality, and the
sharpest new number in the section, the REVIEW-rate disparity on HMDA, is a
property of the underlying decision model's uncertainty as much as of the
envelope: the envelope measures it, it does not create it and it does not
remove it.

## References

- Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference*, 10(2).
- Gardner, J., Popovic, Z., Schmidt, L. (2023). Benchmarking distribution shift in tabular data with TableShift. *NeurIPS Datasets and Benchmarks*.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository.
- Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty*. Springer.
- Ville, J. (1939). *Étude critique de la notion de collectif*. Gauthier-Villars.
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Vovk, V., Nouretdinov, I., Gammerman, A. (2003). Testing exchangeability on-line. *ICML*.
- Vovk, V., Petej, I. (2014). Venn-Abers predictors. *UAI*.
- Vovk, V., Petej, I., Fedorova, V. (2015). Large-scale probabilistic predictors with and without guarantees. *NeurIPS*.
- Yeh, I-C., Lien, C-h. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2).

---

### Figures

![Figure 1: error capture vs referral rate](results/fig1_capture.png)
*Figure 1. Selective prediction on Taiwan default (seed 0). Confidence-linked
signals capture 2.2 to 2.3x random at a 10% referral rate; SL fusion tracks
the best curves and improves on the v1 blend; Venn-Abers width is a weak
global ranker.*

![Figure 2: per-segment coverage](results/fig2_segments.png)
*Figure 2. Per-segment empirical coverage, marginal vs Mondrian, five seeds.
No severe undercoverage on this dataset; the repair is cost-free insurance.*

![Figure 3: martingale trajectories](results/fig3_martingale.png)
*Figure 3. Exchangeability martingale on real applicants: i.i.d. streams stay
below the thresholds; streams from the delayed-payment subpopulation accumulate
evidence and cross ALARM.*

![Figure 4: calibration and width](results/fig4_calibration.png)
*Figure 4. Left: the Taiwan model is already calibrated and Venn-Abers tracks
it. Right: the widest decile of PD intervals carries a 32% relatively higher
error rate.*
