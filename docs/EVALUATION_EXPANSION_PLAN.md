# Reliax evaluation expansion plan

**Version:** v1, 13 September 2026
**Status:** plan, not results. Nothing here is claimable until it is measured.
**Owner:** CEO (method) + first engineering hire (pipeline)

---

## 0. What this is for

The current evaluation is 31,000 applicants across two public credit datasets (UCI Taiwan
30,000; UCI German 1,000). That is enough to demonstrate the theorems hold. It is not enough
to answer three questions a technical investor or a model validator will ask:

1. **Does this generalise beyond credit?** Every number in the whitepaper is a loan.
2. **What is the latency at production scale?** The 8.0 ms figure is at 7,500 calibration rows.
   The deck itself marks the 1M-row figure as pending.
3. **Can you audit parity on real protected attributes?** Slide 12 currently relies on BISG
   proxies. There is public data with actual reported race, ethnicity and sex.

This plan closes those three, plus replaces the engineered payment-delay split with a real
temporal regime change.

**What it does not do:** none of this substitutes for a shadow replay on a customer's book.
Slide 09 already says so. Dataset work is comfortable in a way that ten CRO conversations are
not: do this because slides 09 and 11 need it, not instead of the calls.

---

## 1. STEP ZERO: the pre-registration amendment (do this first, no exceptions)

**Do not download a single new file before this step is complete and committed.**

The pre-registration is described on slide 16 as "the discipline is the asset." Adding datasets
is the single most common way a pre-registration quietly dies, and a validator who catches it
will discount every number you have published, including the ones that were fine.

### 1.1 Disambiguate "Home Credit"

There are two different Home Credit competitions and the pre-registration currently names one
ambiguously:

| Name | Year | What it is | Status for you |
|---|---|---|---|
| Home Credit Default Risk | 2018 | ~307k applications, no reliable decision timestamps | **CONFIRMATORY: stays sealed** |
| Home Credit: Credit Risk Model Stability | 2024 | Base table with `case_id`, `date_decision`, `WEEK_NUM`, `MONTH`, binary `target`; designed around stability over time | Proposed **exploratory** (see §5) |

Write down which one is the frozen confirmatory set. If the original pre-registration is
genuinely ambiguous, say so in the amendment rather than picking the convenient reading.

### 1.2 Write the amendment before opening anything

Update `PREREGISTRATION_AMENDMENT.md` with, at minimum:

- Every dataset named in §2–§5 below, each tagged `exploratory` or `confirmatory`.
- For each: the hypothesis, the metric, the split rule, the seed count: written *before* the
  data is touched.
- An explicit statement that **GMSC and Home Credit Default Risk (2018) remain sealed** and are
  not used in any of this work.
- The date, and a commit hash.

### 1.3 The standing rule for everything below

> New data enters the **exploratory** bucket. It can inform the product, the whitepaper's
> exploratory section and the deck's "not yet confirmed" boxes. It cannot be used to re-tune
> the triage rank, move the gate bar, or generate a confirmatory claim.

**Definition of done:** amendment committed and pushed to
`github.com/Ouatt-Isma/reliax-evaluation` before any `wget`.

---

## 2. WORKSTREAM A: Cross-domain generalisation (TableShift)

**Priority: highest.** This is the one that fixes the "you are a credit company" problem, and
it is the cheapest of the four.

### What it is

TableShift (Gardner, Popovic & Schmidt, NeurIPS 2023 Datasets & Benchmarks) is a benchmark of
15 binary classification tasks, each paired with an associated distribution shift, spanning
finance, education, public policy, healthcare and civic participation. Tasks load through a
single Python API with identical calls, so the per-task cost after the first one is near zero.

- Repo: `github.com/mlfoundations/tableshift`
- Paper: arXiv:2312.07577
- Note: the maintainers recommend running under their Docker image; the dependency set is
  awkward outside it. Budget half a day for environment setup and do not fight it natively.

### Why it is the right instrument

It gives you a shift that *someone else defined*, on domains you did not choose, with published
baselines. Every objection to your payment-delay split ("you engineered the shift") disappears.
And it produces the sentence you currently cannot say: *coverage held at target across N domains,
not just credit.*

### What to measure (per task, both in-distribution and out-of-distribution)

| Quantity | What you are showing |
|---|---|
| Marginal coverage vs target (1−α) | The conformal theorem holds on data you did not pick |
| Per-segment (Mondrian) coverage | The per-segment claim is not a credit artefact |
| Coverage degradation ID -> OOD | Honest statement of how much validity you lose under a real shift |
| Venn-Abers calibration disbelief, claimed vs realised | The FUSION 2025 stage generalises off credit |
| Martingale ALARM rate on the OOD split, false-alarm rate on the ID split | The tripwire fires where it should and not where it shouldn't |
| REVIEW rate, ID vs OOD | Set-size routing self-adjusts under shift, as it did on Taiwan |

Run >=5 seeds. Report mean ± std, as the whitepaper already does.

### Gotchas

- Some TableShift tasks do not support domain generalisation (no multiple domains). Check the
  task table before promising all 15; pick the subset that has a usable shift and **say which
  ones you excluded and why**.
- Health tasks may carry their own access conditions. Verify per task.
- Do not cherry-pick the tasks where coverage held. Report every task you ran. If coverage
  breaks somewhere, that is a finding and it belongs in the paper: your existing credibility
  comes from publishing 0.952 -> 0.903, not from hiding it.

### Definition of done

A table in the whitepaper: one row per task, coverage ID / coverage OOD / target / ALARM /
false alarm. Plus one slide-ready sentence with a real number in it.

**What this does NOT prove:** nothing about your latency, nothing about parity, and nothing
about customer data.

---

## 3. WORKSTREAM B: Production-scale latency and real macro drift (Fannie Mae / Freddie Mac)

### What it is

Fannie Mae's Single-Family Loan Performance dataset: acquisition data plus monthly performance
for a subset of 30-year fixed-rate single-family mortgages, updated quarterly (the July 2026
release carried data through Q1 2026). Freddie Mac publishes an equivalent. Millions of loans,
real outcomes, spanning 2008: which is a genuine macro regime change, not a constructed split.

- Access: Fannie Mae Data Dynamics, free but **registration required**.
- FHFA also publishes the Enterprise Public Use Database (PUDB) annually for both GSEs.

### Licensing: read before you plan the repo

Fannie Mae's terms require accepting conditions that, among other things, **prohibit
distributing the data to third parties or using it in support of external commercial purposes
without Fannie Mae's express written consent.** Freddie Mac's terms are similar in spirit.

Consequences you must design around:

1. You **cannot** ship this data in `reliax-evaluation` the way you ship the UCI files. Your
   current claim: "code + data public and verified reproducible": has to become tiered
   (see §6).
2. "External commercial purposes" is doing real work in that sentence. A seed-stage company
   publishing a whitepaper that markets a product is close enough to the line that you should
   either (a) request written consent, or (b) get a short read from a lawyer before publishing.
   Do not decide this by reading the T&C yourself and hoping.
3. If (a) and (b) both look slow, **use it for the latency number only** and take the macro-drift
   experiment to the Home Credit stability dataset instead (§5). Latency is a property of your
   engine, not a claim about their data, which is a much easier conversation.

### What to measure

**B1: Latency at scale (the pending figure on slides 07 and 21):**

- Median and p95 per-decision envelope latency at calibration set sizes of 7,500 / 100k / 1M rows.
- Break it down by component: conformal quantile lookup, Venn-Abers query, kNN/OOD distance,
  martingale update. State which one dominates as n grows: kNN and quantile lookup both scale,
  and a validator will ask.
- Say whether you use an approximate-nearest-neighbour index, and report both with and without.
- Report the in-VPC network hop separately, as the deck promises.

**B2: Real macro drift (only if licensing clears):**

- Train on a pre-2007 vintage, test across 2008–2010. This is the "macro cycle" that slide 08
  says the live input/score verdict cannot catch and the outcome verdict must.
- Show: live martingale verdict vs outcome verdict, with the lag between them. That lag *is*
  the product argument for carrying two drift verdicts, and right now you assert it rather
  than measure it.

### Definition of done

A latency table at three scales with component breakdown, replacing the "1M-row figure pending"
caveat on slides 07 and 21 with a number.

**What this does NOT prove:** generalisation beyond credit.

---

## 4. WORKSTREAM C: Coverage parity on real protected attributes (HMDA)

### What it is

The Home Mortgage Disclosure Act Modified Loan Application Register. The 2025 data was
published on the FFIEC platform in March 2026, covering roughly 4,768 filers, and a single
combined file containing all institutions' modified LAR data is downloadable. Crucially it
carries `derived_race` and `derived_ethnicity` fields, plus sex, denial reasons and the
automated-underwriting-system name: **actual reported attributes**, not proxies.

- Platform: `ffiec.cfpb.gov/data-publication/modified-lar`
- Field reference: `ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/`
- CFPB publishes a beginner's guide; use it, the file layout has traps.

### Why this matters more than it looks

Slide 12 currently says parity by protected class is "an offline audit with BISG proxies under
the §1002.15 self-test." BISG is a proxy method with known error, and a fair-lending reviewer
knows it. Demonstrating your coverage-parity audit on *reported* race and ethnicity, then
showing what the BISG-proxy version of the same audit looks like on the same data, turns your
weakest compliance slide into a strength: you would be the only vendor in the landscape with a
measured proxy-vs-truth comparison for their own parity audit.

### The limitation you must state up front

**HMDA's outcome is an origination or denial decision, not a default.** There is no repayment
label. That means:

- You can demonstrate: coverage parity per segment, REVIEW-rate parity per protected group,
  set-size distribution by group, disparate-impact screening on the *routing decision*.
- You cannot demonstrate: PD calibration, Venn-Abers brackets on default probability, or
  anything requiring a repayment outcome.

Write that limitation into the whitepaper section before the results, not after. If a reader
has to work out the label is a decision rather than a default, you have lost them.

### What to measure

- Coverage and set size per `derived_race` / `derived_ethnicity` / sex group.
- REVIEW rate per group, with confidence intervals: this is the "thin-file routing cannot
  quietly become a disparate-impact problem" claim on slide 12, measured.
- The same audit run with BISG proxies instead of reported attributes: report the gap. That is
  a publishable finding on its own.
- Flag thin cells, as the calibration stage already does.

### Definition of done

A parity table by reported protected class, a proxy-vs-truth delta, and a rewritten slide 12
bullet that cites a measurement instead of a method.

---

## 5. WORKSTREAM D: Native temporal drift (Home Credit: Credit Risk Model Stability, 2024)

**Use this if §3's licensing path is slow, or in addition to it.**

The 2024 competition dataset was built around exactly your problem: predictions that stay
stable over time. The base table has one row per application keyed by `case_id`, with
`date_decision`, `WEEK_NUM`, `MONTH` and the binary `target`, provided in CSV and Parquet.
Temporal splits are native rather than invented, and the competition's own evaluation rewarded
stability, so there are published baselines to compare against.

### Before you touch it

Re-read §1.1. This is **not** the 2018 Home Credit Default Risk set. If your pre-registration
named "Home Credit" without a year, resolve that in the amendment first and state the reading
you adopted. Getting this wrong is the difference between a rigorous evaluation and one a
reviewer can dismiss in a sentence.

### What to measure

- Weekly/monthly rolling coverage against target across the full time index.
- Martingale wealth trajectory over calendar time, with ALARM crossings dated.
- Venn-Abers claimed vs realised disbelief, month by month.
- The weighted-conformal recalibration loop (Tibshirani et al. 2019) that slide 16 lists as the
  repair path: does coverage recover after recalibration, and by how much? **This is the single
  most valuable number in the whole plan**, because right now the repair path is a design, not
  a result.

### Gotchas

- Kaggle competition data carries competition-specific terms. Check redistribution rights
  before assuming you can ship it, same as §3.
- The feature tables are large and multi-source; feature engineering choices will dominate model
  quality. You are not competing on AUC: fix a reasonable baseline model (LightGBM with
  published preprocessing), freeze it, and evaluate the *envelope*, not the model. Say so.

---

## 6. Repository and licensing structure

Your current claim is "code + data public and verified reproducible." That survives UCI
(CC BY 4.0). It does not survive Fannie Mae or Kaggle terms. Restructure now rather than after
someone notices.

```
reliax-evaluation/
├── tier1-open/          # UCI Taiwan, UCI German: code AND data shipped (CC BY 4.0)
├── tier2-fetch/         # TableShift, HMDA: code + fetch script; data is public but large
├── tier3-gated/         # Fannie/Freddie, Kaggle: code + instructions ONLY, no data
└── PREREGISTRATION_AMENDMENT.md
```

Update the language everywhere it appears (slide 09 footnote, whitepaper, README) to something
you can defend line by line, e.g.:

> "Code is public for every experiment. Data is redistributed where the licence permits
> (UCI, CC BY 4.0), fetched by script where the source is public (TableShift, HMDA), and
> instructions-only where the provider's terms prohibit redistribution (Fannie Mae, Kaggle)."

That sentence is more credible than the current one, not less. It shows you read the licences.

---

## 7. Sequence and effort

| # | Workstream | Effort | Blocks | Why this order |
|---|---|---|---|---|
| 0 | Pre-registration amendment | 0.5 day | everything | Nothing is claimable without it |
| 1 | A: TableShift | 4–6 days |: | Highest leverage, cheapest, fixes the credit-focus problem |
| 2 | C: HMDA parity | 3–4 days |: | Closes the fairness gap flagged as KEY FINDING 3 |
| 3 | D: Home Credit stability | 3 days | §1.1 | Delivers the recalibration-recovery number |
| 4 | B1: Latency at 1M | 2–3 days |: | Pure engineering, no licensing risk |
| 5 | B2: GSE macro drift | 4–5 days | legal read | Do last; the only one gated on someone else |
| 6 | Repo restructure + whitepaper v0.3 | 3 days | 1–5 |: |

Roughly three to four weeks of focused work. **Run it alongside the CRO conversations, not
before them.**

---

## 8. What changes in the deck when this lands

| Slide | Current | After |
|---|---|---|
| 01 hero | "measured on two public UCI credit datasets, 31,000 applicants" | Coverage at target across N domains; keep the credit numbers as the headline |
| 07 / 21 | "The 1M-row figure … is the next measurement, not yet published" | An actual table at 7.5k / 100k / 1M with component breakdown |
| 09 evidence | "Measured on two public credit datasets" | Retitle: the theorems are not credit-specific, the evidence now spans N domains |
| 12 regulation | "Parity … with BISG proxies" | Parity measured on reported protected class, plus the proxy-vs-truth delta |
| 13 competition | "Evaluation released on real public data: YES" | Reframe: this is parity with MAPIE and White Circle, not a win. Change the row to the *guarantee* being verified, not the evaluation being released |
| 16 the gate | Repair path described as design | Measured coverage recovery after weighted recalibration |

---

## 9. Rules to hold yourself to

1. **Report every task you ran.** Excluding a dataset because coverage broke there is the one
   thing that would destroy the discipline asset. A broken result stated plainly is worth more
   than five clean ones with a selection process nobody can see.
2. **Exploratory stays exploratory.** None of this touches the triage-rank gate, moves the
   1.5× bar, or opens GMSC.
3. **State the label.** HMDA is a decision, not a default. TableShift tasks are not credit.
   Fannie is US mortgage, not EU consumer credit. Each of those is fine; each is fatal if the
   reader discovers it themselves.
4. **Licences before downloads.** Two of the five sources restrict redistribution. Decide the
   repo tier before you have 40 GB on a laptop and a sunk-cost argument.
5. **The real proof is still a customer replay.** If at day 30 you have four new datasets and
   zero CRO conversations, this plan has made things worse, not better.

---

## Sources

- TableShift: Gardner, Popovic & Schmidt, *Benchmarking Distribution Shift in Tabular Data
  with TableShift*, NeurIPS 2023 Datasets & Benchmarks (arXiv:2312.07577);
  `github.com/mlfoundations/tableshift`
- Fannie Mae Single-Family Loan Performance Data, Data Dynamics (registration and terms
  acceptance required; Q1 2026 data released 31 July 2026)
- FHFA Enterprise Public Use Database (Fannie Mae and Freddie Mac single-family acquisitions)
- HMDA Modified LAR, FFIEC/CFPB: 2025 data published 31 March 2026, ~4,768 filers; field
  documentation at `ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/`
- Home Credit: Credit Risk Model Stability, Kaggle 2024 (base table with `case_id`,
  `date_decision`, `WEEK_NUM`, `MONTH`, `target`)
- Weighted conformal under covariate shift: Tibshirani, Barber, Candès & Ramdas (2019)

*Licence terms summarised above were read from vendor documentation in September 2026 and are
not legal advice. Confirm current terms before publishing, and get a lawyer's read on the GSE
question specifically.*
