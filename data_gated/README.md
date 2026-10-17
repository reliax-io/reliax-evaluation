# Tier 3: gated data, instructions only

These providers' terms prohibit redistribution, and in two cases restrict
use to non-commercial purposes. This directory holds instructions and loader
code only. No data, no derived tables, no cached files are committed. The
workstreams that depend on them are gated on a written licence read
(Amendment 2, sections 2.6 and 2.7) and are reported as "not run" if the
gate does not clear.

## Fannie Mae Single-Family Loan Performance Data (workstream B2)

1. Register at Fannie Mae Data Dynamics and accept the terms.
2. Download the acquisition and performance files for 2005 Q1 through 2010 Q4.
3. Place them under `data_gated/cache/fannie/` (git-ignored).
4. The runner `eval/expansion/run_gse_drift.py` is written and tested only
   once the licence gate in Amendment 2, section 2.7, has cleared; the
   protocol it must implement is fixed there.

Freddie Mac's Single-Family Loan-Level Dataset is the alternative source
with equivalent terms; the loader accepts either layout.

## Home Credit, Credit Risk Model Stability (Kaggle, 2024; workstream D)

This is not the sealed 2018 "Home Credit Default Risk" release. See
Amendment 2, section 2.1.

1. Accept the competition rules on Kaggle.
2. `kaggle competitions download -c home-credit-credit-risk-model-stability`
3. Unpack under `data_gated/cache/homecredit2024/` (git-ignored).
4. The runner `eval/expansion/run_homecredit_stability.py` is written and
   tested only once the licence gate in Amendment 2, section 2.6, has
   cleared; the protocol it must implement is fixed there.

## Sealed confirmatory sets (never placed here)

Give Me Some Credit (2011) and Home Credit Default Risk (2018) are the
confirmatory sets of Amendment 1, accepted 17 October 2026 (see the end of
PREREGISTRATION_AMENDMENT.md). They are opened only by the confirmatory runs:

1. Accept the rules of both Kaggle competitions (GiveMeSomeCredit,
   home-credit-default-risk) on the account whose API key is used.
2. `.venv/bin/pip install kaggle`, with `KAGGLE_USERNAME` and `KAGGLE_KEY` set.
3. `.venv/bin/python data_gated/fetch_confirmatory.py` writes
   `cache/gmsc/cs-training.csv`, `cache/homecredit2018/application_train.csv`,
   `cache/homecredit2018/bureau.csv` and their checksums.
4. `.venv/bin/python eval/run_confirmatory.py` writes `results/confirmatory/`.
