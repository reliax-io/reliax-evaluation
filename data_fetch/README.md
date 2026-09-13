# Tier 2: public data fetched by script, not redistributed

The sources here are public but large, or carry terms that allow use and
not redistribution. Nothing in this directory is data; each script fetches
into `data_fetch/cache/` (git-ignored) and records the source URL, the fetch
date and the file hashes in `data_fetch/cache/MANIFEST.json`.

| script | source | licence note |
|---|---|---|
| `fetch_tableshift.py` | TableShift tasks (`github.com/mlfoundations/tableshift`), which pull ACS via folktables, BRFSS and NHANES from CDC, College Scorecard from ed.gov, Diabetes readmission from UCI | public; per-source terms in the TableShift repo. ASSISTments is Kaggle-hosted and needs a Kaggle login. |
| `fetch_hmda.py` | HMDA Modified LAR 2025, FFIEC and CFPB | public domain (US government work) |
| `fetch_acs_tracts.py` | ACS 5-year table B03002, census-tract race and ethnicity shares (the geography-only proxy for the HMDA audit) | public domain |

The exploratory protocol for every source is fixed in
`../PREREGISTRATION_AMENDMENT.md`, Amendment 2, before any fetch.
