# TableShift environment (native, no Docker)

TableShift (`github.com/mlfoundations/tableshift`, commit fca9429) pins a
2023 dependency set. Its Docker image is the supported route; the Docker
daemon was not available here, so this is the native recipe that worked on
macOS arm64 with Python 3.10:

```bash
uv venv --python 3.10 .venv-tableshift
uv pip install --python .venv-tableshift/bin/python \
  "numpy<2" "pandas==1.5.3" "scikit-learn==1.1.3" "ray==2.9.3" "pyarrow<15" \
  "datasets==2.11.0" "fsspec<2023.10" "setuptools<70" torch folktables requests \
  xport fairlearn frozendict category_encoders tqdm h5py tables openpyxl hyperopt rtdl einops
git clone https://github.com/mlfoundations/tableshift.git && cd tableshift && git checkout fca9429
git apply ../data_fetch/tableshift_passthrough.patch
uv pip install --python ../.venv-tableshift/bin/python --no-deps -e .
```

`tableshift_passthrough.patch` is a one-line fix: `Preprocessor.fit_transform`
calls `get_passthrough_columns` without forwarding `domain_label_colname`,
so a categorical domain column (ACS `DIVISION`, `SCHL`) is one-hot encoded
and the split then fails with `KeyError`. The patch forwards the argument. It
changes nothing about the data or the splits.

Sources that need credentials and were not fetched: `college_scorecard` and
`assistments` (Kaggle API token at `~/.kaggle/kaggle.json`), `anes` (ANES
registration), `mimic_extract_*` (PhysioNet credentialed access). BRFSS is
parsed from 1 GB SAS transport files by the pure-Python `xport` reader,
about 20 minutes per survey year.

## Kaggle-hosted tasks (added 14 Sep 2026)

TableShift shells out to the `kaggle` CLI. Kaggle's current tokens
(`KGAT_...`) need CLI 2.x, which needs Python 3.11+, so install it as a
standalone tool and export the token:

```bash
uv tool install --python 3.12 "kaggle>=1.8"
export KAGGLE_API_TOKEN="$(tr -d '\n' < ~/.kaggle/kaggle.json)"   # file holds the bare token
PATH="$HOME/.local/bin:$PATH" .venv-tableshift/bin/python data_fetch/fetch_tableshift.py college_scorecard assistments
```

`assistments`: TableShift's loader reads `assistments-subset.feather`, a
column subset the authors never shipped (the conversion is commented out in
`data_source.py`). Build it once from the downloaded zip, exactly as their
comment describes (select `ASSISTMENTS_FEATURES.names` from
`2012-2013-data-with-predictions-4-final.csv`, write feather); the fetch
script then runs unchanged. Installing `kaggle` into `.venv-tableshift`
upgrades setuptools past ray's pin; re-run `uv pip install "setuptools<70"`.
