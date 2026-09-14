"""Fetch TableShift tasks and export their splits for the frozen evaluation code.

Runs in the TableShift environment (Python 3.10, `.venv-tableshift`), which is
separate from the evaluation environment because TableShift pins a 2023
dependency set. Exports each task to `data_fetch/cache/tableshift/<task>.npz`
with the four splits TableShift defines (train, validation, id_test, ood_test),
the sensitive attributes it returns for the audit segments, and the domain
label, plus a manifest with row counts, column counts and the fetch date.

Nothing is modified: the domain split is TableShift's own. The evaluation
protocol is fixed in PREREGISTRATION_AMENDMENT.md, Amendment 2, section 2.4.

Usage (from the repository root):
  .venv-tableshift/bin/python data_fetch/fetch_tableshift.py            # all tasks
  .venv-tableshift/bin/python data_fetch/fetch_tableshift.py acsincome  # one task
"""
import json
import pathlib
import sys
import time

import numpy as np

CACHE = pathlib.Path(__file__).resolve().parent / "cache" / "tableshift"
SPLITS = ("train", "validation", "id_test", "ood_test")

# The ten TableShift benchmark tasks with a domain split (README table), as
# declared in Amendment 2. The five tasks without a domain split are not
# fetched: there is no OOD split to evaluate.
TASKS = [
    "diabetes_readmission", "college_scorecard", "brfss_diabetes",
    "acsincome", "acsfoodstamps", "acsunemployment", "assistments",
    "anes", "mimic_extract_mort_hosp", "mimic_extract_los_3",
]


# ANES: TableShift hard-codes the 16 Sep 2022 release of the Time Series
# Cumulative Data File. The release used here is the one on disk; the loader's
# resource path is pointed at it. Variables TableShift reads (VCF0004, VCF0112,
# VCF0702, VCF0901b) are stable across releases. Recorded in the manifest.
def _anes_release():
    hits = sorted(CACHE.glob("anes_timeseries_cdf_csv_*/anes_timeseries_cdf_csv_*.csv"))
    return hits[-1] if hits else None


def export(task: str) -> dict:
    from tableshift import get_dataset
    t0 = time.time()
    kwargs = {}
    if task == "anes":
        csv = _anes_release()
        if csv is not None:
            import tableshift.core.data_source as ds
            ds.ANESDataSource.__init__.__defaults__ = tuple(
                (str(csv.relative_to(CACHE)),) if d == ("anes_timeseries_cdf_csv_20220916/anes_timeseries_cdf_csv_20220916.csv",) else d
                for d in ds.ANESDataSource.__init__.__defaults__)
    dset = get_dataset(task, cache_dir=str(CACHE), **kwargs)
    arrays, meta = {}, {"task": task, "splits": {}}
    columns = None
    for split in SPLITS:
        X, y, G, dom = dset.get_pandas(split)
        if columns is None:
            columns = list(X.columns)
            meta["columns"] = columns
            meta["segment_attributes"] = list(G.columns)
            meta["domain_attribute"] = getattr(dom, "name", None)
        assert list(X.columns) == columns, f"{task}/{split}: column mismatch"
        # int8 one-hot columns and float columns stored separately to keep
        # the export small; the runner concatenates after subsampling.
        is_float = np.array([str(d).startswith("float") for d in X.dtypes])
        arrays[f"{split}/X_int"] = X.loc[:, ~is_float].to_numpy(dtype=np.int16)
        arrays[f"{split}/X_float"] = X.loc[:, is_float].to_numpy(dtype=np.float32)
        arrays[f"{split}/y"] = y.to_numpy().astype(np.int8)
        for col in G.columns:
            arrays[f"{split}/G/{col}"] = G[col].astype(str).to_numpy().astype("U")
        arrays[f"{split}/domain"] = dom.astype(str).to_numpy().astype("U")
        meta["splits"][split] = {"n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
                                 "domains": sorted(set(dom.astype(str)))[:50]}
    meta["int_columns"] = [c for c, f in zip(columns, is_float) if not f]
    meta["float_columns"] = [c for c, f in zip(columns, is_float) if f]
    meta["n_features"] = len(columns)
    meta["fetched"] = time.strftime("%Y-%m-%d")
    if task == "anes" and _anes_release() is not None:
        meta["release_file"] = _anes_release().name
    meta["seconds"] = round(time.time() - t0)
    np.savez_compressed(CACHE / f"{task}.npz", **arrays)
    (CACHE / f"{task}.meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(tasks):
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for task in tasks:
        print(f"=== {task}", flush=True)
        try:
            meta = export(task)
            manifest[task] = {"status": "exported", **{k: meta[k] for k in
                              ("splits", "n_features", "segment_attributes", "domain_attribute",
                               "fetched", "seconds")}}
            print(f"  ok: {meta['n_features']} features, "
                  + ", ".join(f"{s}={v['n']}" for s, v in meta["splits"].items())
                  + f" ({meta['seconds']}s)", flush=True)
        except Exception as exc:                      # noqa: BLE001 - reported, not hidden
            manifest[task] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500],
                              "fetched": time.strftime("%Y-%m-%d")}
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or TASKS)
