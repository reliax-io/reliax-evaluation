"""Fetch census-tract race and ethnicity shares (ACS 5-year, table B03002)
for the geography-only proxy of workstream C (Amendment 2, section 2.5).

Source: the Census Bureau's table-based ACS summary file (public domain, no
API key). Only tract rows (summary level 140, GEO_ID prefix 1400000US) are
kept. Output: `data_fetch/cache/acs/tracts_b03002_<vintage>.csv` with one row
per tract: the 11-digit FIPS code HMDA uses in `census_tract`, the total
population and the non-Hispanic White, Black, American Indian or Alaska
Native, Asian, Native Hawaiian or Pacific Islander counts and the Hispanic
count. This is the geocoding component of BISG only; there is no surname
component because the modified LAR carries no names.

Usage:
  .venv/bin/python data_fetch/fetch_acs_tracts.py
"""
import csv
import json
import pathlib
import time
import urllib.request

VINTAGE = 2023
URL = (f"https://www2.census.gov/programs-surveys/acs/summary_file/{VINTAGE}/table-based-SF/"
       f"data/5YRData/acsdt5y{VINTAGE}-b03002.dat")
CACHE = pathlib.Path(__file__).resolve().parent / "cache" / "acs"
COLS = {"total": "B03002_E001", "white_nh": "B03002_E003", "black_nh": "B03002_E004",
        "aian_nh": "B03002_E005", "asian_nh": "B03002_E006", "nhpi_nh": "B03002_E007",
        "hispanic": "B03002_E012"}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    raw = CACHE / f"acsdt5y{VINTAGE}-b03002.dat"
    out = CACHE / f"tracts_b03002_{VINTAGE}.csv"
    t0 = time.time()
    if not raw.exists():
        print(f"downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, raw)
    n = 0
    with open(raw, newline="", encoding="utf-8-sig") as fh, open(out, "w", newline="") as fo:
        reader = csv.DictReader(fh, delimiter="|")
        writer = csv.writer(fo)
        writer.writerow(["tract"] + list(COLS))
        for row in reader:
            if not row["GEO_ID"].startswith("1400000US"):
                continue
            writer.writerow([row["GEO_ID"][9:]] + [row[c] for c in COLS.values()])
            n += 1
    (CACHE / "MANIFEST.json").write_text(json.dumps({
        "source": URL, "vintage": VINTAGE, "tracts": n, "bytes_raw": raw.stat().st_size,
        "fetched": time.strftime("%Y-%m-%d"), "seconds": round(time.time() - t0)}, indent=1))
    print(f"{n:,} tracts written to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
