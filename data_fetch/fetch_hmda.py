"""Fetch the HMDA 2025 loan-level data (public, US government work) for
workstream C, state by state, from the FFIEC/CFPB HMDA data browser API.

Server-side filters (the API accepts at most two): action_taken in {1, 3}
and loan_purpose in {1, 31, 32}. The remaining population filters of
Amendment 2, section 2.5 (first lien, site-built, 1 to 4 units, principal
residence, non-reverse, non-business) are applied in the runner. One gzipped
CSV per state is written to `data_fetch/cache/hmda/2025/<state>.csv.gz` and
`MANIFEST.json` records rows, bytes and the fetch date. Nothing is
redistributed from the repository.

Usage:
  .venv/bin/python data_fetch/fetch_hmda.py            # all states
  .venv/bin/python data_fetch/fetch_hmda.py DE VT      # some states
"""
import gzip
import json
import pathlib
import sys
import time
import urllib.request

YEAR = 2025
CACHE = pathlib.Path(__file__).resolve().parent / "cache" / "hmda" / str(YEAR)
API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
FILTERS = {"actions_taken": "1,3", "loan_purposes": "1,31,32"}
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
          "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
          "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
          "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR"]


def fetch_state(state: str, attempts: int = 4) -> dict:
    url = f"{API}?years={YEAR}&states={state}&" + "&".join(f"{k}={v}" for k, v in FILTERS.items())
    out = CACHE / f"{state}.csv.gz"
    for attempt in range(1, attempts + 1):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers={"Accept-Encoding": "identity",
                                                       "User-Agent": "reliax-evaluation fetch"})
            with urllib.request.urlopen(req, timeout=1800) as resp:
                raw = resp.read()
            text = raw.decode("utf-8")
            if not text.startswith("activity_year"):
                raise RuntimeError(f"unexpected response: {text[:120]!r}")
            with gzip.open(out, "wt", encoding="utf-8") as fh:
                fh.write(text)
            return {"status": "fetched", "rows": text.count("\n") - 1, "bytes": len(raw),
                    "seconds": round(time.time() - t0), "url": url,
                    "fetched": time.strftime("%Y-%m-%d")}
        except Exception as exc:                      # noqa: BLE001
            if attempt == attempts:
                return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300],
                        "url": url, "fetched": time.strftime("%Y-%m-%d")}
            time.sleep(30 * attempt)


def main(states):
    CACHE.mkdir(parents=True, exist_ok=True)
    mpath = CACHE / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for st in states:
        if manifest.get(st, {}).get("status") == "fetched" and (CACHE / f"{st}.csv.gz").exists():
            print(f"{st}: cached ({manifest[st]['rows']} rows)", flush=True)
            continue
        m = fetch_state(st)
        manifest[st] = m
        mpath.write_text(json.dumps(manifest, indent=1))
        print(f"{st}: {m['status']} " + (f"{m['rows']} rows in {m['seconds']}s" if m["status"] == "fetched"
                                          else m["error"]), flush=True)
    total = sum(m.get("rows", 0) for m in manifest.values())
    print(f"total rows fetched: {total:,}")


if __name__ == "__main__":
    main(sys.argv[1:] or STATES)
