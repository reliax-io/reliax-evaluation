"""Download the two confirmatory sets of Amendment 1 into data_gated/cache/.

Run this only for the confirmatory runs (PREREGISTRATION_AMENDMENT.md,
Amendment 1 and its acceptance). Needs a Kaggle account that has accepted the
rules of both competitions, and the Kaggle API credentials in the environment
(KAGGLE_USERNAME and KAGGLE_KEY, or ~/.kaggle/kaggle.json).

    .venv/bin/pip install kaggle
    .venv/bin/python data_gated/fetch_confirmatory.py

Writes
    data_gated/cache/gmsc/cs-training.csv
    data_gated/cache/homecredit2018/application_train.csv
    data_gated/cache/homecredit2018/bureau.csv
and data_gated/cache/confirmatory_sha256.txt, the checksums of the files the
run read. The cache is git-ignored; nothing here is committed or redistributed.
"""
import hashlib
import pathlib
import zipfile

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
FILES = {
    "gmsc": ("GiveMeSomeCredit", ["cs-training.csv"]),
    "homecredit2018": ("home-credit-default-risk", ["application_train.csv", "bureau.csv"]),
}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    lines = []
    for folder, (competition, names) in FILES.items():
        out = CACHE / folder
        out.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = out / name
            if not target.exists():
                api.competition_download_file(competition, name, path=str(out), quiet=False)
                for z in out.glob(name + "*.zip"):
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(out)
                    z.unlink()
            lines.append(f"{sha256(target)}  {folder}/{name}")
            print(lines[-1])
    (CACHE / "confirmatory_sha256.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
