"""Real-dataset loaders for the ReliaX evaluation.

Both datasets are public UCI datasets, stored verbatim in paper/data/:

- Taiwan "Default of Credit Card Clients" (Yeh & Lien, 2009; UCI id 350).
  30,000 credit-card holders, binary default-next-month label (22.1% positive).
  Protected attributes SEX, AGE, MARRIAGE are EXCLUDED from the model features
  and used only for the per-segment audits (the ReliaX audit-only design).

- German Credit (Statlog; UCI id 144). 1,000 applicants, 30% bad. Categorical
  attributes one-hot encoded; personal-status/sex (attr 9) and age (attr 13)
  excluded from features and used for audits. Small n: reported with caveats.
"""
import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def load_taiwan():
    df = pd.read_excel(DATA_DIR / "default of credit card clients.xls", header=1)
    y = df["default payment next month"].to_numpy().astype(int)
    sex = np.where(df["SEX"].to_numpy() == 1, "male", "female")
    age = df["AGE"].to_numpy()
    age_band = np.select([age <= 30, age <= 45], ["21-30", "31-45"], default="46+")
    drop = ["ID", "default payment next month", "SEX", "AGE", "MARRIAGE"]
    feats = df.drop(columns=drop)
    return {
        "name": "taiwan",
        "X": feats.to_numpy(dtype=float),
        "y": y,
        "feature_names": list(feats.columns),
        "segments": {"sex": sex, "age_band": age_band},
        "classes": ["repay", "default"],
    }


GERMAN_SEX = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}


def load_german():
    cols = [f"a{i}" for i in range(1, 21)] + ["label"]
    df = pd.read_csv(DATA_DIR / "german.data", sep=" ", header=None, names=cols)
    y = (df["label"] == 2).astype(int).to_numpy()          # 2 = bad credit
    sex = df["a9"].map(GERMAN_SEX).to_numpy()
    age = df["a13"].to_numpy()
    age_band = np.where(age <= 35, "<=35", ">35")
    feats = df.drop(columns=["label", "a9", "a13"])
    feats = pd.get_dummies(feats, dtype=float)             # one-hot categoricals
    return {
        "name": "german",
        "X": feats.to_numpy(dtype=float),
        "y": y,
        "feature_names": list(feats.columns),
        "segments": {"sex": sex, "age_band": age_band},
        "classes": ["good", "bad"],
    }
