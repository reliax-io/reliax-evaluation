"""Loaders for the confirmatory runs of Amendment 1 (PREREGISTRATION_AMENDMENT.md).

Each loader returns the raw inputs (missing cells kept as NaN), the label, the
feature names, the declared reference and shifted populations, the declared
monetary fields and the audit-only attributes. Nothing here is fitted: the
imputation is fitted per seed on the reference training split, in the runner.

The operational readings below were fixed when Amendment 1 was accepted, before
either confirmatory set was downloaded (see "Acceptance" in the amendment).

- Give Me Some Credit (Kaggle, 2011): cs-training.csv, 150,000 borrowers,
  label SeriousDlqin2yrs. Reference: NumberOfTime30-59DaysPastDueNotWorse = 0;
  shifted: >= 1. Monetary fields: MonthlyIncome, and the monthly debt
  DebtRatio x MonthlyIncome. A x100 unit error multiplies both, so the model
  sees MonthlyIncome x100 and an unchanged DebtRatio; where MonthlyIncome is
  missing, DebtRatio holds the debt amount itself and is multiplied by 100.
  age is audit-only (not a model input), as SEX, AGE and MARRIAGE are on Taiwan.

- Home Credit Default Risk (Kaggle, 2018): application_train.csv (label TARGET)
  and bureau.csv. Shifted: any bureau record of the applicant with
  CREDIT_DAY_OVERDUE > 0, AMT_CREDIT_MAX_OVERDUE > 0 or AMT_CREDIT_SUM_OVERDUE
  > 0; reference: every other applicant, including those with no bureau record
  ("no previous late payment in bureau data", read literally). Model inputs:
  the numeric columns of application_train.csv; the categorical columns are not
  encoded (one-hot inputs break the shipped kNN distance, paper section 7.3),
  and bureau.csv defines the populations only. CODE_GENDER, DAYS_BIRTH and
  NAME_FAMILY_STATUS are audit-only. Monetary fields: AMT_INCOME_TOTAL,
  AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE.

- Taiwan (UCI 350, exploratory in Amendment 1): reference PAY_0 <= 0, shifted
  PAY_0 >= 1, as in the v1 and v2 benchmarks; monetary fields LIMIT_BAL,
  BILL_AMT*, PAY_AMT*.
"""
import pathlib

import numpy as np
import pandas as pd

import data as D

BASE = pathlib.Path(__file__).resolve().parent.parent
GATED = BASE / "data_gated" / "cache"
GMSC_DIR = GATED / "gmsc"
HOMECREDIT_DIR = GATED / "homecredit2018"

GMSC_LABEL = "SeriousDlqin2yrs"
GMSC_SHIFT = "NumberOfTime30-59DaysPastDueNotWorse"
HC_MONEY = ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE")
HC_AUDIT = ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS")
HC_OVERDUE = ("CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM_OVERDUE")


def _scale(X, rows, cols, factor=100.0):
    X[np.ix_(rows, cols)] *= factor
    return X


def gmsc_frame(train: pd.DataFrame) -> dict:
    df = train.drop(columns=[c for c in train.columns if c.startswith("Unnamed") or c == ""], errors="ignore")
    y = df[GMSC_LABEL].to_numpy().astype(int)
    age = df["age"].to_numpy()
    feats = df.drop(columns=[GMSC_LABEL, "age"])
    names = list(feats.columns)
    X = feats.to_numpy(dtype=float)
    s = names.index(GMSC_SHIFT)
    inc, dr = names.index("MonthlyIncome"), names.index("DebtRatio")

    def corrupt(Xr, rows):
        Xr = Xr.copy()
        has_income = np.isfinite(Xr[rows, inc])
        Xr[rows[has_income], inc] *= 100.0           # income x100, debt x100: the ratio is unchanged
        Xr[rows[~has_income], dr] *= 100.0           # no income: DebtRatio holds the debt amount
        return Xr

    return {"name": "gmsc", "title": "Give Me Some Credit (Kaggle, 2011)", "status": "confirmatory",
            "X": X, "y": y, "feature_names": names,
            "reference": X[:, s] == 0, "shifted": X[:, s] >= 1,
            "monetary": ["MonthlyIncome", "DebtRatio x MonthlyIncome"], "corrupt": corrupt,
            "audit": {"age_band": np.select([age <= 30, age <= 45], ["<=30", "31-45"], default="46+")}}


def load_gmsc() -> dict:
    return gmsc_frame(pd.read_csv(GMSC_DIR / "cs-training.csv"))


def homecredit_frame(app: pd.DataFrame, bureau: pd.DataFrame) -> dict:
    late = bureau[list(HC_OVERDUE)].fillna(0.0).gt(0).any(axis=1)
    late_ids = set(bureau.loc[late, "SK_ID_CURR"].unique())
    shifted = app["SK_ID_CURR"].isin(late_ids).to_numpy()
    y = app["TARGET"].to_numpy().astype(int)
    audit = {"sex": app["CODE_GENDER"].astype(str).to_numpy(),
             "age_band": np.select([app["DAYS_BIRTH"] >= -30 * 365.25, app["DAYS_BIRTH"] >= -45 * 365.25],
                                   ["<=30", "31-45"], default="46+")}
    feats = app.drop(columns=["SK_ID_CURR", "TARGET", *HC_AUDIT], errors="ignore").select_dtypes(include=[np.number])
    names = list(feats.columns)
    X = feats.to_numpy(dtype=float)
    money = [names.index(c) for c in HC_MONEY]

    def corrupt(Xr, rows):
        return _scale(Xr.copy(), rows, money)       # NaN stays NaN

    return {"name": "homecredit2018", "title": "Home Credit Default Risk (Kaggle, 2018)", "status": "confirmatory",
            "X": X, "y": y, "feature_names": names,
            "reference": ~shifted, "shifted": shifted,
            "monetary": list(HC_MONEY), "corrupt": corrupt, "audit": audit,
            "n_with_bureau": int(app["SK_ID_CURR"].isin(set(bureau["SK_ID_CURR"].unique())).sum())}


def load_homecredit2018() -> dict:
    app = pd.read_csv(HOMECREDIT_DIR / "application_train.csv")
    bureau = pd.read_csv(HOMECREDIT_DIR / "bureau.csv", usecols=["SK_ID_CURR", *HC_OVERDUE])
    return homecredit_frame(app, bureau)


def load_taiwan() -> dict:
    ds = D.load_taiwan()
    X, names = ds["X"], ds["feature_names"]
    p0 = names.index("PAY_0")
    money = [i for i, f in enumerate(names) if f.startswith(("LIMIT_BAL", "BILL_AMT", "PAY_AMT"))]

    def corrupt(Xr, rows):
        return _scale(Xr.copy(), rows, money)

    return {"name": "taiwan", "title": "Default of Credit Card Clients (Taiwan, UCI 350)", "status": "exploratory",
            "X": X, "y": ds["y"], "feature_names": names,
            "reference": X[:, p0] <= 0, "shifted": X[:, p0] >= 1,
            "monetary": ["LIMIT_BAL", "BILL_AMT1-6", "PAY_AMT1-6"], "corrupt": corrupt, "audit": ds["segments"]}


LOADERS = {"gmsc": load_gmsc, "homecredit2018": load_homecredit2018, "taiwan": load_taiwan}
