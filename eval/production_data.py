"""Loaders for the production-model track (Amendment 3, exploratory).

Same datasets, populations and corruption as eval/confirmatory_data.py; what
changes is what the model may read. Each loader returns the inputs as floats
(categorical columns as integer codes, missing as NaN), a mask of the
categorical columns (the model treats them as categories; the kNN distance
leaves them out) and the declared populations.

- Give Me Some Credit: the same ten columns less age (audit-only); nothing to
  add, the competition has one table.
- Home Credit Default Risk 2018: every column of application_train.csv except
  the identifiers, the label and the audit-only attributes (CODE_GENDER,
  DAYS_BIRTH, NAME_FAMILY_STATUS); categorical columns as categories;
  DAYS_EMPLOYED = 365243 (the release's "not employed" code) as missing; the
  ratios found in most public solutions (credit / income, annuity / income,
  annuity / credit, goods / credit, the mean of EXT_SOURCE_1-3); and per-applicant
  aggregates of bureau.csv (record and active counts; sum, mean and max of the
  amount, debt and overdue amounts; max days overdue; prolongations; age of the
  oldest and newest bureau credit). The shifted population is still "any late
  payment in bureau data", which these aggregates now let the model see.
- Taiwan: the same 20 columns as before.
"""
import numpy as np
import pandas as pd

import confirmatory_data as CD

HC_RATIOS = {"CREDIT_INCOME": ("AMT_CREDIT", "AMT_INCOME_TOTAL"), "ANNUITY_INCOME": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
             "ANNUITY_CREDIT": ("AMT_ANNUITY", "AMT_CREDIT"), "GOODS_CREDIT": ("AMT_GOODS_PRICE", "AMT_CREDIT")}


def _with_masks(ds, cat_cols=()):
    names = ds["feature_names"]
    cat = np.array([n in cat_cols for n in names])
    ds["categorical"] = cat
    return ds


def gmsc_frame(train: pd.DataFrame) -> dict:
    return _with_masks(CD.gmsc_frame(train))


def bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    b = bureau.copy()
    b["ACTIVE"] = (b["CREDIT_ACTIVE"].astype(str) == "Active").astype(float)
    g = b.groupby("SK_ID_CURR")
    agg = pd.DataFrame({
        "BUREAU_N": g.size(),
        "BUREAU_ACTIVE_N": g["ACTIVE"].sum(),
        "BUREAU_DAYS_CREDIT_MIN": g["DAYS_CREDIT"].min(),
        "BUREAU_DAYS_CREDIT_MAX": g["DAYS_CREDIT"].max(),
        "BUREAU_CREDIT_SUM_SUM": g["AMT_CREDIT_SUM"].sum(),
        "BUREAU_CREDIT_SUM_MEAN": g["AMT_CREDIT_SUM"].mean(),
        "BUREAU_DEBT_SUM": g["AMT_CREDIT_SUM_DEBT"].sum(),
        "BUREAU_DEBT_MAX": g["AMT_CREDIT_SUM_DEBT"].max(),
        "BUREAU_OVERDUE_SUM": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BUREAU_MAX_OVERDUE_MAX": g["AMT_CREDIT_MAX_OVERDUE"].max(),
        "BUREAU_DAY_OVERDUE_MAX": g["CREDIT_DAY_OVERDUE"].max(),
        "BUREAU_PROLONG_SUM": g["CNT_CREDIT_PROLONG"].sum(),
    })
    agg["BUREAU_DEBT_CREDIT"] = agg["BUREAU_DEBT_SUM"] / agg["BUREAU_CREDIT_SUM_SUM"].replace(0, np.nan)
    return agg


def homecredit_frame(app: pd.DataFrame, bureau: pd.DataFrame) -> dict:
    base = CD.homecredit_frame(app, bureau[["SK_ID_CURR", *CD.HC_OVERDUE]])     # populations, label, audit, as declared
    a = app.drop(columns=["TARGET", *CD.HC_AUDIT], errors="ignore").copy()
    a["DAYS_EMPLOYED"] = a["DAYS_EMPLOYED"].replace(365243, np.nan)
    for k, (num, den) in HC_RATIOS.items():
        a[k] = a[num] / a[den].replace(0, np.nan)
    a["EXT_SOURCE_MEAN"] = a[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    a = a.merge(bureau_features(bureau), left_on="SK_ID_CURR", right_index=True, how="left")
    a = a.drop(columns=["SK_ID_CURR"])
    cats = [c for c in a.columns if not pd.api.types.is_numeric_dtype(a[c])]
    for c in cats:
        a[c] = a[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    names = list(a.columns)
    X = a.to_numpy(dtype=float)
    money = [names.index(c) for c in CD.HC_MONEY]

    def corrupt(Xr, rows):                       # the ratios are money / money: unchanged by a x100 on all four
        Xr = Xr.copy()
        Xr[np.ix_(rows, money)] *= 100.0
        return Xr

    ds = {**base, "X": X, "feature_names": names, "corrupt": corrupt}
    return _with_masks(ds, cats)


def load_gmsc():
    return gmsc_frame(pd.read_csv(CD.GMSC_DIR / "cs-training.csv"))


def load_homecredit2018():
    app = pd.read_csv(CD.HOMECREDIT_DIR / "application_train.csv")
    bureau = pd.read_csv(CD.HOMECREDIT_DIR / "bureau.csv")
    return homecredit_frame(app, bureau)


def load_taiwan():
    return _with_masks(CD.load_taiwan())


LOADERS = {"gmsc": load_gmsc, "homecredit2018": load_homecredit2018, "taiwan": load_taiwan}
