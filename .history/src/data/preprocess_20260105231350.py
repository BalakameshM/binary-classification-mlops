# src/data/preprocess.py
import re
import numpy as np
import pandas as pd

NUM_COLS = ["calories","protein","fat","sodium","fiber","carbo","sugars","potass","vitamins","weight","cups"]

def clean_cereal_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.lower().strip().replace("-", " ")
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return re.sub(r"\s+", " ", name)

def health_class(row: pd.Series) -> int:
    # 2: Healthy, 1: Moderately healthy, 0: Unhealthy
    if row["fiber"] >= 7 and row["sugars"] <= 5 and row["calories"] <= 110:
        return 2
    if row["fiber"] >= 3 and row["sugars"] <= 10:
        return 1
    return 0

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "name" in df.columns:
        df["name"] = df["name"].apply(clean_cereal_name)

    # invalid negatives -> NaN
    existing_num = [c for c in NUM_COLS if c in df.columns]
    if existing_num:
        df[existing_num] = df[existing_num].mask(df[existing_num] < 0, np.nan)

    # create target
    if "target" not in df.columns:
        df["target"] = df.apply(health_class, axis=1)

    # leakage handling (since target derived from these)
    leakage_cols = [c for c in ["fiber","sugars","calories"] if c in df.columns]
    drop_cols = [c for c in ["name","rating"] if c in df.columns] + leakage_cols
    df = df.drop(columns=drop_cols, errors="ignore")

    return df
