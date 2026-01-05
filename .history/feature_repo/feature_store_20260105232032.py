# feature_repo/feature_store.py
from pathlib import Path
import yaml
import pandas as pd
from feature_repo.feature_definition import FeatureSpec, LEAKAGE_COLS

def load_feature_spec(yaml_path: Path) -> FeatureSpec:
    cfg = yaml.safe_load(yaml_path.read_text())
    cat = cfg.get("categorical_features", [])
    num = cfg.get("numerical_features", [])

    # safety: ensure leakage never gets in
    cat = [c for c in cat if c not in LEAKAGE_COLS]
    num = [c for c in num if c not in LEAKAGE_COLS]
    return FeatureSpec(categorical=cat, numerical=num)

def select_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    cols = [c for c in spec.all_features if c in df.columns]
    return df[cols].copy()
