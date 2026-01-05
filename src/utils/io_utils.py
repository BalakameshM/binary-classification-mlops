# src/utils/io_utils.py
from pathlib import Path
import json
import joblib
import pandas as pd

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))

def read_json(path: Path) -> dict:
    return json.loads(path.read_text())

def save_joblib(path: Path, obj) -> None:
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_joblib(path: Path):
    return joblib.load(path)

def save_df(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() in [".parquet"]:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".parquet"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)
