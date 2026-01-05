# src/data/load_local.py
from pathlib import Path
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)

def load_cereal_csv(path: Path) -> pd.DataFrame:
    log.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded shape={df.shape}")
    return df
