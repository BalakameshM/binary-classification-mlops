# tests/conftest.py
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is importable (so "import src" works)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def raw_csv_path() -> Path:
    """
    Path to raw cereal dataset in repo.
    """
    p = PROJECT_ROOT / "data" / "raw" / "cereal.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset at: {p}")
    return p


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Small dataframe that mimics the cereal schema (enough for preprocess tests).
    Includes leakage columns + name/rating so tests can verify they are dropped.
    """
    return pd.DataFrame({
        "name": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "mfr": ["K", "K", "G", "G", "N", "N"],
        "type": ["C", "C", "C", "C", "C", "C"],
        "shelf": [1, 2, 1, 2, 3, 3],
        "calories": [110, 100, 120, 90, 105, 130],
        "protein": [3, 2, 4, 1, 2, 3],
        "fat": [1, 1, 2, 1, 1, 2],
        "sodium": [100, 120, 130, 90, 110, 140],
        "fiber": [7, 3, 8, 2, 4, 1],
        "carbo": [10, 12, 11, 9, 13, 14],
        "sugars": [5, 10, 4, 12, 8, 15],
        "potass": [80, 60, 90, 50, 70, 100],
        "vitamins": [25, 25, 0, 25, 0, 25],
        "weight": [1, 1, 1, 1, 1, 1],
        "cups": [1, 1, 1, 1, 1, 1],
        "rating": [50, 40, 60, 35, 45, 30],
    })
