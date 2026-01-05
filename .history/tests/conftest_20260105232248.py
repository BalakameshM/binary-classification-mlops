# tests/conftest.py
import pytest
from pathlib import Path
import pandas as pd

@pytest.fixture
def raw_csv_path():
    return Path("data/raw/cereal.csv")

@pytest.fixture
def sample_df(raw_csv_path):
    return pd.read_csv(raw_csv_path).head(20)
