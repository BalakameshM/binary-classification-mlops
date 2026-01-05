# src/config/settings.py
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    MODELS_LATEST_DIR: Path = MODELS_DIR / "latest"
    MODELS_VERSIONS_DIR: Path = MODELS_DIR / "versions"

    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "cereal-mlops")

    RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))

settings = Settings()
