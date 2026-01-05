# tests/integration/test_full_training_pipeline.py
from pathlib import Path
import os
import json

from src.config.settings import settings

def test_full_training_pipeline_end_to_end():
    # 1) training pipeline: creates processed train/test
    code = os.system("python -m src.pipelines.training_pipeline")
    assert code == 0, "training_pipeline failed"

    train_path = settings.PROCESSED_DIR / "train.parquet"
    test_path = settings.PROCESSED_DIR / "test.parquet"
    assert train_path.exists(), "train.parquet not created"
    assert test_path.exists(), "test.parquet not created"

    # 2) train model: creates versions/<run_id>/ and latest pointer
    code = os.system("python -m src.models.train_classifier")
    assert code == 0, "train_classifier failed"

    pointer = settings.MODELS_LATEST_DIR / "pointer.json"
    assert pointer.exists(), "models/latest/pointer.json not created"

    run_id = json.loads(pointer.read_text())["run_id"]
    model_dir = settings.MODELS_VERSIONS_DIR / run_id
    assert model_di_
