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
    assert model_dir.exists(), f"model dir {model_dir} not found"

    assert (model_dir / "model.joblib").exists(), "model.joblib missing"
    assert (model_dir / "metadata.json").exists(), "metadata.json missing"

    # 3) inference pipeline
    code = os.system("python -m src.pipelines.inference_pipeline")
    assert code == 0, "inference_pipeline failed"

    # 4) monitoring pipeline should produce drift report
    code = os.system("python -m src.pipelines.monitoring_pipeline")
    assert code == 0, "monitoring_pipeline failed"

    drift_report = settings.ARTIFACTS_DIR / "monitoring" / "drift_report.json"
    assert drift_report.exists(), "drift_report.json not created"

    drift_data = json.loads(drift_report.read_text())
    assert "drift" in drift_data
    assert "alerts" in drift_data
