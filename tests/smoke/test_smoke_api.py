# tests/smoke/test_smoke_api.py
from pathlib import Path
import json

from fastapi.testclient import TestClient

from src.api.main import app
from src.config.settings import settings

client = TestClient(app)

def _ensure_model_exists():
    """
    Smoke tests should not assume ordering.
    If pointer.json isn't present, run:
      - training_pipeline (creates processed parquet)
      - train_classifier (creates model + pointer)
    """
    pointer = settings.MODELS_LATEST_DIR / "pointer.json"
    if pointer.exists():
        return

    # Run pipelines as modules to mimic real usage
    import os
    code = os.system("python -m src.pipelines.training_pipeline")
    assert code == 0, "training_pipeline failed"

    code = os.system("python -m src.models.train_classifier")
    assert code == 0, "train_classifier failed"

    assert pointer.exists(), "Model pointer.json not created after training"

def test_smoke_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_smoke_predict():
    _ensure_model_exists()

    payload = {
        "features": {
            "protein": 3, "fat": 1, "sodium": 200, "carbo": 15,
            "potass": 100, "vitamins": 25, "weight": 1, "cups": 1,
            "mfr": "K", "type": "C", "shelf": 2
        }
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text

    body = r.json()
    assert "prediction" in body
    assert "model_run_id" in body
    assert isinstance(body["prediction"], int)
