from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_predict():
    payload = {
        "features": {
            "protein": 3, "fat": 1, "sodium": 200, "carbo": 15,
            "potass": 100, "vitamins": 25, "weight": 1, "cups": 1,
            "mfr": "K", "type": "C", "shelf": 2
        }
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "model_run_id" in body
