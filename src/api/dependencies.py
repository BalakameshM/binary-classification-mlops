# src/api/dependencies.py
from functools import lru_cache
from src.models.predict import load_latest_model

@lru_cache(maxsize=1)
def get_model_bundle():
    # returns (model, run_id)
    model, run_id = load_latest_model()
    return model, run_id
