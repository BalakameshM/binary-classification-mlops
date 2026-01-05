# src/dashboard/utils.py
import pandas as pd
from src.models.predict import load_latest_model

def predict_single(features: dict):
    model, run_id = load_latest_model()
    df = pd.DataFrame([features])
    pred = int(model.predict(df)[0])
    return pred, run_id
