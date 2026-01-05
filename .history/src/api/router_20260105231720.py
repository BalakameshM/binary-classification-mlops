# src/api/router.py
from fastapi import APIRouter
import pandas as pd
from src.api.schemas import PredictRequest, PredictResponse
from src.models.predict import predict_df

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([req.features])
    preds, run_id = predict_df(df)
    return PredictResponse(prediction=int(preds[0]), model_run_id=run_id)
