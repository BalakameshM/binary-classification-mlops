# src/api/router.py 
from fastapi import APIRouter, Depends
import pandas as pd
from src.api.schemas import PredictRequest, PredictResponse
from src.api.dependencies import get_model_bundle

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, bundle=Depends(get_model_bundle)):
    model, run_id = bundle
    df = pd.DataFrame([req.features])
    pred = model.predict(df)[0]
    return PredictResponse(prediction=int(pred), model_run_id=run_id)
