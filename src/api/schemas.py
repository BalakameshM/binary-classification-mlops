# src/api/schemas.py
from pydantic import BaseModel
from typing import Dict, Any, List

class PredictRequest(BaseModel):
    # generic dict; your client sends feature:value
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    prediction: int
    model_run_id: str
