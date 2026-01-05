# src/tracking/mlflow_utils.py
from typing import Optional
import mlflow
from src.config.settings import settings

def setup_mlflow(experiment_name: Optional[str] = None):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    exp = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
    mlflow.set_experiment(exp)

def log_dict(prefix: str, d: dict):
    # flatten one level
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float, str, bool)):
                    mlflow.log_param(f"{prefix}.{k}.{kk}", vv) if isinstance(vv, str) else mlflow.log_metric(f"{prefix}.{k}.{kk}", float(vv))
        elif isinstance(v, (int, float)):
            mlflow.log_metric(f"{prefix}.{k}", float(v))
