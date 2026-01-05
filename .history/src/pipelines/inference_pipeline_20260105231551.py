# src/pipelines/inference_pipeline.py
import pandas as pd
from src.config.settings import settings
from src.utils.logger import get_logger
from src.models.predict import predict_df

log = get_logger(__name__)

def run(input_path: str = None, output_path: str = None):
    # default: run inference on test set features
    if input_path is None:
        input_path = str(settings.PROCESSED_DIR / "test.parquet")

    df = pd.read_parquet(input_path)
    X = df.drop(columns=["target"], errors="ignore")

    preds, run_id = predict_df(X)
    out = df.copy()
    out["pred"] = preds

    if output_path is None:
        output_path = str(settings.ARTIFACTS_DIR / "predictions" / f"pred_{run_id}.parquet")

    out.to_parquet(output_path, index=False)
    log.info(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    run()
