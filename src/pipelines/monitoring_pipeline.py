# src/pipelines/monitoring_pipeline.py
import pandas as pd
from src.config.settings import settings
from src.utils.logger import get_logger
from src.evaluation.drift import compute_drift, drift_alerts
from src.utils.io_utils import write_json, ensure_dir

log = get_logger(__name__)

def run():
    # Reference = train, Current = test (demo). In real prod: current = recent inference batch.
    ref = pd.read_parquet(settings.PROCESSED_DIR / "train.parquet")
    cur = pd.read_parquet(settings.PROCESSED_DIR / "test.parquet")

    refX = ref.drop(columns=["target"], errors="ignore")
    curX = cur.drop(columns=["target"], errors="ignore")

    # numeric cols = all non-categorical known ones
    numeric_cols = [c for c in refX.columns if c not in ["mfr","type","shelf"]]

    drift = compute_drift(refX, curX, numeric_cols=numeric_cols, psi_bins=10)
    alerts = drift_alerts(drift, psi_threshold=0.2)

    ensure_dir(settings.ARTIFACTS_DIR / "monitoring")
    out_path = settings.ARTIFACTS_DIR / "monitoring" / "drift_report.json"
    write_json(out_path, {"drift": drift, "alerts": alerts})

    log.info(f"Saved drift report to {out_path}")
    if alerts:
        log.warning(f"Drift alerts triggered for: {list(alerts.keys())}")
    else:
        log.info("No drift alerts triggered.")

if __name__ == "__main__":
    run()
