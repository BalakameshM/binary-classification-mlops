# src/pipelines/training_pipeline.py
from src.config.settings import settings
from src.utils.logger import get_logger
from src.data.load_local import load_cereal_csv
from src.data.preprocess import preprocess
from src.utils.io_utils import ensure_dir
import pandas as pd
from sklearn.model_selection import train_test_split

log = get_logger(__name__)

def run():
    raw_path = settings.RAW_DIR / "cereal.csv"
    df = load_cereal_csv(raw_path)
    df = preprocess(df)

    ensure_dir(settings.PROCESSED_DIR)

    # split and save
    train_df, test_df = train_test_split(
        df, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=df["target"]
    )
    train_df.to_parquet(settings.PROCESSED_DIR / "train.parquet", index=False)
    test_df.to_parquet(settings.PROCESSED_DIR / "test.parquet", index=False)

    log.info(f"Saved train/test to {settings.PROCESSED_DIR}")

if __name__ == "__main__":
    run()
