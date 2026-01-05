import pandas as pd
from src.models.train_classifier import train
from pathlib import Path

def test_train_returns_metadata(tmp_path):
    # tiny synthetic dataset with required columns
    df = pd.DataFrame({
        "protein":[3,2,4,1,2,3],
        "fat":[1,1,2,1,1,2],
        "sodium":[100,120,130,90,110,140],
        "carbo":[10,12,11,9,13,14],
        "potass":[80,60,90,50,70,100],
        "vitamins":[25,25,0,25,0,25],
        "weight":[1,1,1,1,1,1],
        "cups":[1,1,1,1,1,1],
        "mfr":["K","K","G","G","N","N"],
        "type":["C","C","C","C","C","C"],
        "shelf":[1,2,1,2,3,3],
        "target":[0,1,2,0,1,2],
    })
    _, meta = train(df, Path(tmp_path))
    assert "metrics_holdout" in meta
