import os
from pathlib import Path

def test_training_pipeline_runs():
    code = os.system("python -m src.pipelines.training_pipeline")
    assert code == 0

def test_train_classifier_runs():
    code = os.system("python -m src.models.train_classifier")
    assert code == 0

def test_monitoring_pipeline_runs():
    code = os.system("python -m src.pipelines.monitoring_pipeline")
    assert code == 0
