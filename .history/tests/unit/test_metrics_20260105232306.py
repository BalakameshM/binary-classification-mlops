import numpy as np
from src.evaluation.metrics import compute_metrics

def test_metrics_runs():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2])
    m = compute_metrics(y_true, y_pred)
    assert 0.0 <= m.f1_macro <= 1.0
