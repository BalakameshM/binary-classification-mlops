# src/evaluation/metrics.py
from dataclasses import asdict, dataclass
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

@dataclass
class Metrics:
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float

def compute_metrics(y_true, y_pred) -> Metrics:
    return Metrics(
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted")),
    )

def report_dict(y_true, y_pred) -> dict:
    return classification_report(y_true, y_pred, output_dict=True)

def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

def metrics_to_dict(m: Metrics) -> dict:
    return asdict(m)
