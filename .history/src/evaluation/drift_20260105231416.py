# src/evaluation/drift.py
import numpy as np
import pandas as pd

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    # Protect zeros
    expected = expected.astype(float)
    actual = actual.astype(float)
    eps = 1e-6

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:  # too few unique values
        return 0.0

    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)

    e_perc = np.maximum(e_counts / (len(expected) + eps), eps)
    a_perc = np.maximum(a_counts / (len(actual) + eps), eps)

    psi = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return float(psi)

def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, numeric_cols: list[str], psi_bins: int = 10) -> dict:
    out = {}
    for c in numeric_cols:
        if c not in reference.columns or c not in current.columns:
            continue
        ref = reference[c].dropna().to_numpy()
        cur = current[c].dropna().to_numpy()
        if len(ref) < 20 or len(cur) < 20:
            continue
        out[c] = {"psi": _psi(ref, cur, bins=psi_bins)}
    return out

def drift_alerts(drift: dict, psi_threshold: float = 0.2) -> dict:
    alerts = {}
    for col, m in drift.items():
        psi = m.get("psi", 0.0)
        if psi >= psi_threshold:
            alerts[col] = {"psi": psi, "alert": True}
    return alerts
