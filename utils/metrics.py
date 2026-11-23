"""
utils/metrics.py
----------------
Evaluation metric computation for circuit power prediction.

All predictions and targets arrive in z-score normalised space (the same
space used during training). This module first inverts the normalisation
to recover actual power values, then computes standard regression metrics.

Metrics reported:
    MAE  — Mean Absolute Error  (lower is better)
    RMSE — Root Mean Squared Error  (lower is better)
    R²   — Coefficient of Determination  (higher is better, max = 1.0)
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    y_mean:      float,
    y_std:       float,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and R² on the original (unnormalised) power scale.

    Parameters
    ----------
    y_true_norm : (N,) array — ground-truth normalised power values
    y_pred_norm : (N,) array — predicted normalised power values
    y_mean      : float — mean used during normalisation (for inverse transform)
    y_std       : float — std  used during normalisation (for inverse transform)

    Returns
    -------
    dict with keys: "MAE", "RMSE", "R2"
    """
    # Invert z-score normalisation:  y_actual = (y_norm * std) + mean
    y_true = y_true_norm * y_std + y_mean
    y_pred = y_pred_norm * y_std + y_mean

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
