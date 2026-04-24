"""Evaluation metrics for classification, forecasting, and anomaly detection tasks."""

from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground-truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).

    Returns:
        Dict with 'acc' and 'f1' keys.
    """
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute forecasting metrics in normalised and (optionally) raw space.

    Args:
        y_true: Ground-truth values, shape ``(N, H, C_raw)``, ``(N, H*C_raw)``
            or any shape whose last dimension — once reshaped by the caller
            if needed — is ``C_raw`` for scaler compatibility.
        y_pred: Predicted values, same shape as ``y_true``.
        scaler: Optional scikit-learn ``StandardScaler`` (or compatible) fit
            on the raw variables.  When provided, an ``inverse_transform`` is
            applied before computing the ``mse_raw`` / ``mae_raw`` keys.  The
            last axis of ``y_true`` / ``y_pred`` must match
            ``scaler.n_features_in_``; reshape externally otherwise.

    Returns:
        Dict with keys ``mse``, ``mae`` always; ``mse_raw``, ``mae_raw`` when
        ``scaler`` is provided.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    out: Dict[str, float] = {"mse": mse, "mae": mae}

    if scaler is not None:
        C = y_true.shape[-1]
        flat_true = y_true.reshape(-1, C)
        flat_pred = y_pred.reshape(-1, C)
        raw_true = scaler.inverse_transform(flat_true)
        raw_pred = scaler.inverse_transform(flat_pred)
        out["mse_raw"] = float(np.mean((raw_true - raw_pred) ** 2))
        out["mae_raw"] = float(np.mean(np.abs(raw_true - raw_pred)))

    return out


def compute_anomaly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute anomaly detection metrics.

    Args:
        y_true: Binary ground-truth labels (0=normal, 1=anomaly), shape (N,).
        y_pred: Binary predicted labels, shape (N,).

    Returns:
        Dict with 'f1', 'precision', and 'recall' keys.
    """
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def compute_metrics(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Dispatch to the correct metric computation based on task type.

    Args:
        task_type: One of 'classification', 'forecasting', 'anomaly_detection'.
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dict of metric name → value.

    Raises:
        ValueError: If task_type is not recognised.
    """
    if task_type == "classification":
        return compute_classification_metrics(y_true, y_pred)
    elif task_type == "forecasting":
        return compute_forecasting_metrics(y_true, y_pred)
    elif task_type == "anomaly_detection":
        return compute_anomaly_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")
