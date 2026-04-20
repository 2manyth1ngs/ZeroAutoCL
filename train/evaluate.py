"""Downstream task evaluation for contrastive pre-trained encoders.

All evaluation is *representation-based*: the encoder is frozen and classical
ML models (SVM, Ridge regression) or threshold-based detectors are applied
to the pooled embeddings.  This follows the standard AutoCLS / TS2Vec
evaluation protocol.

Functions
---------
encode_and_pool       — encode a dataset and mean-pool over time
eval_classification   — SVM (RBF) → accuracy + macro-F1
eval_forecasting      — Ridge per horizon → {H: (MSE, MAE)}
eval_anomaly_detection— neighbour-distance threshold → F1 / Precision / Recall
evaluate              — unified dispatcher
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from data.dataset import TimeSeriesDataset
from models.encoder.dilated_cnn import DilatedCNNEncoder
from utils.metrics import (
    compute_classification_metrics,
    compute_forecasting_metrics,
    compute_anomaly_metrics,
)

logger = logging.getLogger(__name__)

_DEFAULT_HORIZONS = [24, 48, 168, 336, 720]


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_and_pool(
    encoder: nn.Module,
    dataset: TimeSeriesDataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Encode all samples and mean-pool over the time axis.

    Args:
        encoder: Pre-trained encoder (set to eval mode internally).
        dataset: Dataset whose ``.data`` attribute has shape ``(N, T, C)``.
        batch_size: Inference batch size.
        device: Device.  ``None`` → auto-detect.

    Returns:
        NumPy array of shape ``(N, D)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    encoder.eval()
    parts: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            x = dataset.data[i : i + batch_size].to(device)
            h = encoder(x).mean(dim=1).cpu().numpy()   # (B, D)
            parts.append(h)
    return np.concatenate(parts, axis=0)               # (N, D)


def _encode_timesteps(
    encoder: nn.Module,
    dataset: TimeSeriesDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode all samples, returning per-timestep embeddings.

    Returns:
        NumPy array of shape ``(N, T, D)``.
    """
    encoder = encoder.to(device)
    encoder.eval()
    parts: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            x = dataset.data[i : i + batch_size].to(device)
            h = encoder(x).cpu().numpy()   # (B, T, D)
            parts.append(h)
    return np.concatenate(parts, axis=0)   # (N, T, D)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def eval_classification(
    encoder: nn.Module,
    train_data: TimeSeriesDataset,
    test_data: TimeSeriesDataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate via SVM (RBF) on time-pooled embeddings.

    Args:
        encoder: Pre-trained encoder.
        train_data: Labelled training split.
        test_data: Labelled test split.
        batch_size: Inference batch size.
        device: Torch device.

    Returns:
        Dict with keys ``'acc'`` and ``'f1'``.
    """
    from sklearn.svm import SVC

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_repr = encode_and_pool(encoder, train_data, batch_size, device)
    test_repr  = encode_and_pool(encoder, test_data, batch_size, device)

    train_labels = train_data.labels.numpy()
    test_labels  = test_data.labels.numpy()

    svm = SVC(kernel="rbf")
    svm.fit(train_repr, train_labels)
    preds = svm.predict(test_repr)

    metrics = compute_classification_metrics(test_labels, preds)
    logger.info(
        "eval_classification: acc=%.4f  f1=%.4f",
        metrics["acc"], metrics["f1"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def eval_forecasting(
    encoder: nn.Module,
    train_data: TimeSeriesDataset,
    test_data: TimeSeriesDataset,
    horizons: Optional[List[int]] = None,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    val_data: Optional[TimeSeriesDataset] = None,
) -> Dict[int, Dict[str, float]]:
    """Evaluate via Ridge regression per horizon, TS2Vec-aligned protocol.

    Three things differ from a naive Ridge on ``encoder(x)``:

    1. **Causal sliding encode** — each timestep's repr sees only past
       context (window ``[t-padding, t]``).  A dilated same-padding CNN
       would otherwise leak future values into ``h[t]``, turning the Ridge
       into a near-identity on ``x[t+H]``.
    2. **Multi-step target** — for each timestep Ridge predicts the full
       sequence ``x[t+1..t+H]`` (flattened), not just the single point
       ``x[t+H]``.  This matches the numbers TS2Vec / AutoCLS papers report.
    3. **α selected on val** — Ridge regularisation is picked from
       ``{0.1, 0.2, …, 1000}`` by minimising ``sqrt(MSE)+MAE`` on a held-out
       val split; if ``val_data`` is missing a 20% tail of train serves as
       a proxy.

    See Bug #006 in CLAUDE_DEBUG.md for the full rationale.

    Args:
        encoder: Pre-trained encoder, any ``(B, T, C) → (B, T, D)`` module.
        train_data: Training split (forecasting, shape ``(1, T_tr, C)``).
        test_data:  Test split     (forecasting, shape ``(1, T_te, C)``).
        horizons: Prediction horizons.  Defaults to ``[24, 48, 168, 336, 720]``.
        batch_size: Sliding-window batch size for the causal encode.
        device: Torch device.
        val_data: Optional val split used to pick Ridge α.

    Returns:
        Dict mapping each horizon to ``{'mse': …, 'mae': …}``.
    """
    from train.forecasting_eval import (
        DEFAULT_PADDING, causal_sliding_encode,
        fit_ridge, generate_pred_samples,
    )

    if horizons is None:
        horizons = _DEFAULT_HORIZONS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    encoder.eval()
    padding = DEFAULT_PADDING

    # ── Prefix encoding: concat train [+ val] + test, encode once, slice ──
    # With a causal sliding window this ensures that val/test timesteps near
    # the split boundaries see real preceding context rather than zero-pad.
    x_tr_t = train_data.data          # (1, T_tr, C)
    x_te_t = test_data.data           # (1, T_te, C)
    T_tr, T_te = x_tr_t.shape[1], x_te_t.shape[1]

    if val_data is not None:
        x_va_t = val_data.data        # (1, T_va, C)
        T_va = x_va_t.shape[1]
        full = torch.cat([x_tr_t, x_va_t, x_te_t], dim=1)
    else:
        T_va = 0
        full = torch.cat([x_tr_t, x_te_t], dim=1)

    h_full = causal_sliding_encode(
        encoder, full, padding=padding, batch_size=256, device=device,
    )                                 # (T_tr + T_va + T_te, D)
    h_tr = h_full[:T_tr]
    h_va = h_full[T_tr : T_tr + T_va] if T_va > 0 else None
    h_te = h_full[T_tr + T_va : T_tr + T_va + T_te]

    d_tr = x_tr_t[0].numpy()
    d_te = x_te_t[0].numpy()
    d_va = x_va_t[0].numpy() if T_va > 0 else None

    results: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        if h_tr.shape[0] <= H + padding or h_te.shape[0] <= H:
            logger.warning("Horizon %d too large for series length; skipping.", H)
            continue

        tr_f, tr_y = generate_pred_samples(h_tr, d_tr, H, drop=padding)
        te_f, te_y = generate_pred_samples(h_te, d_te, H, drop=0)

        if T_va > 0 and h_va.shape[0] > H:
            va_f, va_y = generate_pred_samples(h_va, d_va, H, drop=0)
        else:
            # No separate val split: use a 20% tail of train to pick α.
            split = max(1, int(0.8 * tr_f.shape[0]))
            va_f, va_y = tr_f[split:], tr_y[split:]
            tr_f, tr_y = tr_f[:split], tr_y[:split]

        lr   = fit_ridge(tr_f, tr_y, va_f, va_y)
        pred = lr.predict(te_f)

        metrics = compute_forecasting_metrics(te_y, pred)
        results[H] = metrics
        logger.info(
            "eval_forecasting H=%d: mse=%.4f  mae=%.4f",
            H, metrics["mse"], metrics["mae"],
        )

    return results


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def eval_anomaly_detection(
    encoder: nn.Module,
    train_data: TimeSeriesDataset,
    test_data: TimeSeriesDataset,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate via neighbour-distance anomaly scoring.

    Per-sample anomaly score = mean L2 distance between adjacent
    time-step embeddings.  Higher score → more irregular → potential anomaly.

    The optimal binary threshold is found by scanning candidate thresholds
    (percentiles of training scores) and maximising F1 on test data.

    Args:
        encoder: Pre-trained encoder.
        train_data: Normal training samples (used to calibrate threshold).
        test_data: Test samples with binary ground-truth labels.
        batch_size: Inference batch size.
        device: Torch device.

    Returns:
        Dict with keys ``'f1'``, ``'precision'``, ``'recall'``, ``'threshold'``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)

    # ── Per-sample anomaly scores ─────────────────────────────────────
    def anomaly_score(h_seq: np.ndarray) -> np.ndarray:
        """h_seq: (N, T, D) → scores (N,)."""
        # Mean L2 distance between consecutive time steps.
        diff = h_seq[:, 1:, :] - h_seq[:, :-1, :]     # (N, T-1, D)
        return np.mean(np.linalg.norm(diff, axis=-1), axis=-1)  # (N,)

    h_train = _encode_timesteps(encoder, train_data, batch_size, device)  # (N_tr, T, D)
    h_test  = _encode_timesteps(encoder, test_data,  batch_size, device)  # (N_te, T, D)

    train_scores = anomaly_score(h_train)   # (N_tr,)
    test_scores  = anomaly_score(h_test)    # (N_te,)

    test_labels = test_data.labels.numpy()  # (N_te,)

    # ── Threshold search on training score distribution ───────────────
    # Use percentiles of training scores as candidate thresholds, then
    # pick the one giving highest F1 on the test set.
    candidates = np.percentile(train_scores, np.arange(5, 100, 5))
    best_thresh = candidates[0]
    best_f1 = -1.0

    for thresh in candidates:
        preds = (test_scores > thresh).astype(int)
        if preds.sum() == 0:
            continue
        from sklearn.metrics import f1_score as _f1
        f1 = _f1(test_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds_final = (test_scores > best_thresh).astype(int)
    metrics = compute_anomaly_metrics(test_labels, preds_final)
    metrics["threshold"] = float(best_thresh)

    logger.info(
        "eval_anomaly_detection: f1=%.4f  precision=%.4f  recall=%.4f  thresh=%.4f",
        metrics["f1"], metrics["precision"], metrics["recall"], metrics["threshold"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def evaluate(
    encoder: nn.Module,
    train_data: TimeSeriesDataset,
    test_data: TimeSeriesDataset,
    task_type: str,
    horizons: Optional[List[int]] = None,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict:
    """Dispatch to the appropriate evaluation function.

    Args:
        encoder: Pre-trained encoder.
        train_data: Training split.
        test_data: Test split.
        task_type: ``'classification'``, ``'forecasting'``, or
            ``'anomaly_detection'``.
        horizons: Forecasting horizons (only used when
            *task_type* is ``'forecasting'``).
        batch_size: Inference batch size.
        device: Torch device.

    Returns:
        Metric dict (structure depends on *task_type*).

    Raises:
        ValueError: If *task_type* is not recognised.
    """
    if task_type == "classification":
        return eval_classification(encoder, train_data, test_data, batch_size, device)
    elif task_type == "forecasting":
        return eval_forecasting(encoder, train_data, test_data, horizons, batch_size, device)
    elif task_type == "anomaly_detection":
        return eval_anomaly_detection(encoder, train_data, test_data, batch_size, device)
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")
