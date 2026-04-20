"""Forecasting-evaluation primitives aligned with TS2Vec's official protocol.

This module is a drop-in port of ``reference/ts2vec/tasks/forecasting.py`` +
``reference/ts2vec/tasks/_eval_protocols.py::fit_ridge`` that works with any
``(B, T, C) -> (B, T, D)`` ``nn.Module`` (not tied to TS2Vec's own class).

Three primitives
----------------
``causal_sliding_encode(encoder, x, padding, batch_size)``
    Per-timestep representations where each position t sees only context
    ``x[max(0, t-padding) : t+1]`` (left-zero-padded when t < padding).
    Equivalent to TS2Vec's ``encode(..., causal=True, sliding_length=1,
    sliding_padding=padding)``.  This is the critical step that removes the
    forward-information leak a dilated same-padding CNN otherwise has.

``generate_pred_samples(features, data, pred_len, drop)``
    Multi-step labels: for each timestep t, the target is the sequence
    ``data[t+1..t+pred_len]`` flattened.  ``drop`` skips the first few
    training samples whose causal context is mostly zero padding.

``fit_ridge(train_x, train_y, valid_x, valid_y, alphas)``
    Ridge with α picked by minimising ``sqrt(MSE)+MAE`` on a validation
    split — TS2Vec's alpha-selection rule.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Matches reference/ts2vec/tasks/_eval_protocols.py::fit_ridge.
RIDGE_ALPHAS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Default causal-window padding.  TS2Vec uses 200 for forecasting.
DEFAULT_PADDING = 200


@torch.no_grad()
def causal_sliding_encode(
    encoder: nn.Module,
    x: torch.Tensor,
    padding: int = DEFAULT_PADDING,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Causal per-timestep representations of a single time series.

    For each timestep ``t ∈ [0, T)`` the returned row is the encoder output
    at the last position of the window ``x[max(0, t-padding) : t+1]``
    (left-zero-padded when ``t < padding``).

    Args:
        encoder: Any ``nn.Module`` mapping ``(B, T, C) → (B, T, D)``.
        x: Input series of shape ``(1, T, C)``.
        padding: Size of the past-only context used per timestep.
        batch_size: Number of sliding windows stacked per forward pass.
        device: Torch device.  Defaults to the encoder's device.

    Returns:
        NumPy array of shape ``(T, D)``.
    """
    if device is None:
        device = next(encoder.parameters()).device

    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"causal_sliding_encode expects (1, T, C), got {tuple(x.shape)}")
    _, T, C = x.shape
    x = x.to(device)

    # Left-pad the series so that every sliding window has uniform length
    # (padding + 1) — the earliest windows are zero-padded on the left.
    pad = torch.zeros(1, padding, C, device=device, dtype=x.dtype)
    padded = torch.cat([pad, x], dim=1)                # (1, T + padding, C)

    was_training = encoder.training
    encoder.eval()
    try:
        out_rows = []
        win_len = padding + 1
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            # Vectorised window gather: windows[i] = padded[0, start+i : start+i+win_len]
            row_idx = torch.arange(start, end, device=device).unsqueeze(1)
            col_idx = torch.arange(win_len, device=device).unsqueeze(0)
            idx = row_idx + col_idx                    # (B_w, win_len)
            windows = padded[0, idx]                   # (B_w, win_len, C)
            h = encoder(windows)                       # (B_w, win_len, D)
            out_rows.append(h[:, -1, :].cpu())         # (B_w, D)
    finally:
        if was_training:
            encoder.train()

    return torch.cat(out_rows, dim=0).numpy()          # (T, D)


def generate_pred_samples(
    features: np.ndarray,
    data: np.ndarray,
    pred_len: int,
    drop: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-step features/labels for a single-series Ridge regression.

    Mirrors ``reference/ts2vec/tasks/forecasting.py::generate_pred_samples``
    (specialised to a single series — ZeroAutoCL's forecasting datasets are
    always stored as ``(1, T, C)``).

    Args:
        features: Per-timestep representations, shape ``(T, D)``.
        data:     Raw target series, shape ``(T, C)``.
        pred_len: Prediction horizon H.
        drop:     Skip the first ``drop`` timesteps (used to discard training
                  samples whose causal context is mostly zero padding).

    Returns:
        ``features_flat`` of shape ``(T - H - drop, D)``;
        ``labels_flat``   of shape ``(T - H - drop, H * C)``.
    """
    T = data.shape[0]
    if T <= pred_len + drop:
        raise ValueError(
            f"series too short: T={T}, pred_len={pred_len}, drop={drop}"
        )

    feats = features[: T - pred_len]                               # (T-H, D)
    labels = np.stack(
        [data[i : i + T - pred_len] for i in range(1, pred_len + 1)],
        axis=1,
    )                                                              # (T-H, H, C)

    feats = feats[drop:]
    labels = labels[drop:]
    return feats, labels.reshape(labels.shape[0], -1)


def fit_ridge(
    train_features: np.ndarray,
    train_y: np.ndarray,
    valid_features: np.ndarray,
    valid_y: np.ndarray,
    alphas: Iterable[float] = RIDGE_ALPHAS,
    max_samples: int = 100_000,
) -> Ridge:
    """Ridge with α selected by minimising ``sqrt(MSE) + MAE`` on val.

    Faithful port of
    ``reference/ts2vec/tasks/_eval_protocols.py::fit_ridge``.
    """
    if train_features.shape[0] > max_samples:
        train_features, _, train_y, _ = train_test_split(
            train_features, train_y, train_size=max_samples, random_state=0,
        )
    if valid_features.shape[0] > max_samples:
        valid_features, _, valid_y, _ = train_test_split(
            valid_features, valid_y, train_size=max_samples, random_state=0,
        )

    best_alpha, best_score = None, np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(train_features, train_y)
        pred  = model.predict(valid_features)
        score = (np.sqrt(((pred - valid_y) ** 2).mean())
                 + np.abs(pred - valid_y).mean())
        if score < best_score:
            best_score, best_alpha = score, alpha

    return Ridge(alpha=best_alpha).fit(train_features, train_y)
