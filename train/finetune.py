"""Downstream fine-tuning via linear probing.

After contrastive pre-training the encoder is frozen and a lightweight linear
head is trained on labelled downstream data.  This is the standard evaluation
protocol for self-supervised time-series methods.

Supported task types
--------------------
- ``'classification'``    → cross-entropy trained linear classifier
- ``'forecasting'``       → MSE trained multi-output linear regressor
- ``'anomaly_detection'`` → binary cross-entropy trained linear scorer
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import TimeSeriesDataset
from models.encoder.dilated_cnn import DilatedCNNEncoder

logger = logging.getLogger(__name__)

_DEFAULT_FINETUNE_CFG = {
    "epochs": 50,
    "lr": 1e-3,
    "batch_size": 64,
}


# ---------------------------------------------------------------------------
# Representation extraction helper
# ---------------------------------------------------------------------------

def _encode_pool(
    encoder: DilatedCNNEncoder,
    dataset: TimeSeriesDataset,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Encode dataset and mean-pool over the time axis.

    Returns:
        Float tensor of shape ``(N, D)`` on CPU.
    """
    encoder = encoder.to(device)
    encoder.eval()
    parts: List[Tensor] = []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            x = dataset.data[i : i + batch_size].to(device)
            h = encoder(x).mean(dim=1).cpu()
            parts.append(h)
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Linear head training
# ---------------------------------------------------------------------------

class _LinearHead(nn.Module):
    """Single linear layer used as downstream head."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def finetune_linear_probe(
    encoder: DilatedCNNEncoder,
    train_data: TimeSeriesDataset,
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    horizons: Optional[List[int]] = None,
) -> Dict:
    """Train a linear head on frozen encoder representations.

    The encoder weights are **not** updated; only the linear head is trained.

    Args:
        encoder: Pre-trained encoder (frozen during fine-tuning).
        train_data: Labelled training split.
        config: Fine-tuning config.  Recognised keys: ``epochs``, ``lr``,
            ``batch_size``.  Missing keys fall back to defaults.
        device: Torch device.  ``None`` → auto-detect.
        horizons: Forecasting horizons (only used for ``'forecasting'`` task).
            Defaults to ``[24, 48, 168, 336, 720]``.

    Returns:
        Dict with key ``'head'`` mapping to the trained
        :class:`_LinearHead`, plus task-specific metadata.
    """
    if config is None:
        config = {}
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if horizons is None:
        horizons = [24, 48, 168, 336, 720]

    epochs     = int(config.get("finetune_epochs", config.get("epochs", _DEFAULT_FINETUNE_CFG["epochs"])))
    lr         = float(config.get("finetune_lr", config.get("lr", _DEFAULT_FINETUNE_CFG["lr"])))
    batch_size = int(config.get("batch_size", _DEFAULT_FINETUNE_CFG["batch_size"]))
    task_type  = train_data.task_type

    # Move encoder to device once (all branches need this).
    encoder = encoder.to(device)

    if task_type == "classification":
        # ── Encode training set (encoder frozen) ─────────────────────
        repr_train = _encode_pool(encoder, train_data, batch_size, device)  # (N, D)
        D = repr_train.shape[1]
        n_classes = int(train_data.labels.max().item()) + 1
        head = _LinearHead(D, n_classes).to(device)
        labels = train_data.labels  # (N,)

        ds = TensorDataset(repr_train, labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(head.parameters(), lr=lr)
        head.train()
        for epoch in range(epochs):
            for h_batch, y_batch in loader:
                h_batch, y_batch = h_batch.to(device), y_batch.to(device)
                loss = F.cross_entropy(head(h_batch), y_batch)
                opt.zero_grad(); loss.backward(); opt.step()
        logger.info("Linear probe (classification) trained for %d epochs", epochs)
        return {"head": head, "n_classes": n_classes}

    elif task_type == "forecasting":
        # For forecasting the dataset is a single long series (N=1).
        # Encode per-timestep to get sliding-window representations.
        encoder.eval()
        with torch.no_grad():
            x_full = train_data.data.to(device)   # (1, T, C)
            h_full = encoder(x_full).cpu()         # (1, T, D)
            h_seq = h_full[0]                      # (T, D)
            x_seq = train_data.data[0]             # (T, C)
            D = h_seq.shape[1]

        results: Dict[str, _LinearHead] = {}
        for H in horizons:
            if h_seq.shape[0] <= H:
                continue
            X = h_seq[:-H]       # (T-H, D)
            y = x_seq[H:]        # (T-H, C)

            head_h = _LinearHead(D, x_seq.shape[1]).to(device)
            ds_h = TensorDataset(X, y)
            loader_h = DataLoader(ds_h, batch_size=min(batch_size, len(ds_h)), shuffle=True)

            opt_h = torch.optim.Adam(head_h.parameters(), lr=lr)
            head_h.train()
            for _ in range(epochs):
                for hb, yb in loader_h:
                    hb, yb = hb.to(device), yb.to(device)
                    loss_h = F.mse_loss(head_h(hb), yb)
                    opt_h.zero_grad(); loss_h.backward(); opt_h.step()
            results[H] = head_h
        logger.info("Linear probe (forecasting) trained for horizons %s", list(results.keys()))
        return {"heads": results}

    elif task_type == "anomaly_detection":
        repr_train = _encode_pool(encoder, train_data, batch_size, device)  # (N, D)
        D = repr_train.shape[1]
        n_out = 1  # anomaly score
        head = _LinearHead(D, n_out).to(device)
        # Use proxy labels: normal=0 (all training samples are normal)
        pseudo_labels = torch.zeros(len(repr_train), 1)
        ds = TensorDataset(repr_train, pseudo_labels)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(head.parameters(), lr=lr)
        head.train()
        for _ in range(epochs):
            for h_batch, y_batch in loader:
                h_batch, y_batch = h_batch.to(device), y_batch.to(device)
                loss = F.binary_cross_entropy_with_logits(head(h_batch), y_batch)
                opt.zero_grad(); loss.backward(); opt.step()
        logger.info("Linear probe (anomaly_detection) trained for %d epochs", epochs)
        return {"head": head}

    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")
