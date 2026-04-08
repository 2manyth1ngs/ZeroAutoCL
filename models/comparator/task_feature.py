"""Extract a fixed-size task-feature vector from a dataset.

The feature vector summarises both the *data distribution* (via a
lightweight contrastive encoder) and *meta-properties* of the dataset
(length, channels, size, task type, horizon).

Pipeline
--------
1. Train a small dilated-CNN encoder with instance-contrastive loss for a few
   epochs on the dataset (no need for the full CL strategy).
2. Encode all training samples → h (N, T, D).
3. Time-pool → h_pooled (N, D).
4. Compute statistics:  mean, std, quantiles [0.25, 0.5, 0.75] → 5 × D.
5. Append meta-features: [log(T), log(C), log(N), task_type_onehot(3), horizon_norm].
6. Concatenate into a single vector of dimension  5 * D + 7.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.encoder.encoder_config import EncoderConfig
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.losses import InfoNCELoss

logger = logging.getLogger(__name__)

_TASK_TYPES = ["classification", "forecasting", "anomaly_detection"]

# Default lightweight encoder config for feature extraction.
_FEAT_ENCODER_CFG = EncoderConfig(n_layers=4, hidden_dim=32, output_dim=64)
_FEAT_D = _FEAT_ENCODER_CFG.output_dim  # 64
TASK_FEATURE_DIM = 5 * _FEAT_D + 7      # 327


class TaskFeatureExtractor:
    """Extract a fixed-size feature vector for a given dataset / task.

    The extractor is *stateless* — each call to :meth:`extract` trains a
    fresh lightweight encoder from scratch so that the resulting features
    are independent across tasks.

    Args:
        encoder_config: Config for the lightweight encoder used internally.
            Defaults to a small 4-layer, hidden=32, output=64 encoder.
        pretrain_epochs: Number of instance-contrastive pretraining epochs.
        lr: Learning rate for the internal encoder training.
        batch_size: Batch size for the internal training loop.
        device: Device to use.  ``None`` → auto-detect.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig = _FEAT_ENCODER_CFG,
        pretrain_epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.encoder_config = encoder_config
        self.pretrain_epochs = pretrain_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        dataset: "torch.utils.data.Dataset",
        task_type: str,
        horizon: int = 0,
    ) -> Tensor:
        """Extract a task-feature vector for *dataset*.

        Args:
            dataset: A :class:`~data.dataset.TimeSeriesDataset`.  Only the
                ``data`` attribute (shape N, T, C) is used.
            task_type: ``'classification'``, ``'forecasting'``, or
                ``'anomaly_detection'``.
            horizon: Forecasting horizon; 0 for non-forecasting tasks.

        Returns:
            A 1-d tensor of shape ``(TASK_FEATURE_DIM,)``.
        """
        # ── Step 1: lightweight contrastive pretraining ──────────────
        data_tensor: Tensor = dataset.data  # (N, T, C)
        N, T, C = data_tensor.shape

        encoder = DilatedCNNEncoder(
            input_dim=C, config=self.encoder_config,
        ).to(self.device)

        self._pretrain(encoder, data_tensor)

        # ── Step 2-3: encode & pool ──────────────────────────────────
        encoder.eval()
        with torch.no_grad():
            h_pooled = self._encode_and_pool(encoder, data_tensor)  # (N, D)

        # ── Step 4: statistics ───────────────────────────────────────
        D = h_pooled.shape[1]
        mean = h_pooled.mean(dim=0)                              # (D,)
        std  = h_pooled.std(dim=0, correction=0)                  # (D,) — correction=0 avoids div-by-zero when N=1
        q25  = torch.quantile(h_pooled, 0.25, dim=0)            # (D,)
        q50  = torch.quantile(h_pooled, 0.50, dim=0)            # (D,)
        q75  = torch.quantile(h_pooled, 0.75, dim=0)            # (D,)

        stats = torch.cat([mean, std, q25, q50, q75])           # (5D,)

        # ── Step 5: meta-features (7-d) ──────────────────────────────
        task_oh = [1.0 if t == task_type else 0.0 for t in _TASK_TYPES]
        meta = torch.tensor(
            [np.log(T), np.log(C), np.log(N)] + task_oh + [horizon / 1000.0],
            dtype=torch.float32,
            device=stats.device,
        )

        # ── Step 6: concatenate ──────────────────────────────────────
        return torch.cat([stats, meta])   # (5D + 7,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pretrain(self, encoder: DilatedCNNEncoder, data: Tensor) -> None:
        """Train *encoder* with a basic instance-contrastive loss."""
        encoder.train()
        loss_fn = InfoNCELoss(sim_func="dot", temperature=0.5)
        opt = torch.optim.Adam(encoder.parameters(), lr=self.lr)
        N = data.shape[0]

        for epoch in range(self.pretrain_epochs):
            perm = torch.randperm(N)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                x = data[idx].to(self.device)  # (B, T, C)
                B = x.shape[0]
                if B < 2:
                    continue

                h = encoder(x)            # (B, T, D)
                a = h.mean(dim=1)         # (B, D)

                # Second view: simple jitter augmentation
                noise = torch.randn_like(x) * 0.1
                h2 = encoder(x + noise)
                p = h2.mean(dim=1)        # (B, D)

                # In-batch negatives
                D = a.shape[-1]
                p_exp = p.unsqueeze(0).expand(B, B, D)
                mask = ~torch.eye(B, dtype=torch.bool, device=a.device)
                neg = p_exp[mask].reshape(B, B - 1, D)

                loss = loss_fn(a, p, neg)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                n_batches += 1

            if n_batches > 0:
                logger.debug(
                    "TaskFeature pretrain epoch %d/%d  loss=%.4f",
                    epoch + 1, self.pretrain_epochs, total_loss / n_batches,
                )

    def _encode_and_pool(self, encoder: DilatedCNNEncoder, data: Tensor) -> Tensor:
        """Encode all samples and mean-pool over time.

        Returns:
            Tensor of shape ``(N, D)`` on CPU.
        """
        parts = []
        N = data.shape[0]
        for start in range(0, N, self.batch_size):
            x = data[start : start + self.batch_size].to(self.device)
            h = encoder(x)               # (B, T, D)
            h_pooled = h.mean(dim=1)     # (B, D)
            parts.append(h_pooled.cpu())
        return torch.cat(parts, dim=0)   # (N, D)
