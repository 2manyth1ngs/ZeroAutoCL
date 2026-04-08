"""Embedding-space transformations applied after the encoder.

Three operations are applied in sequence (training mode only for stochastic ops):
  1. Jitter  — additive Gaussian noise with std = jitter_p
  2. Mask    — Bernoulli dropout on the feature (D) dimension with prob mask_p
  3. Norm    — LayerNorm, L2 normalisation, or identity
"""

import torch
import torch.nn as nn
from torch import Tensor


class EmbeddingTransform(nn.Module):
    """Apply jitter, feature-masking, and normalisation to encoder embeddings.

    Stochastic operations (jitter, mask) are only active during training,
    mirroring the behaviour of ``nn.Dropout``.  Normalisation is always
    applied.

    Args:
        jitter_p: Standard deviation of additive Gaussian noise applied to
            each embedding value.  ``0.0`` disables jitter.
        mask_p: Probability of zeroing each feature dimension independently
            (Bernoulli mask over D).  ``0.0`` disables masking.
        norm_type: One of ``'none'``, ``'layer_norm'``, ``'l2'``.
        embed_dim: Embedding dimensionality D.  Required when
            ``norm_type='layer_norm'``.
    """

    def __init__(
        self,
        jitter_p: float,
        mask_p: float,
        norm_type: str,
        embed_dim: int,
    ) -> None:
        super().__init__()

        if norm_type not in ("none", "layer_norm", "l2"):
            raise ValueError(
                f"norm_type must be 'none', 'layer_norm', or 'l2', got {norm_type!r}"
            )

        self.jitter_p = jitter_p
        self.mask_p = mask_p
        self.norm_type = norm_type

        self.layer_norm: nn.LayerNorm | None = (
            nn.LayerNorm(embed_dim) if norm_type == "layer_norm" else None
        )

    def forward(self, h: Tensor) -> Tensor:
        """Apply embedding transform.

        Args:
            h: Encoder output of shape (B, T, D).

        Returns:
            Transformed embeddings, same shape (B, T, D).
        """
        if self.training:
            # ── Jitter ──────────────────────────────────────────────────────
            if self.jitter_p > 0.0:
                h = h + torch.randn_like(h) * self.jitter_p

            # ── Feature mask (Bernoulli on D dimension) ─────────────────────
            if self.mask_p > 0.0:
                # Shape (B, 1, D): the same mask is applied to all time steps
                # of a sample, but differs per feature and per sample.
                keep_prob = 1.0 - self.mask_p
                mask = torch.bernoulli(
                    torch.full(
                        (h.shape[0], 1, h.shape[2]),
                        keep_prob,
                        device=h.device,
                        dtype=h.dtype,
                    )
                )
                h = h * mask

        # ── Normalisation (always active) ───────────────────────────────────
        if self.norm_type == "layer_norm":
            assert self.layer_norm is not None
            h = self.layer_norm(h)
        elif self.norm_type == "l2":
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)

        return h
