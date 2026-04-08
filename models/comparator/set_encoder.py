"""Set Transformer encoder for aggregating variable-length feature sets.

Simplified implementation with:
  - 2 ISAB (Induced Set Attention Block) layers, *m* inducing points each
  - 1 PMA  (Pooling by Multihead Attention)  layer, *k* = 1 seed vector
  - Output: fixed-size vector of dimension *d_out*

Since the task feature extractor already computes fixed-dimension statistics,
a plain MLP fallback is also provided (``SetTransformerEncoder.from_fixed``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Multi-head attention building block
# ---------------------------------------------------------------------------

class _MAB(nn.Module):
    """Multihead Attention Block:  MAB(X, Y) = LN(X + MHA(X, Y, Y))."""

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out, _ = self.attn(x, y, y)
        return self.ln(x + out)


class _ISAB(nn.Module):
    """Induced Set Attention Block using *m* inducing points."""

    def __init__(self, d: int, m: int, n_heads: int) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, m, d) * 0.01)
        self.mab1 = _MAB(d, n_heads)
        self.mab2 = _MAB(d, n_heads)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, D)
        B = x.shape[0]
        I = self.inducing.expand(B, -1, -1)       # (B, m, D)
        H = self.mab1(I, x)                        # (B, m, D)
        return self.mab2(x, H)                      # (B, N, D)


class _PMA(nn.Module):
    """Pooling by Multihead Attention with *k* seed vectors."""

    def __init__(self, d: int, k: int, n_heads: int) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, k, d) * 0.01)
        self.mab = _MAB(d, n_heads)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, D)
        B = x.shape[0]
        S = self.seeds.expand(B, -1, -1)           # (B, k, D)
        return self.mab(S, x)                       # (B, k, D)


# ---------------------------------------------------------------------------
# SetTransformerEncoder
# ---------------------------------------------------------------------------

class SetTransformerEncoder(nn.Module):
    """Encode a variable-size set of feature vectors into a single vector.

    Architecture::

        input_proj → ISAB(m) → ISAB(m) → PMA(k=1) → squeeze → output_proj

    Args:
        d_in: Dimension of each input vector in the set.
        d_out: Output vector dimension.
        d_hidden: Hidden dimension used throughout the transformer blocks.
        n_inducing: Number of inducing points *m* in each ISAB.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int = 128,
        n_inducing: int = 16,
        n_heads: int = 4,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(d_in, d_hidden)
        self.isab1 = _ISAB(d_hidden, n_inducing, n_heads)
        self.isab2 = _ISAB(d_hidden, n_inducing, n_heads)
        self.pma = _PMA(d_hidden, k=1, n_heads=n_heads)
        self.output_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a set of vectors.

        Args:
            x: Shape ``(B, N, D_in)`` — a batch of *B* sets, each containing
               *N* vectors of dimension *D_in*.

        Returns:
            Shape ``(B, D_out)``.
        """
        h = self.input_proj(x)          # (B, N, d_hidden)
        h = self.isab1(h)               # (B, N, d_hidden)
        h = self.isab2(h)               # (B, N, d_hidden)
        h = self.pma(h)                 # (B, 1, d_hidden)
        h = h.squeeze(1)                # (B, d_hidden)
        return self.output_proj(h)      # (B, d_out)

    @classmethod
    def from_fixed(cls, d_in: int, d_out: int, d_hidden: int = 128) -> nn.Sequential:
        """Create a simple MLP for fixed-dimension input (no set structure).

        Use this when the task feature is already a pre-computed statistics
        vector of known size.

        Args:
            d_in: Input dimension.
            d_out: Output dimension.
            d_hidden: Hidden layer width.

        Returns:
            An ``nn.Sequential`` MLP.
        """
        return nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
