"""Encode (encoder_config, cl_strategy) pairs into fixed-size vectors.

The raw feature vector (dim=31) is a concatenation of normalised scalars and
one-hot encodings for every dimension of the joint configuration space.  An
MLP then maps this to the comparator's hidden dimension.

Layout (31-d raw vector)::

    encoder  [3] : n_layers/10, hidden_dim/128, output_dim/320
    aug probs[6] : resize, rescale, jitter, point_mask, freq_mask, crop
    aug order[5] : one-hot over 5 orders
    emb xform[5] : jitter_p, mask_p, norm_type one-hot(3)
    pairs    [6] : temporal, cross_scale, kernel/5, pool one-hot(2), adj
    loss     [6] : loss_type one-hot(2), sim one-hot(3), log10(temp)/2
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

RAW_DIM: int = 31  # total dimension of the raw feature vector


# ---------------------------------------------------------------------------
# One-hot helpers
# ---------------------------------------------------------------------------

_NORM_TYPES: List[str]  = ["none", "layer_norm", "l2"]
_POOL_OPS: List[str]    = ["avg", "max"]
_LOSS_TYPES: List[str]  = ["infonce", "triplet"]
_SIM_FUNCS: List[str]   = ["dot", "cosine", "euclidean"]


def _onehot(value: str, choices: List[str]) -> List[float]:
    """Return a one-hot list for *value* among *choices*."""
    return [1.0 if c == value else 0.0 for c in choices]


# ---------------------------------------------------------------------------
# CandidateEncoder
# ---------------------------------------------------------------------------

class CandidateEncoder(nn.Module):
    """Encode a (encoder_config, cl_strategy) pair into a hidden-dim vector.

    The encoding is deterministic (no learned embedding tables) up to the
    final MLP projection.

    Args:
        raw_dim: Dimension of the raw concatenated feature vector (default 31).
        hidden_dim: Output dimension after MLP projection.
    """

    def __init__(self, raw_dim: int = RAW_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.raw_dim = raw_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        encoder_config: Dict[str, int],
        strategy_config: Dict,
    ) -> Tensor:
        """Encode a single candidate into a vector of shape ``(hidden_dim,)``.

        Args:
            encoder_config: Dict with ``'n_layers'``, ``'hidden_dim'``,
                ``'output_dim'``.
            strategy_config: Full strategy dict (augmentation, embedding_transform,
                pair_construction, loss).

        Returns:
            Tensor of shape ``(hidden_dim,)``.
        """
        raw = self._to_raw_vector(encoder_config, strategy_config)  # (raw_dim,)
        return self.mlp(raw)

    def encode_batch(
        self,
        encoder_configs: List[Dict[str, int]],
        strategy_configs: List[Dict],
    ) -> Tensor:
        """Encode a batch of candidates.

        Args:
            encoder_configs: List of N encoder config dicts.
            strategy_configs: List of N strategy config dicts.

        Returns:
            Tensor of shape ``(N, hidden_dim)``.
        """
        raws = torch.stack([
            self._to_raw_vector(ec, sc)
            for ec, sc in zip(encoder_configs, strategy_configs)
        ])  # (N, raw_dim)
        return self.mlp(raws)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _to_raw_vector(
        self,
        encoder_config: Dict[str, int],
        strategy_config: Dict,
    ) -> Tensor:
        """Build the 31-d raw feature vector from config dicts.

        Returns:
            Tensor of shape ``(raw_dim,)`` on the same device as the MLP.
        """
        aug  = strategy_config.get("augmentation", {})
        emb  = strategy_config.get("embedding_transform", {})
        pair = strategy_config.get("pair_construction", {})
        loss = strategy_config.get("loss", {})

        feats: List[float] = []

        # ── Encoder (3-d) ──────────────────────────────────────────────
        feats.append(encoder_config.get("n_layers", 10) / 10.0)
        feats.append(encoder_config.get("hidden_dim", 64) / 128.0)
        feats.append(encoder_config.get("output_dim", 320) / 320.0)

        # ── Augmentation probabilities (6-d) ───────────────────────────
        for aug_name in ["resize", "rescale", "jitter", "point_mask", "freq_mask", "crop"]:
            feats.append(float(aug.get(aug_name, 0.0)))

        # ── Augmentation order one-hot (5-d) ──────────────────────────
        order = int(aug.get("order", 0))
        feats.extend([1.0 if i == order else 0.0 for i in range(5)])

        # ── Embedding transform (5-d) ─────────────────────────────────
        feats.append(float(emb.get("jitter_p", 0.0)))
        feats.append(float(emb.get("mask_p", 0.0)))
        feats.extend(_onehot(str(emb.get("norm_type", "none")), _NORM_TYPES))

        # ── Pair construction (6-d) ───────────────────────────────────
        feats.append(1.0 if pair.get("temporal", False) else 0.0)
        feats.append(1.0 if pair.get("cross_scale", False) else 0.0)
        feats.append(int(pair.get("kernel_size", 0)) / 5.0)
        feats.extend(_onehot(str(pair.get("pool_op", "avg")), _POOL_OPS))
        feats.append(1.0 if pair.get("adj_neighbor", False) else 0.0)

        # ── Loss (6-d) ────────────────────────────────────────────────
        feats.extend(_onehot(str(loss.get("type", "infonce")), _LOSS_TYPES))
        feats.extend(_onehot(str(loss.get("sim_func", "dot")), _SIM_FUNCS))
        temp = float(loss.get("temperature", 1.0))
        feats.append(math.log10(max(temp, 1e-9)) / 2.0)

        device = next(self.mlp.parameters()).device
        return torch.tensor(feats, dtype=torch.float32, device=device)
