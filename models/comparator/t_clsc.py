"""T-CLSC: Task-aware CL Strategy Comparator.

A pairwise ranking model that takes two candidate configurations and a
**set-based** task-feature tensor and outputs P(A > B | task) ∈ (0, 1).

Architecture::

    z_A    = candidate_encoder.encode(config_A)            # (H,)
    z_B    = candidate_encoder.encode(config_B)            # (H,)
    z_T    = task_encoder(task_features)                   # (H,)   ← Set Transformer
    feat   = [z_A, z_B, z_A − z_B, z_A ⊙ z_B, z_T]         # (5H,)
    logit  = comparison_head(feat)                          # (1,)
    P(A>B) = σ(logit)

Compared with the previous version, ``task_mlp`` (a plain 2-layer MLP that
consumed a flat 327-d vector of pooled statistics) has been replaced by
:class:`_SetTaskFeatureEncoder` — it accepts the AutoCTS++-style set tensor
of shape ``(N_set, seq_len, D)`` directly and aggregates it via the existing
:class:`~models.comparator.set_encoder.SetTransformerEncoder`.

The constructor signature is unchanged (``task_dim`` now refers to the
*per-element* repr dim instead of the total feature dim — equal to
:data:`TASK_FEATURE_DIM` in both old and new code, so the default value is
backward-compatible).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.search_space.space_encoder import CandidateEncoder, RAW_DIM
from .set_encoder import SetTransformerEncoder
from .task_feature import TASK_FEATURE_DIM


class _SetTaskFeatureEncoder(nn.Module):
    """Encode a (N_set, seq_len, D) task feature tensor into a (H,) vector.

    Pipeline:
      1. Mean-pool over ``seq_len`` → ``(N_set, D)``.
      2. Add batch dim → ``(1, N_set, D)``.
      3. :class:`SetTransformerEncoder` (ISAB×2 → PMA(k=1) → Linear) → ``(1, H)``.
      4. Squeeze → ``(H,)``.

    Args:
        repr_dim: Per-element repr dim ``D``.
        hidden_dim: Output dim ``H`` (the comparator's hidden size).
        n_inducing: ISAB inducing-point count.
        n_heads:    Multihead-attention heads.
    """

    def __init__(
        self,
        repr_dim: int,
        hidden_dim: int,
        n_inducing: int = 16,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.set_encoder = SetTransformerEncoder(
            d_in=repr_dim,
            d_out=hidden_dim,
            d_hidden=hidden_dim,
            n_inducing=n_inducing,
            n_heads=n_heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode one task's set feature.

        Args:
            x: Shape ``(N_set, seq_len, D)``.  A 2-d ``(N_set, D)`` tensor is
               also accepted (skips the seq_len pooling step) for cases where
               the caller has already pooled.

        Returns:
            Shape ``(hidden_dim,)``.
        """
        if x.dim() == 3:
            z = x.mean(dim=-2)        # (N_set, D)
        elif x.dim() == 2:
            z = x                      # (N_set, D)
        else:
            raise ValueError(
                f"_SetTaskFeatureEncoder expects (N_set, seq_len, D) or "
                f"(N_set, D); got {tuple(x.shape)}",
            )
        z = z.unsqueeze(0)             # (1, N_set, D)
        return self.set_encoder(z).squeeze(0)   # (hidden_dim,)


class TCLSC(nn.Module):
    """Task-aware CL Strategy Comparator.

    Args:
        candidate_dim: Raw feature dimension for candidate encoding.
        task_dim: **Per-element** dim of the task-feature set (default
            :data:`TASK_FEATURE_DIM` = 128 — i.e. the encoder output_dim
            used by the precompute script).
        hidden_dim: Hidden size used by the candidate encoder, the task
            encoder output, and the comparison head input.
    """

    def __init__(
        self,
        candidate_dim: int = RAW_DIM,
        task_dim: int = TASK_FEATURE_DIM,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared candidate encoder (same weights for A and B).
        self.candidate_encoder = CandidateEncoder(candidate_dim, hidden_dim)

        # Set-based task feature encoder (replaces the old task_mlp MLP).
        # Was: ``Sequential(Linear(327, H), ReLU, Linear(H, H))`` which
        # consumed a flat statistics vector — see CLAUDE_DEBUG.md for why it
        # was insufficient.
        self.task_encoder = _SetTaskFeatureEncoder(
            repr_dim=task_dim, hidden_dim=hidden_dim,
        )

        # Comparison head: input = [z_A, z_B, z_A-z_B, z_A*z_B, z_task].
        self.comparison_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    # Forward — single pair
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_config_a: Dict[str, int],
        strategy_a: Dict,
        encoder_config_b: Dict[str, int],
        strategy_b: Dict,
        task_features: Tensor,
    ) -> Tensor:
        """Predict P(A > B | task).

        Args:
            encoder_config_a: Encoder config dict for candidate A.
            strategy_a: Strategy config dict for candidate A.
            encoder_config_b: Encoder config dict for candidate B.
            strategy_b: Strategy config dict for candidate B.
            task_features: Task feature tensor of shape ``(N_set, seq_len, D)``
                (or ``(N_set, D)`` if seq_len has already been pooled).

        Returns:
            Scalar tensor in (0, 1).
        """
        z_a = self.candidate_encoder.encode(encoder_config_a, strategy_a)   # (H,)
        z_b = self.candidate_encoder.encode(encoder_config_b, strategy_b)   # (H,)
        z_task = self.task_encoder(task_features.to(z_a.device))            # (H,)

        feat = torch.cat([z_a, z_b, z_a - z_b, z_a * z_b, z_task])          # (5H,)
        logit = self.comparison_head(feat)                                  # (1,)
        return torch.sigmoid(logit).squeeze(-1)

    # ------------------------------------------------------------------
    # Batched forward — for efficient tournament ranking
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        encoder_configs_a: List[Dict[str, int]],
        strategies_a: List[Dict],
        encoder_configs_b: List[Dict[str, int]],
        strategies_b: List[Dict],
        task_features: Tensor,
    ) -> Tensor:
        """Batched pairwise comparison for N pairs sharing one task.

        Args:
            encoder_configs_a: List of N encoder config dicts (candidates A).
            strategies_a: List of N strategy config dicts (candidates A).
            encoder_configs_b: List of N encoder config dicts (candidates B).
            strategies_b: List of N strategy config dicts (candidates B).
            task_features: Task feature tensor of shape ``(N_set, seq_len, D)``
                (or ``(N_set, D)`` already-pooled), shared across the batch.

        Returns:
            Tensor of shape ``(N,)`` with P(A_i > B_i | task) values.
        """
        N = len(encoder_configs_a)
        z_a = self.candidate_encoder.encode_batch(encoder_configs_a, strategies_a)  # (N, H)
        z_b = self.candidate_encoder.encode_batch(encoder_configs_b, strategies_b)  # (N, H)
        device = z_a.device
        z_task = self.task_encoder(task_features.to(device))                        # (H,)
        z_task = z_task.unsqueeze(0).expand(N, -1)                                  # (N, H)

        feat = torch.cat([z_a, z_b, z_a - z_b, z_a * z_b, z_task], dim=1)           # (N, 5H)
        logit = self.comparison_head(feat)                                           # (N, 1)
        return torch.sigmoid(logit).squeeze(-1)                                     # (N,)
