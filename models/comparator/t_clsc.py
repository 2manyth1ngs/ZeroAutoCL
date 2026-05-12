"""T-CLSC: Task-aware CL Strategy Comparator.

A pairwise ranking model that takes two candidate configurations and a
**set-based** task-feature tensor and outputs P(A > B | task) ∈ (0, 1).

Architecture (post P0-2 fix, 2026-05-12)::

    z_A    = candidate_encoder.encode(config_A)            # (H,)
    z_B    = candidate_encoder.encode(config_B)            # (H,)
    z_T    = task_encoder(task_features)                   # (H,)   ← Set Transformer
    feat   = [z_A, z_B, z_T]                               # (3H,)
    logit  = comparison_head(feat)                          # (1,)
    P(A>B) = σ(logit)

The previous architecture concatenated ``[z_A, z_B, z_A−z_B, z_A⊙z_B, z_T]``
(5H), but the ``z_A−z_B`` and ``z_A⊙z_B`` terms are **task-agnostic** — they
let the head fit pair labels using only candidate features, never forcing
the gradient back through ``task_encoder``.  Combined with per-task batching
in the trainer, this caused the task encoder to collapse to a near-constant
mapping (see ``Debug/comparator_bug_2026_05_12.md`` §3 for the full diagnosis
and §9 for the P0-0 confirmation that the task-feature INPUT carries 57×
random-baseline source identity — the problem was always the head, not the
data).  The 3H head forces the comparison to depend on ``z_T``: pred(A,B|T₁)
must differ from pred(A,B|T₂) whenever T₁ ≠ T₂, otherwise the loss on
mixed-task batches won't drop.

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

        # P0-2 fix (2026-05-12): comparison-head input is [z_A, z_B, z_task]
        # (3H), NOT [z_A, z_B, z_A-z_B, z_A*z_B, z_task] (5H).  The diff/prod
        # terms are task-agnostic; their presence let the head fit batches
        # without ever pushing gradient through ``task_encoder``.  Removing
        # them forces the head to USE ``z_task``, which in turn forces the
        # task encoder to learn task-discriminative features.
        # NOTE: this is a breaking change vs. older checkpoints (5H Linear
        # weight cannot be loaded into the 3H head).  Pre-fix comparators
        # must be retrained from scratch.
        self.comparison_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
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

        # P0-2: 3H head, no z_a-z_b / z_a*z_b shortcut.
        feat = torch.cat([z_a, z_b, z_task])                                # (3H,)
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

        # P0-2: 3H head — z_a-z_b / z_a*z_b shortcuts removed.
        feat = torch.cat([z_a, z_b, z_task], dim=1)                                 # (N, 3H)
        logit = self.comparison_head(feat)                                           # (N, 1)
        return torch.sigmoid(logit).squeeze(-1)                                     # (N,)

    # ------------------------------------------------------------------
    # P0-1: Mixed-task batched forward — for training
    # ------------------------------------------------------------------

    def forward_batch_multitask(
        self,
        encoder_configs_a: List[Dict[str, int]],
        strategies_a: List[Dict],
        encoder_configs_b: List[Dict[str, int]],
        strategies_b: List[Dict],
        task_features_list: List[Tensor],
        task_ids: Optional[List[str]] = None,
    ) -> Tensor:
        """Batched pairwise comparison where each pair has its OWN task feature.

        This is the training-time forward — it lets a batch mix pairs from
        many different tasks, which forces the task encoder to produce
        task-discriminative ``z_task`` to fit the BCE loss across the batch.
        Without this (i.e. the per-task batching of pre-P0-1), every pair in
        a batch sees the same ``z_task``, and the gradient never pushes the
        task encoder away from a near-constant output.

        Args:
            encoder_configs_a: N encoder config dicts (candidates A).
            strategies_a: N strategy config dicts (candidates A).
            encoder_configs_b: N encoder config dicts (candidates B).
            strategies_b: N strategy config dicts (candidates B).
            task_features_list: List of N task feature tensors.  Each tensor
                is shape ``(N_set, seq_len, D)`` or ``(N_set, D)``.  Pairs
                from the same task may share the same tensor — see
                ``task_ids`` for the dedup-optimised path.
            task_ids: Optional list of N string task identifiers used to
                dedup ``task_encoder`` calls when many pairs come from the
                same task in one batch.  When ``None`` we fall back to
                tensor-identity dedup (``id(tensor)``) which works when the
                caller reuses the same task-feature object across pairs.
                Skipping dedup entirely is correct but ~K× slower when a
                batch of N pairs contains only K unique tasks.

        Returns:
            Tensor of shape ``(N,)`` with P(A_i > B_i | task_i) values.
        """
        N = len(encoder_configs_a)
        z_a = self.candidate_encoder.encode_batch(encoder_configs_a, strategies_a)  # (N, H)
        z_b = self.candidate_encoder.encode_batch(encoder_configs_b, strategies_b)  # (N, H)
        device = z_a.device

        # Dedup task-encoder calls: build a unique-task index so identical
        # task features pass through ``task_encoder`` only once per batch.
        # In a typical training batch of 64 pairs covering ~5-15 unique
        # tasks this cuts task-encode cost by ~4-12×.  Falls back to per-pair
        # encoding when the caller doesn't supply ``task_ids``.
        if task_ids is not None and len(task_ids) == N:
            unique_ids: Dict[str, int] = {}
            mapping: List[int] = []
            unique_feats: List[Tensor] = []
            for i, tid in enumerate(task_ids):
                if tid not in unique_ids:
                    unique_ids[tid] = len(unique_feats)
                    unique_feats.append(task_features_list[i])
                mapping.append(unique_ids[tid])
            # Encode unique tasks one at a time (task_encoder expects a
            # single (N_set, seq_len, D) tensor — it's not batched over a
            # leading set-of-sets dim).
            z_task_unique = torch.stack(
                [self.task_encoder(tf.to(device)) for tf in unique_feats], dim=0,
            )                                                                       # (U, H)
            z_task = z_task_unique[torch.tensor(mapping, device=device)]            # (N, H)
        else:
            # Slow path — no dedup.  Each pair pays a full task_encoder
            # forward.  Only used by tests / callers that don't track ids.
            z_task = torch.stack(
                [self.task_encoder(tf.to(device)) for tf in task_features_list],
                dim=0,
            )                                                                       # (N, H)

        feat = torch.cat([z_a, z_b, z_task], dim=1)                                 # (N, 3H)
        logit = self.comparison_head(feat)                                           # (N, 1)
        return torch.sigmoid(logit).squeeze(-1)                                     # (N,)
