"""T-CLSC: Task-aware CL Strategy Comparator.

A pairwise ranking model that takes two candidate configurations and a
task-feature vector and outputs P(A > B | task) ∈ (0, 1).

Architecture::

    z_A  = candidate_encoder.encode(config_A)         # (H,)
    z_B  = candidate_encoder.encode(config_B)         # (H,)
    z_T  = task_mlp(task_features)                    # (H,)
    feat = [z_A, z_B, z_A − z_B, z_A ⊙ z_B, z_T]   # (5H,)
    logit = comparison_head(feat)                     # (1,)
    P(A > B) = σ(logit)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.search_space.space_encoder import CandidateEncoder, RAW_DIM
from .task_feature import TASK_FEATURE_DIM


class TCLSC(nn.Module):
    """Task-aware CL Strategy Comparator.

    Args:
        candidate_dim: Raw feature dimension for candidate encoding.
        task_dim: Dimension of the task-feature vector produced by
            :class:`TaskFeatureExtractor`.
        hidden_dim: Hidden size used by candidate encoder and task MLP.
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

        # Task feature projection.
        self.task_mlp = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
            task_features: Task feature vector, shape ``(D_task,)``.

        Returns:
            Scalar tensor in (0, 1).
        """
        z_a = self.candidate_encoder.encode(encoder_config_a, strategy_a)   # (H,)
        z_b = self.candidate_encoder.encode(encoder_config_b, strategy_b)   # (H,)
        z_task = self.task_mlp(task_features.to(z_a.device))                # (H,)

        feat = torch.cat([z_a, z_b, z_a - z_b, z_a * z_b, z_task])         # (5H,)
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
        """Batched pairwise comparison for N pairs.

        The task feature is shared across the batch (same task).

        Args:
            encoder_configs_a: List of N encoder config dicts (candidates A).
            strategies_a: List of N strategy config dicts (candidates A).
            encoder_configs_b: List of N encoder config dicts (candidates B).
            strategies_b: List of N strategy config dicts (candidates B).
            task_features: Task feature vector, shape ``(D_task,)``.

        Returns:
            Tensor of shape ``(N,)`` with P(A_i > B_i | task) values.
        """
        N = len(encoder_configs_a)
        z_a = self.candidate_encoder.encode_batch(encoder_configs_a, strategies_a)  # (N, H)
        z_b = self.candidate_encoder.encode_batch(encoder_configs_b, strategies_b)  # (N, H)
        device = z_a.device
        z_task = self.task_mlp(task_features.to(device)).unsqueeze(0).expand(N, -1) # (N, H)

        feat = torch.cat([z_a, z_b, z_a - z_b, z_a * z_b, z_task], dim=1)          # (N, 5H)
        logit = self.comparison_head(feat)                                          # (N, 1)
        return torch.sigmoid(logit).squeeze(-1)                                     # (N,)
