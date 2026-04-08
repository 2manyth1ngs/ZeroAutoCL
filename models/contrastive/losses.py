"""Contrastive loss functions for ZeroAutoCL.

Provides:
  compute_similarity  — point-wise similarity (dot / cosine / euclidean)
  InfoNCELoss         — in-batch or explicit-negatives NCE loss
  TripletLoss         — margin ranking loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Similarity function
# ---------------------------------------------------------------------------

def compute_similarity(a: Tensor, b: Tensor, method: str) -> Tensor:
    """Compute element-wise similarity between paired vectors.

    Args:
        a: Tensor of shape (..., D).
        b: Tensor of shape (..., D), same shape as *a*.
        method: One of ``'dot'``, ``'cosine'``, ``'euclidean'``.

    Returns:
        Similarity scores of shape (...).  Higher values indicate greater
        similarity for all three methods (euclidean returns negative distance).

    Raises:
        ValueError: If *method* is not recognised.
    """
    if method == "dot":
        return (a * b).sum(dim=-1)
    elif method == "cosine":
        return F.cosine_similarity(a, b, dim=-1)
    elif method == "euclidean":
        # Return negative distance so that "higher = more similar" holds.
        return -torch.norm(a - b, dim=-1)
    else:
        raise ValueError(f"Unknown similarity method: {method!r}")


# ---------------------------------------------------------------------------
# InfoNCE Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """InfoNCE (Noise-Contrastive Estimation) loss.

    Treats the problem as multi-class classification: for each anchor the
    positive is class 0 and the N negatives are classes 1 … N.

    Args:
        sim_func: Similarity function — ``'dot'``, ``'cosine'``, or
            ``'euclidean'``.
        temperature: Softmax temperature τ > 0.
    """

    def __init__(self, sim_func: str = "dot", temperature: float = 0.1) -> None:
        super().__init__()
        self.sim_func = sim_func
        self.temperature = temperature

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negatives: Tensor,
    ) -> Tensor:
        """Compute InfoNCE loss.

        Args:
            anchor:    Shape (B, D).
            positive:  Shape (B, D).
            negatives: Shape (B, N_neg, D).

        Returns:
            Scalar loss.
        """
        # ── Positive similarity ──────────────────────────────────────────
        # (B,)
        pos_sim = compute_similarity(anchor, positive, self.sim_func) / self.temperature

        # ── Negative similarities ────────────────────────────────────────
        # Broadcast anchor to (B, N_neg, D) for element-wise similarity
        B, N_neg, D = negatives.shape
        anchor_exp = anchor.unsqueeze(1).expand(B, N_neg, D)
        # (B, N_neg)
        neg_sims = compute_similarity(anchor_exp, negatives, self.sim_func) / self.temperature

        # ── Logits: positive first, then negatives ───────────────────────
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # (B, 1+N_neg)
        logits = logits.clamp(-100.0, 100.0)

        # ── Cross-entropy with label = 0 (positive is always index 0) ───
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Triplet Loss
# ---------------------------------------------------------------------------

class TripletLoss(nn.Module):
    """Soft-margin triplet ranking loss.

    Penalises configurations where ``sim(anchor, negative) + margin ≥
    sim(anchor, positive)``.

    Args:
        sim_func: Similarity function — ``'dot'``, ``'cosine'``, or
            ``'euclidean'``.
        margin: Minimum required gap between positive and negative
            similarity.  Default ``1.0``.
    """

    def __init__(self, sim_func: str = "dot", margin: float = 1.0) -> None:
        super().__init__()
        self.sim_func = sim_func
        self.margin = margin

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """Compute triplet loss.

        Args:
            anchor:   Shape (B, D).
            positive: Shape (B, D).
            negative: Shape (B, D).

        Returns:
            Scalar loss.
        """
        sim_pos = compute_similarity(anchor, positive, self.sim_func)
        sim_neg = compute_similarity(anchor, negative, self.sim_func)
        # sim is "higher = more similar", so we penalise sim_neg ≥ sim_pos.
        loss = F.relu(self.margin + sim_neg - sim_pos)
        return loss.mean()
