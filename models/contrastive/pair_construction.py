"""Contrastive pair construction and loss computation.

Three contrast types are supported (following AutoCLS):

  Instance   — same sample, different views; in-batch negatives (always on)
  Temporal   — same time step across views; within-sample time negatives
  Cross-scale— fine-resolution embeddings vs. their pooled coarse versions

``ContrastivePairConstructor`` orchestrates all active contrast types given
the strategy config and returns a per-type loss dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Union

from .losses import InfoNCELoss, TripletLoss, compute_similarity


# ---------------------------------------------------------------------------
# Hierarchical pooling helper
# ---------------------------------------------------------------------------

def hierarchical_pooling(
    h: Tensor,
    kernel_size: int,
    pool_op: str,
) -> List[Tensor]:
    """Recursively pool *h* along the time axis.

    Args:
        h: Shape (B, T, D).
        kernel_size: Pooling kernel / stride.  ``0`` → return ``[h]``
            immediately (no pooling).
        pool_op: ``'avg'`` or ``'max'``.

    Returns:
        List of tensors at decreasing time resolutions:
        ``[h^(0), h^(1), …]`` where ``h^(0) = h`` and
        ``T_{s+1} = T_s // kernel_size``.  Pooling stops when
        ``T_s < kernel_size``.
    """
    if kernel_size == 0:
        return [h]

    scales: List[Tensor] = [h]
    current = h  # (B, T, D)

    while current.shape[1] >= kernel_size:
        # F.avg_pool1d / max_pool1d expect (B, C, T)
        x_t = current.permute(0, 2, 1)  # (B, D, T)
        if pool_op == "avg":
            pooled_t = F.avg_pool1d(x_t, kernel_size=kernel_size, stride=kernel_size)
        else:
            pooled_t = F.max_pool1d(x_t, kernel_size=kernel_size, stride=kernel_size)
        current = pooled_t.permute(0, 2, 1)  # (B, T', D)

        if current.shape[1] < 1:
            break
        scales.append(current)

        # Stop early if we only have 1 time step left — no meaningful pooling.
        if current.shape[1] < kernel_size:
            break

    return scales


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batch_sim_matrix(
    a: Tensor,
    b: Tensor,
    sim_func: str,
    temperature: float,
) -> Tensor:
    """Compute pairwise similarity matrix between all (i,j) pairs.

    Args:
        a: Shape (B, M, D).
        b: Shape (B, N, D).
        sim_func: ``'dot'``, ``'cosine'``, or ``'euclidean'``.
        temperature: Divides the raw similarity scores.

    Returns:
        Shape (B, M, N).
    """
    if sim_func == "dot":
        sim = torch.bmm(a, b.transpose(1, 2))
    elif sim_func == "cosine":
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        sim = torch.bmm(a_n, b_n.transpose(1, 2))
    elif sim_func == "euclidean":
        # -||a_i - b_j||_2 via expanded squared distance
        a2 = (a ** 2).sum(-1, keepdim=True)          # (B, M, 1)
        b2 = (b ** 2).sum(-1).unsqueeze(1)            # (B, 1, N)
        ab = torch.bmm(a, b.transpose(1, 2))          # (B, M, N)
        dist = torch.sqrt((a2 + b2 - 2.0 * ab).clamp(min=0.0))
        sim = -dist
    else:
        raise ValueError(f"Unknown sim_func: {sim_func!r}")

    return sim / temperature


def _adj_positive_mask(T: int, device: torch.device) -> Tensor:
    """Build a (T, T) boolean mask for adjacent positive pairs.

    Positions t and t' are positive when |t - t'| <= 1.

    Args:
        T: Sequence length.
        device: Target device.

    Returns:
        Boolean tensor of shape (T, T).
    """
    idx = torch.arange(T, device=device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (T, T)
    return dist <= 1


def _multi_pos_nce(sim: Tensor, pos_mask: Tensor) -> Tensor:
    """Supervised / multi-positive InfoNCE loss.

    For each row (anchor), all positions where ``pos_mask`` is True are
    treated as positives using the ``log(sum_pos / sum_all)`` formulation.

    Args:
        sim: Logits of shape (B, T, T) (already divided by temperature).
        pos_mask: Boolean mask of shape (T, T) or (B, T, T).

    Returns:
        Scalar loss.
    """
    sim = sim.clamp(-100.0, 100.0)

    if pos_mask.dim() == 2:
        pos_mask = pos_mask.unsqueeze(0)  # (1, T, T)

    log_sum_all = torch.logsumexp(sim, dim=-1)              # (B, T)
    # Sum of exp over positives
    pos_sim = sim.masked_fill(~pos_mask, -1e9)
    log_sum_pos = torch.logsumexp(pos_sim, dim=-1)          # (B, T)

    # Normalise by the number of positives per row to avoid scale issues
    n_pos = pos_mask.float().sum(dim=-1).clamp(min=1.0)     # (B, T) or (1, T)
    loss = -(log_sum_pos - log_sum_all) / n_pos
    return loss.mean()


# ---------------------------------------------------------------------------
# ContrastivePairConstructor
# ---------------------------------------------------------------------------

class ContrastivePairConstructor(nn.Module):
    """Construct contrastive pairs and compute losses from (h1, h2).

    This module holds no trainable parameters; it is an ``nn.Module`` purely
    for consistent device/state management.

    Args:
        config: Pair-construction sub-dict of the strategy config::

            {
                'instance':    True,        # always True, not searched
                'temporal':    bool,
                'cross_scale': bool,
                'kernel_size': int,         # 0 → no hierarchical pooling
                'pool_op':     str,         # 'avg' | 'max'
                'adj_neighbor':bool,
            }
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.temporal: bool = bool(config.get("temporal", False))
        self.cross_scale: bool = bool(config.get("cross_scale", False))
        self.kernel_size: int = int(config.get("kernel_size", 0))
        self.pool_op: str = str(config.get("pool_op", "avg"))
        self.adj_neighbor: bool = bool(config.get("adj_neighbor", False))

    # ------------------------------------------------------------------
    # Instance contrast
    # ------------------------------------------------------------------

    def instance_loss(
        self,
        h1: Tensor,
        h2: Tensor,
        loss_fn: Union[InfoNCELoss, TripletLoss],
    ) -> Tensor:
        """Compute instance-level contrastive loss.

        Time-pool both views and contrast the resulting (B, D) vectors.
        Positive: (h1[i], h2[i]).  Negatives: h2[j] for j ≠ i.

        Args:
            h1: Shape (B, T, D).
            h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Scalar loss.
        """
        B = h1.shape[0]
        anchor = h1.mean(dim=1)    # (B, D)
        positive = h2.mean(dim=1)  # (B, D)

        if isinstance(loss_fn, InfoNCELoss):
            if B < 2:
                return torch.tensor(0.0, device=h1.device, requires_grad=True)
            # Build (B, B-1, D) negatives by excluding each sample's own positive
            D = anchor.shape[-1]
            pos_exp = positive.unsqueeze(0).expand(B, B, D)              # (B, B, D)
            mask = ~torch.eye(B, dtype=torch.bool, device=h1.device)     # (B, B)
            negatives = pos_exp[mask].reshape(B, B - 1, D)               # (B, B-1, D)
            return loss_fn(anchor, positive, negatives)

        else:  # TripletLoss
            if B < 2:
                return torch.tensor(0.0, device=h1.device, requires_grad=True)
            # Use the next sample in the batch as the hard negative.
            negative = torch.roll(positive, shifts=1, dims=0)            # (B, D)
            return loss_fn(anchor, positive, negative)

    # ------------------------------------------------------------------
    # Temporal contrast
    # ------------------------------------------------------------------

    def temporal_loss(
        self,
        h1: Tensor,
        h2: Tensor,
        loss_fn: Union[InfoNCELoss, TripletLoss],
    ) -> Tensor:
        """Compute temporal contrastive loss.

        For each time step t, the positive is the same t in the other view.
        Negatives are all other time steps within the same sample.

        Uses vectorised batch matrix multiplication for efficiency.

        Args:
            h1: Shape (B, T, D).
            h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Scalar loss, or zero if T < 3.
        """
        B, T, D = h1.shape
        if T < 3:
            return h1.new_zeros(1).squeeze().requires_grad_(False)

        if isinstance(loss_fn, InfoNCELoss):
            sim_func = loss_fn.sim_func
            temp = loss_fn.temperature

            # ── Similarity matrices (B, T, T) ─────────────────────────
            sim_12 = _batch_sim_matrix(h1, h2, sim_func, temp)  # h1→h2
            sim_21 = _batch_sim_matrix(h2, h1, sim_func, temp)  # h2→h1

            if self.adj_neighbor:
                pos_mask = _adj_positive_mask(T, h1.device)     # (T, T)
                loss_12 = _multi_pos_nce(sim_12, pos_mask)
                loss_21 = _multi_pos_nce(sim_21, pos_mask)
            else:
                # Standard: positive = diagonal
                labels = torch.arange(T, device=h1.device)           # (T,)
                labels = labels.unsqueeze(0).expand(B, T).reshape(B * T)  # (B*T,)

                logits_12 = sim_12.reshape(B * T, T).clamp(-100.0, 100.0)
                logits_21 = sim_21.reshape(B * T, T).clamp(-100.0, 100.0)

                loss_12 = F.cross_entropy(logits_12, labels)
                loss_21 = F.cross_entropy(logits_21, labels)

            return (loss_12 + loss_21) * 0.5

        else:  # TripletLoss
            # anchor: h1[b, t]; positive: h2[b, t]; negative: h1[b, t shifted]
            shift = max(1, T // 2)
            neg = torch.roll(h1, shifts=shift, dims=1)       # (B, T, D)

            anchor   = h1.reshape(B * T, D)
            positive = h2.reshape(B * T, D)
            negative = neg.reshape(B * T, D)
            return loss_fn(anchor, positive, negative)

    # ------------------------------------------------------------------
    # Cross-scale contrast
    # ------------------------------------------------------------------

    def cross_scale_loss(
        self,
        h1: Tensor,
        h2: Tensor,
        loss_fn: Union[InfoNCELoss, TripletLoss],
    ) -> Tensor:
        """Compute cross-scale contrastive loss via hierarchical pooling.

        For each consecutive pair of scales (fine h^s, coarse h^(s+1)):
          - Every fine position t_s is an anchor.
          - Its positive is the corresponding coarse position t_s // k.
          - Negatives are all other coarse positions in the same sample.

        Args:
            h1: Shape (B, T, D).
            h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Scalar loss, or zero when no valid scale pairs exist.
        """
        if self.kernel_size == 0:
            return h1.new_zeros(1).squeeze().requires_grad_(False)

        B, T, D = h1.shape

        scales1 = hierarchical_pooling(h1, self.kernel_size, self.pool_op)
        scales2 = hierarchical_pooling(h2, self.kernel_size, self.pool_op)

        # Need at least 2 scales for cross-scale contrast.
        if len(scales1) < 2:
            return h1.new_zeros(1).squeeze().requires_grad_(False)

        # Limit to 2 scale levels for short sequences (T < 100) to avoid
        # degenerate coarse representations.
        max_pairs = 2 if T < 100 else len(scales1) - 1
        n_pairs = min(max_pairs, len(scales1) - 1)

        loss_total = h1.new_zeros(1).squeeze()
        n_valid = 0

        for s in range(n_pairs):
            h_fine1   = scales1[s]       # (B, T_s, D)
            h_coarse1 = scales1[s + 1]   # (B, T_c, D)  T_c = T_s // k
            h_fine2   = scales2[s]
            h_coarse2 = scales2[s + 1]

            T_fine  = h_fine1.shape[1]
            T_coarse = h_coarse1.shape[1]

            if T_coarse < 2:
                # Cross-scale contrast is ill-defined with only 1 coarse step.
                continue

            # Labels: fine position t maps to coarse position t // kernel_size
            labels_np = torch.arange(T_fine, device=h1.device) // self.kernel_size
            labels_np = labels_np.clamp(max=T_coarse - 1)  # safety clamp

            if isinstance(loss_fn, InfoNCELoss):
                sim_func = loss_fn.sim_func
                temp = loss_fn.temperature

                # sim[b, t_fine, t_coarse] shape: (B, T_fine, T_coarse)
                sim_12 = _batch_sim_matrix(h_fine1, h_coarse2, sim_func, temp)
                sim_21 = _batch_sim_matrix(h_fine2, h_coarse1, sim_func, temp)

                lbl = labels_np.unsqueeze(0).expand(B, T_fine).reshape(B * T_fine)
                log12 = sim_12.reshape(B * T_fine, T_coarse).clamp(-100.0, 100.0)
                log21 = sim_21.reshape(B * T_fine, T_coarse).clamp(-100.0, 100.0)

                loss_s = 0.5 * (F.cross_entropy(log12, lbl) + F.cross_entropy(log21, lbl))

            else:  # TripletLoss
                # For each fine position pick the correct coarse positive and a
                # shifted coarse negative.
                pos1 = h_coarse1[:, labels_np, :]   # (B, T_fine, D)
                pos2 = h_coarse2[:, labels_np, :]

                neg_idx = (labels_np + T_coarse // 2) % T_coarse
                neg1 = h_coarse1[:, neg_idx, :]     # (B, T_fine, D)
                neg2 = h_coarse2[:, neg_idx, :]

                anchor1   = h_fine1.reshape(B * T_fine, D)
                positive1 = pos2.reshape(B * T_fine, D)
                negative1 = neg2.reshape(B * T_fine, D)

                anchor2   = h_fine2.reshape(B * T_fine, D)
                positive2 = pos1.reshape(B * T_fine, D)
                negative2 = neg1.reshape(B * T_fine, D)

                loss_s = 0.5 * (
                    loss_fn(anchor1, positive1, negative1)
                    + loss_fn(anchor2, positive2, negative2)
                )

            loss_total = loss_total + loss_s
            n_valid += 1

        if n_valid == 0:
            return h1.new_zeros(1).squeeze().requires_grad_(False)

        return loss_total / n_valid

    # ------------------------------------------------------------------
    # Unified dispatch
    # ------------------------------------------------------------------

    def compute_all_losses(
        self,
        h1: Tensor,
        h2: Tensor,
        loss_fn: Union[InfoNCELoss, TripletLoss],
    ) -> Dict[str, Tensor]:
        """Compute all active contrast losses.

        Instance contrast is always active.

        Args:
            h1: Shape (B, T, D).
            h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Dict mapping contrast type name to scalar loss.
        """
        losses: Dict[str, Tensor] = {}

        losses["instance"] = self.instance_loss(h1, h2, loss_fn)

        if self.temporal:
            losses["temporal"] = self.temporal_loss(h1, h2, loss_fn)

        if self.cross_scale:
            losses["cross_scale"] = self.cross_scale_loss(h1, h2, loss_fn)

        return losses
