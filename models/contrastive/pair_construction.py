"""Contrastive pair construction and loss computation.

Three contrast types are supported (following AutoCLS, which itself follows
TS2Vec for the multi-scale mechanism):

  Instance    — same-time-step, cross-batch contrast (always on). At each
                time step t the anchor z1[i,t] is pulled towards z2[i,t] and
                pushed away from every other sample in both views
                (2B-2 negatives — TS2Vec formulation).
  Temporal    — same-sample, cross-time contrast. For each time step t the
                anchor z1[b,t] is pulled towards z2[b,t] and pushed away
                from every other time step in both views of the same sample
                (2T-2 negatives — TS2Vec formulation).
  Cross-scale — fine-resolution embeddings vs. their pooled coarse versions.

Following AutoCLS §2.2 ("We follow [TS2Vec] and apply hierarchical pooling
over h1, h2 ... The above instance contrast and temporal contrast are
applied for all the scales"), when ``kernel_size > 0`` the instance and
temporal losses are additionally computed at every hierarchical scale
produced by recursive pooling with the given ``kernel_size`` / ``pool_op``,
and averaged across scales.  ``kernel_size = 0`` disables the hierarchical
loop (single-scale instance/temporal only).
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
    current = h

    while current.shape[1] >= kernel_size:
        x_t = current.permute(0, 2, 1)  # (B, D, T)
        if pool_op == "avg":
            pooled_t = F.avg_pool1d(x_t, kernel_size=kernel_size, stride=kernel_size)
        else:
            pooled_t = F.max_pool1d(x_t, kernel_size=kernel_size, stride=kernel_size)
        current = pooled_t.permute(0, 2, 1)  # (B, T', D)

        if current.shape[1] < 1:
            break
        scales.append(current)

        if current.shape[1] < kernel_size:
            break

    return scales


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _pairwise_sim(x: Tensor, y: Tensor, sim_func: str) -> Tensor:
    """Pairwise similarity between every row of *x* and every row of *y*.

    Args:
        x: Shape (..., N, D).
        y: Shape (..., M, D). Leading dims must match.
        sim_func: ``'dot'``, ``'cosine'``, ``'euclidean'``, or ``'distance'``
            (``'distance'`` is an AutoCLS-terminology alias for
            ``'euclidean'``).

    Returns:
        Tensor of shape (..., N, M). Higher = more similar for all funcs
        (euclidean / distance return negative L2 distance).
    """
    if sim_func == "dot":
        return torch.matmul(x, y.transpose(-1, -2))
    if sim_func == "cosine":
        xn = F.normalize(x, dim=-1)
        yn = F.normalize(y, dim=-1)
        return torch.matmul(xn, yn.transpose(-1, -2))
    if sim_func in ("euclidean", "distance"):
        x2 = (x ** 2).sum(-1, keepdim=True)
        y2 = (y ** 2).sum(-1, keepdim=True).transpose(-1, -2)
        xy = torch.matmul(x, y.transpose(-1, -2))
        # +1e-12 keeps sqrt's argument strictly positive so autograd
        # does not produce NaN gradients on the (zero-valued) diagonal
        # when x is y (per-timestep instance contrast).
        dist = (x2 + y2 - 2.0 * xy).clamp(min=0.0).add(1e-12).sqrt()
        return -dist
    raise ValueError(f"Unknown sim_func: {sim_func!r}")


def _adj_positive_mask(T: int, device: torch.device) -> Tensor:
    """(T, T) bool mask with True where |t - t'| <= 1."""
    idx = torch.arange(T, device=device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return dist <= 1


# ---------------------------------------------------------------------------
# TS2Vec-style InfoNCE primitives
# ---------------------------------------------------------------------------

def _ts2vec_instance_infonce(
    z1: Tensor,
    z2: Tensor,
    sim_func: str,
    temperature: float,
) -> Tensor:
    """Per-timestep instance contrast with 2B-2 in-batch negatives.

    Follows TS2Vec's ``instance_contrastive_loss`` but parameterised by
    similarity function and temperature as in AutoCLS Table 1.

    Args:
        z1: Shape (B, T, D).
        z2: Shape (B, T, D).
        sim_func: ``'dot'`` | ``'cosine'`` | ``'euclidean'``.
        temperature: τ > 0.

    Returns:
        Scalar loss. Returns 0 when B < 2.
    """
    B, T, _ = z1.shape
    if B < 2:
        return z1.new_zeros(())

    # (T, 2B, D): stack both views along batch, swap batch↔time.
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)
    sim = _pairwise_sim(z, z, sim_func) / temperature  # (T, 2B, 2B)

    # Drop the diagonal (self-similarity) via tril/triu trick → (T, 2B, 2B-1).
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = logits.clamp(-100.0, 100.0)
    neg_log_p = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    # Row i (view-1 anchor, sample i): positive = row index B+i in original,
    # column index after diagonal removal = B+i-1.
    # Row B+i (view-2 anchor, sample i): positive col = i.
    loss = (neg_log_p[:, i, B + i - 1].mean() + neg_log_p[:, B + i, i].mean()) / 2
    return loss


def _ts2vec_temporal_infonce(
    z1: Tensor,
    z2: Tensor,
    sim_func: str,
    temperature: float,
) -> Tensor:
    """Per-sample temporal contrast with 2T-2 within-sample negatives.

    Follows TS2Vec's ``temporal_contrastive_loss``. Same-view time steps
    other than the anchor act as negatives (in addition to cross-view
    non-matching time steps).

    Args:
        z1: Shape (B, T, D).
        z2: Shape (B, T, D).
        sim_func: ``'dot'`` | ``'cosine'`` | ``'euclidean'``.
        temperature: τ > 0.

    Returns:
        Scalar loss. Returns 0 when T < 2.
    """
    B, T, _ = z1.shape
    if T < 2:
        return z1.new_zeros(())

    z = torch.cat([z1, z2], dim=1)  # (B, 2T, D)
    sim = _pairwise_sim(z, z, sim_func) / temperature  # (B, 2T, 2T)

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = logits.clamp(-100.0, 100.0)
    neg_log_p = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (neg_log_p[:, t, T + t - 1].mean() + neg_log_p[:, T + t, t].mean()) / 2
    return loss


def _ts2vec_temporal_infonce_adj(
    z1: Tensor,
    z2: Tensor,
    sim_func: str,
    temperature: float,
) -> Tensor:
    """Temporal contrast with adjacent-neighbour multi-positive (AutoCLS).

    In the 2T structure, positives for an anchor at time t are the same
    time step in the other view *and* ``t ± 1`` in the same view (when
    they exist). Negatives are all remaining 2T-1 positions. Uses the
    multi-positive log-sum formulation.

    Args:
        z1: Shape (B, T, D).
        z2: Shape (B, T, D).
        sim_func: ``'dot'`` | ``'cosine'`` | ``'euclidean'``.
        temperature: τ > 0.

    Returns:
        Scalar loss. Returns 0 when T < 2.
    """
    B, T, _ = z1.shape
    if T < 2:
        return z1.new_zeros(())

    device = z1.device
    z = torch.cat([z1, z2], dim=1)  # (B, 2T, D)
    sim = _pairwise_sim(z, z, sim_func) / temperature  # (B, 2T, 2T)

    # Mask self on the diagonal so it is excluded from both positive and
    # negative sets via logsumexp.
    diag_idx = torch.arange(2 * T, device=device)
    sim = sim.clone()
    sim[:, diag_idx, diag_idx] = -1e9
    sim = sim.clamp(-100.0, 100.0)

    # Build (2T, 2T) positive mask.
    t = torch.arange(T, device=device)
    pos_mask = torch.zeros(2 * T, 2 * T, dtype=torch.bool, device=device)
    # Cross-view same-t positives (both directions).
    pos_mask[t, T + t] = True
    pos_mask[T + t, t] = True
    # Same-view adjacent positives in view 1.
    if T >= 2:
        pos_mask[t[:-1], t[:-1] + 1] = True
        pos_mask[t[1:], t[1:] - 1] = True
        # Same-view adjacent positives in view 2.
        pos_mask[T + t[:-1], T + t[:-1] + 1] = True
        pos_mask[T + t[1:], T + t[1:] - 1] = True

    log_sum_all = torch.logsumexp(sim, dim=-1)  # (B, 2T)
    pos_sim = sim.masked_fill(~pos_mask.unsqueeze(0), -1e9)
    log_sum_pos = torch.logsumexp(pos_sim, dim=-1)  # (B, 2T)

    n_pos = pos_mask.float().sum(dim=-1).clamp(min=1.0).unsqueeze(0)
    loss = -(log_sum_pos - log_sum_all) / n_pos
    return loss.mean()


# ---------------------------------------------------------------------------
# ContrastivePairConstructor
# ---------------------------------------------------------------------------

class ContrastivePairConstructor(nn.Module):
    """Construct contrastive pairs and compute losses from (h1, h2).

    Holds no trainable parameters; it is an ``nn.Module`` purely for
    consistent device/state management.

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
        max_temporal_len: Cap time length for the temporal and cross-scale
            similarity matrices to keep VRAM bounded.
    """

    def __init__(self, config: Dict, max_temporal_len: int = 200) -> None:
        super().__init__()
        self.temporal: bool = bool(config.get("temporal", False))
        self.cross_scale: bool = bool(config.get("cross_scale", False))
        self.kernel_size: int = int(config.get("kernel_size", 0))
        self.pool_op: str = str(config.get("pool_op", "avg"))
        self.adj_neighbor: bool = bool(config.get("adj_neighbor", False))
        self.max_temporal_len: int = max_temporal_len

    # ------------------------------------------------------------------
    # Single-scale instance / temporal losses
    # ------------------------------------------------------------------

    def instance_loss(
        self,
        h1: Tensor,
        h2: Tensor,
        loss_fn: Union[InfoNCELoss, TripletLoss],
    ) -> Tensor:
        """Per-timestep instance contrast at a single scale.

        InfoNCE path follows TS2Vec (2B-2 negatives). Triplet path uses a
        rolled batch as a per-timestep hard negative.

        Args:
            h1, h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Scalar loss.
        """
        B, T, D = h1.shape
        if B < 2:
            return h1.new_zeros(())

        if isinstance(loss_fn, InfoNCELoss):
            return _ts2vec_instance_infonce(
                h1, h2, loss_fn.sim_func, loss_fn.temperature
            )

        # TripletLoss: per-timestep anchor/pos/neg, neg = next sample in batch.
        neg = torch.roll(h2, shifts=1, dims=0)
        anchor = h1.reshape(B * T, D)
        positive = h2.reshape(B * T, D)
        negative = neg.reshape(B * T, D)
        return loss_fn(anchor, positive, negative)

    def temporal_loss(
        self,
        h1: Tensor,
        h2: Tensor,
        loss_fn: Union[InfoNCELoss, TripletLoss],
    ) -> Tensor:
        """Per-sample temporal contrast at a single scale.

        InfoNCE path follows TS2Vec (2T-2 negatives, same-view included).
        ``adj_neighbor=True`` expands the positive set to include ±1
        neighbours in the same view.

        Args:
            h1, h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Scalar loss, or zero if T < 2.
        """
        B, T, D = h1.shape
        if T < 2:
            return h1.new_zeros(())

        # Subsample time to cap O(B·T²) memory.
        if T > self.max_temporal_len:
            idx = torch.randperm(T, device=h1.device)[: self.max_temporal_len]
            idx = idx.sort().values
            h1 = h1[:, idx, :]
            h2 = h2[:, idx, :]
            T = self.max_temporal_len

        if isinstance(loss_fn, InfoNCELoss):
            if self.adj_neighbor:
                return _ts2vec_temporal_infonce_adj(
                    h1, h2, loss_fn.sim_func, loss_fn.temperature
                )
            return _ts2vec_temporal_infonce(
                h1, h2, loss_fn.sim_func, loss_fn.temperature
            )

        # TripletLoss: anchor=h1[b,t], positive=h2[b,t], negative=h1[b, t shifted].
        shift = max(1, T // 2)
        neg = torch.roll(h1, shifts=shift, dims=1)
        anchor = h1.reshape(B * T, D)
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
        """Cross-scale contrast between fine and pooled coarse embeddings.

        For each consecutive scale pair (fine, coarse), every fine position
        t is contrasted against all coarse positions in the same sample;
        the positive is ``t // kernel_size``.

        Args:
            h1, h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Scalar loss, or zero when no valid scale pair exists.
        """
        if self.kernel_size == 0:
            return h1.new_zeros(())

        B, T, D = h1.shape
        if T > self.max_temporal_len:
            idx = torch.randperm(T, device=h1.device)[: self.max_temporal_len]
            idx = idx.sort().values
            h1 = h1[:, idx, :]
            h2 = h2[:, idx, :]

        scales1 = hierarchical_pooling(h1, self.kernel_size, self.pool_op)
        scales2 = hierarchical_pooling(h2, self.kernel_size, self.pool_op)

        if len(scales1) < 2:
            return h1.new_zeros(())

        max_pairs = 2 if T < 100 else len(scales1) - 1
        n_pairs = min(max_pairs, len(scales1) - 1)

        loss_total = h1.new_zeros(())
        n_valid = 0

        for s in range(n_pairs):
            h_fine1, h_coarse1 = scales1[s], scales1[s + 1]
            h_fine2, h_coarse2 = scales2[s], scales2[s + 1]

            T_fine = h_fine1.shape[1]
            T_coarse = h_coarse1.shape[1]
            if T_coarse < 2:
                continue

            labels = torch.arange(T_fine, device=h1.device) // self.kernel_size
            labels = labels.clamp(max=T_coarse - 1)

            if isinstance(loss_fn, InfoNCELoss):
                sim_12 = _pairwise_sim(h_fine1, h_coarse2, loss_fn.sim_func) / loss_fn.temperature
                sim_21 = _pairwise_sim(h_fine2, h_coarse1, loss_fn.sim_func) / loss_fn.temperature

                lbl = labels.unsqueeze(0).expand(B, T_fine).reshape(B * T_fine)
                log12 = sim_12.reshape(B * T_fine, T_coarse).clamp(-100.0, 100.0)
                log21 = sim_21.reshape(B * T_fine, T_coarse).clamp(-100.0, 100.0)

                loss_s = 0.5 * (F.cross_entropy(log12, lbl) + F.cross_entropy(log21, lbl))
            else:
                pos1 = h_coarse1[:, labels, :]
                pos2 = h_coarse2[:, labels, :]
                neg_idx = (labels + T_coarse // 2) % T_coarse
                neg1 = h_coarse1[:, neg_idx, :]
                neg2 = h_coarse2[:, neg_idx, :]

                anchor1 = h_fine1.reshape(B * T_fine, D)
                positive1 = pos2.reshape(B * T_fine, D)
                negative1 = neg2.reshape(B * T_fine, D)
                anchor2 = h_fine2.reshape(B * T_fine, D)
                positive2 = pos1.reshape(B * T_fine, D)
                negative2 = neg1.reshape(B * T_fine, D)
                loss_s = 0.5 * (
                    loss_fn(anchor1, positive1, negative1)
                    + loss_fn(anchor2, positive2, negative2)
                )

            loss_total = loss_total + loss_s
            n_valid += 1

        if n_valid == 0:
            return h1.new_zeros(())
        return loss_total / n_valid

    # ------------------------------------------------------------------
    # Unified dispatch with hierarchical loop
    # ------------------------------------------------------------------

    def compute_all_losses(
        self,
        h1: Tensor,
        h2: Tensor,
        loss_fn: Union[InfoNCELoss, TripletLoss],
    ) -> Dict[str, Tensor]:
        """Compute all active contrast losses.

        Following AutoCLS §2.2 / TS2Vec, when ``kernel_size > 0`` instance
        and temporal losses are averaged over all hierarchical scales
        produced by recursive pooling. ``kernel_size = 0`` disables the
        loop (single-scale only). Cross-scale contrast is additive and
        independent.

        Args:
            h1, h2: Shape (B, T, D).
            loss_fn: Configured loss function.

        Returns:
            Dict mapping contrast type name → scalar loss. Always contains
            ``'instance'``; optionally ``'temporal'`` / ``'cross_scale'``.
        """
        # ── Build hierarchical scales ────────────────────────────────────
        if self.kernel_size > 0:
            scales1 = hierarchical_pooling(h1, self.kernel_size, self.pool_op)
            scales2 = hierarchical_pooling(h2, self.kernel_size, self.pool_op)
        else:
            scales1, scales2 = [h1], [h2]

        # ── Instance + temporal at every scale, TS2Vec-style averaging ───
        # Following TS2Vec: sum (alpha*instance + (1-alpha)*temporal) per
        # scale, then divide by the total number of scales d.  This ensures
        # temporal loss weight naturally decreases when some scales lack it
        # (T < 2), rather than being independently averaged.
        alpha = 0.5 if self.temporal else 1.0
        n_scales = len(scales1)
        hierarchical_loss = h1.new_zeros(())

        for s1, s2 in zip(scales1, scales2):
            hierarchical_loss = hierarchical_loss + alpha * self.instance_loss(s1, s2, loss_fn)
            if self.temporal and s1.shape[1] >= 2:
                hierarchical_loss = hierarchical_loss + (1 - alpha) * self.temporal_loss(s1, s2, loss_fn)

        hierarchical_loss = hierarchical_loss / n_scales

        losses: Dict[str, Tensor] = {"hierarchical": hierarchical_loss}

        # ── Cross-scale contrast (independent, not part of the loop) ────
        if self.cross_scale:
            losses["cross_scale"] = self.cross_scale_loss(h1, h2, loss_fn)

        return losses
