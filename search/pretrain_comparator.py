"""Pretrain the T-CLSC comparator with curriculum learning.

Training data is constructed from seed records: for every pair of candidates
evaluated on the same task, a training example records which one performed
better.  Curriculum learning presents easy pairs (large performance gap)
first and gradually introduces harder pairs.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from models.comparator.t_clsc import TCLSC
from .seed_generator import SeedRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pairwise dataset construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_pairs(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
) -> Tuple[List[Dict], List[float]]:
    """Build pairwise training examples from seed records.

    For every pair of seeds on the same task:
      - label = 1.0  if A.performance > B.performance  else 0.0
      - gap   = |A.performance − B.performance|

    Returns:
        pairs: List of dicts with keys ``enc_a, strat_a, enc_b, strat_b,
               task_feat, label``.
        gaps:  Corresponding absolute performance gaps.
    """
    # Group seeds by task
    by_task: Dict[str, List[SeedRecord]] = {}
    for s in seeds:
        by_task.setdefault(s.task_id, []).append(s)

    pairs: List[Dict] = []
    gaps: List[float] = []

    for task_id, task_seeds in by_task.items():
        feat = task_features[task_id]  # (D_task,)
        n = len(task_seeds)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = task_seeds[i], task_seeds[j]
                gap = abs(a.performance - b.performance)
                label = 1.0 if a.performance > b.performance else 0.0

                pairs.append({
                    "enc_a": a.encoder_config,
                    "strat_a": a.strategy,
                    "enc_b": b.encoder_config,
                    "strat_b": b.strategy,
                    "task_feat": feat,
                    "task_id": task_id,
                    "label": label,
                })
                gaps.append(gap)

    return pairs, gaps


def _curriculum_schedule(
    n_pairs: int,
    n_levels: int,
    total_epochs: int,
) -> List[int]:
    """Return the epoch boundary at which each curriculum level activates.

    Level 0 (easiest, largest gap) is always active.  Level k becomes
    active at epoch ``k * (total_epochs // n_levels)``.

    Returns:
        List of length *n_levels* with the activation epoch per level.
    """
    step = max(1, total_epochs // n_levels)
    return [k * step for k in range(n_levels)]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def pretrain_comparator(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    config: Dict,
    comparator: Optional[TCLSC] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> TCLSC:
    """Pretrain a T-CLSC comparator using curriculum learning.

    Args:
        seeds: Seed records produced by :func:`~search.seed_generator.generate_seeds`.
        task_features: Mapping ``task_id → task_feature_tensor`` (each
            of shape ``(D_task,)``).
        config: Training config dict with keys ``epochs``, ``lr``,
            ``batch_size``, ``curriculum_levels``, ``hidden_dim``.
        comparator: An existing :class:`TCLSC` instance to continue
            training.  ``None`` → create a new one.
        save_path: If given, save the trained model weights here.
        device: Torch device.

    Returns:
        Trained :class:`TCLSC` instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs      = int(config.get("epochs", 100))
    lr          = float(config.get("lr", 1e-4))
    batch_size  = int(config.get("batch_size", 256))
    n_levels    = int(config.get("curriculum_levels", 5))
    hidden_dim  = int(config.get("hidden_dim", 128))

    # ── Build comparator ──────────────────────────────────────────────
    if comparator is None:
        comparator = TCLSC(hidden_dim=hidden_dim)
    comparator = comparator.to(device)
    comparator.train()

    # ── Build pairwise training data ──────────────────────────────────
    pairs, gaps = _build_pairs(seeds, task_features)
    n_pairs = len(pairs)
    if n_pairs == 0:
        logger.warning("No pairwise training data — returning untrained comparator.")
        return comparator

    logger.info("Built %d pairwise training examples", n_pairs)

    # Sort by gap descending → level 0 = easiest (largest gap).
    sorted_idx = sorted(range(n_pairs), key=lambda i: -gaps[i])
    level_size = max(1, n_pairs // n_levels)
    # Assign each pair to a curriculum level (0 = easiest).
    pair_levels = [0] * n_pairs
    for rank, idx in enumerate(sorted_idx):
        pair_levels[idx] = min(rank // level_size, n_levels - 1)

    schedule = _curriculum_schedule(n_pairs, n_levels, epochs)
    logger.info("Curriculum schedule (activation epochs): %s", schedule)

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(comparator.parameters(), lr=lr)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(epochs):
        # Determine which levels are active at this epoch.
        max_active_level = 0
        for lvl, start_epoch in enumerate(schedule):
            if epoch >= start_epoch:
                max_active_level = lvl

        # Filter to active pairs.
        active_indices = [
            i for i in range(n_pairs)
            if pair_levels[i] <= max_active_level
        ]

        # Group active pairs by task_id so each batch uses one task feature.
        task_groups: Dict[str, List[int]] = {}
        for i in active_indices:
            tid = pairs[i]["task_id"]
            task_groups.setdefault(tid, []).append(i)

        epoch_loss = 0.0
        n_batches = 0

        for tid, group_indices in task_groups.items():
            # Shuffle within each task group.
            perm = torch.randperm(len(group_indices)).tolist()
            task_feat = pairs[group_indices[0]]["task_feat"].to(device)

            for bs_start in range(0, len(perm), batch_size):
                batch_idx = [group_indices[perm[j]]
                             for j in range(bs_start, min(bs_start + batch_size, len(perm)))]

                enc_a_list   = [pairs[i]["enc_a"] for i in batch_idx]
                strat_a_list = [pairs[i]["strat_a"] for i in batch_idx]
                enc_b_list   = [pairs[i]["enc_b"] for i in batch_idx]
                strat_b_list = [pairs[i]["strat_b"] for i in batch_idx]
                labels       = torch.tensor(
                    [pairs[i]["label"] for i in batch_idx],
                    dtype=torch.float32, device=device,
                )

                pred = comparator.forward_batch(
                    enc_a_list, strat_a_list,
                    enc_b_list, strat_b_list,
                    task_feat,
                )  # (B,)

                loss = F.binary_cross_entropy(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "Comparator epoch %d/%d  loss=%.4f  active_pairs=%d (level≤%d)",
                epoch + 1, epochs, avg_loss, len(active_indices), max_active_level,
            )

    # ── Save ──────────────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(comparator.state_dict(), save_path)
        logger.info("Saved comparator weights to %s", save_path)

    return comparator
