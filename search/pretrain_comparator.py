"""Pretrain the T-CLSC comparator with **per-task z-score normalisation +
gap-split valid + symmetric pairs + valid-loss early stopping** — ported
from AutoCTS++ (``random_search.py``) and adapted for ZeroAutoCL.

Pipeline shape
--------------
1. **Per-task z-score** of seed performance:  the gap-split logic uses an
   absolute threshold, but raw performance scales differ by 50–100× across
   sources (e.g. ExchangeRate ≈ −3 vs PEMS04 ≈ −0.27).  Without
   normalisation a fixed ``valid_gap_threshold`` rejects almost every
   PEMS04 seed (everything looks "too close") and accepts almost every
   ExchangeRate seed (everything looks "far apart"), starving the
   comparator's training pool of one task and overflowing the valid pool of
   the other.  Z-score gives the threshold a uniform meaning across tasks.
   Labels are unaffected because z-score is monotone (preserves rank).
2. **Symmetric pair injection**: each unordered pair ``{A, B}`` yields BOTH
   ``(A, B, 1)`` *and* ``(B, A, 0)`` — twice as many training examples,
   teaches the comparator symmetric invariance explicitly.
3. **Gap-based train/valid split**: per task, seeds are greedily placed into
   a valid pool only when they are at least ``valid_gap_threshold`` apart
   (in z-score units) from every already-accepted valid seed.
4. **Valid-loss early stopping**: training tracks BCE on the valid pool and
   restores the best-by-valid comparator state on exit.

The old gap-magnitude curriculum was removed in P0 — it relied on
trust-worthy absolute performance numbers, which our fixed-seed seed
records do not provide.  The previously-supported noisy→clean two-stage
curriculum was also removed because it failed to learn beyond the
random-guess baseline at ZeroAutoCL's data scale (see CLAUDE_DEBUG.md).
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from models.comparator.t_clsc import TCLSC
from utils.logging_utils import get_logger
from .seed_generator import SeedRecord

# Use the project-wide formatted stdout logger.  Without this, bare
# ``logging.getLogger(__name__)`` defaults to WARNING level and silently
# swallows all of the per-stage / per-epoch INFO progress output.
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pairwise dataset construction
# ---------------------------------------------------------------------------

def _emit_symmetric_pairs(
    seeds: List[SeedRecord],
    task_feat: Tensor,
    task_id: str,
    out_pairs: List[Dict],
    out_gaps: Optional[List[float]] = None,
) -> None:
    """Generate both (A, B, 1) and (B, A, 0) for every i<j pair in *seeds*.

    The symmetric emission matches AutoCTS++'s ``generate_task_pairs``
    (``reference/AutoCTS_plusplus/exps/random_search.py:402-412``), which
    doubles the effective training-example count at no evaluation cost.
    """
    n = len(seeds)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = seeds[i], seeds[j]
            gap = abs(a.performance - b.performance)
            for x, y in ((a, b), (b, a)):
                out_pairs.append({
                    "enc_a":     x.encoder_config,
                    "strat_a":   x.strategy,
                    "enc_b":     y.encoder_config,
                    "strat_b":   y.strategy,
                    "task_feat": task_feat,
                    "task_id":   task_id,
                    "label":     1.0 if x.performance > y.performance else 0.0,
                })
                if out_gaps is not None:
                    out_gaps.append(gap)


def _zscore_seeds(seeds: List[SeedRecord]) -> List[SeedRecord]:
    """Return a copy of *seeds* with ``performance`` z-score normalised.

    Mean and std are computed over the input list (intended use is per-task
    so the caller is responsible for grouping first).  Std is clamped to a
    small positive floor to keep behaviour sane when every seed in a task
    has the same performance (degenerate but possible).
    """
    n = len(seeds)
    if n == 0:
        return []
    perfs = [s.performance for s in seeds]
    mean = sum(perfs) / n
    if n > 1:
        var = sum((p - mean) ** 2 for p in perfs) / (n - 1)
    else:
        var = 0.0
    std = max(var ** 0.5, 1e-8)
    return [
        SeedRecord(
            encoder_config=s.encoder_config,
            strategy=s.strategy,
            task_id=s.task_id,
            performance=(s.performance - mean) / std,
        )
        for s in seeds
    ]


@torch.no_grad()
def _split_seeds_and_pairs(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    valid_gap_threshold: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Split seeds per task into train/valid with gap-dedup, emit symmetric pairs.

    Algorithm (per task, matches AutoCTS++ ``random_search.py:362-371``):
      - Z-score the task's performance values so the gap threshold has a
        uniform meaning regardless of the source's absolute scale.
      - Iterate seeds in their natural order; for each seed *s*, put *s* into
        ``valid_seeds`` if no already-accepted valid seed has performance
        within ``valid_gap_threshold`` (z-score units) of *s*; otherwise
        put *s* into ``train_seeds``.
      - Fallback: if the greedy split leaves valid with < 2 seeds (cannot
        form any pair) but the task had ≥ 4 seeds total, force-include the
        min-performance and max-performance seeds into valid.

    Args:
        seeds: All evaluated seed records (multiple tasks allowed).
        task_features: ``task_id → task-feature tensor`` mapping.
        valid_gap_threshold: Minimum performance gap (in z-score units)
            required for a seed to qualify for the valid pool.  Typical
            values are 0.3 – 0.6 (a fraction of one std).

    Returns:
        ``(train_pairs, valid_pairs)`` — each a list of dicts compatible
        with :meth:`TCLSC.forward_batch`.  Both lists already contain the
        symmetric (A, B, 1) / (B, A, 0) examples.
    """
    by_task: Dict[str, List[SeedRecord]] = {}
    for s in seeds:
        by_task.setdefault(s.task_id, []).append(s)

    train_pairs: List[Dict] = []
    valid_pairs: List[Dict] = []

    for task_id, raw_seeds in by_task.items():
        if task_id not in task_features:
            logger.warning("Task %s has seeds but no task_feature; skipping.", task_id)
            continue
        feat = task_features[task_id]
        task_seeds = _zscore_seeds(raw_seeds)

        # Greedy gap-based split (gap is in z-score units).
        valid_seeds: List[SeedRecord] = []
        train_seeds: List[SeedRecord] = []
        for s in task_seeds:
            too_close = any(
                abs(s.performance - v.performance) < valid_gap_threshold
                for v in valid_seeds
            )
            if too_close:
                train_seeds.append(s)
            else:
                valid_seeds.append(s)

        # Fallback 1: valid too small → need ≥ 2 valid seeds to form any pair.
        if len(valid_seeds) < 2 and len(task_seeds) >= 4:
            sorted_by_perf = sorted(task_seeds, key=lambda r: r.performance)
            forced_valid = [sorted_by_perf[0], sorted_by_perf[-1]]
            valid_seeds = forced_valid
            train_seeds = [s for s in task_seeds if s not in forced_valid]
            logger.warning(
                "Task %s: greedy split left < 2 valid seeds; forced min+max split.",
                task_id,
            )

        # Fallback 2: train too small → threshold was too loose OR the task
        # simply has few seeds.  Rebalance 50/50 across ALL seeds by sorted
        # performance so both pools see a spread of labels and contain at
        # least 2 members each (the minimum needed to form any pair).
        if len(train_seeds) < 2 and len(task_seeds) >= 4:
            all_sorted = sorted(task_seeds, key=lambda r: r.performance)
            valid_seeds = all_sorted[::2]
            train_seeds = all_sorted[1::2]
            logger.warning(
                "Task %s: train pool had %d seeds (threshold=%s, valid=%d); "
                "rebalancing 50/50 between train and valid across %d total.",
                task_id, len(train_seeds), valid_gap_threshold,
                len(valid_seeds), len(task_seeds),
            )

        _emit_symmetric_pairs(train_seeds, feat, task_id, train_pairs)
        _emit_symmetric_pairs(valid_seeds, feat, task_id, valid_pairs)

        logger.info(
            "Task %s: %d seeds → %d train (%d pairs) + %d valid (%d pairs)",
            task_id, len(task_seeds),
            len(train_seeds), len(train_seeds) * max(0, len(train_seeds) - 1),
            len(valid_seeds), len(valid_seeds) * max(0, len(valid_seeds) - 1),
        )

    return train_pairs, valid_pairs


# ---------------------------------------------------------------------------
# Valid loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def _valid_loss(
    comparator: TCLSC,
    valid_pairs: List[Dict],
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float]:
    """Compute mean BCE loss + accuracy over the valid pair pool.

    Pairs are grouped by task_id (each group shares a task_feat vector);
    within each group pairs are batched according to ``batch_size``.
    """
    if not valid_pairs:
        return float("nan"), float("nan")

    comparator.eval()

    task_groups: Dict[str, List[int]] = {}
    for k, p in enumerate(valid_pairs):
        task_groups.setdefault(p["task_id"], []).append(k)

    losses: List[float] = []
    correct = 0
    n_examples = 0

    for tid, idxs in task_groups.items():
        task_feat = valid_pairs[idxs[0]]["task_feat"].to(device)
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start : start + batch_size]
            enc_a   = [valid_pairs[i]["enc_a"]   for i in batch_idx]
            strat_a = [valid_pairs[i]["strat_a"] for i in batch_idx]
            enc_b   = [valid_pairs[i]["enc_b"]   for i in batch_idx]
            strat_b = [valid_pairs[i]["strat_b"] for i in batch_idx]
            labels  = torch.tensor(
                [valid_pairs[i]["label"] for i in batch_idx],
                dtype=torch.float32, device=device,
            )
            pred = comparator.forward_batch(enc_a, strat_a, enc_b, strat_b, task_feat)
            loss = F.binary_cross_entropy(pred, labels)
            losses.append(loss.item() * len(batch_idx))
            correct += int(((pred > 0.5).float() == labels).sum().item())
            n_examples += len(batch_idx)

    comparator.train()
    mean_loss = sum(losses) / max(n_examples, 1)
    acc = correct / max(n_examples, 1)
    return mean_loss, acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_stage(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    comparator: TCLSC,
    config: Dict,
    device: torch.device,
) -> TCLSC:
    """Train *comparator* over *seeds* and load best-by-valid weights.

    Args:
        seeds: All seed records.
        task_features: Shared ``task_id → feature`` mapping.
        comparator: Comparator to train in place.  Caller is responsible
            for constructing it / setting the device.
        config: Training config (see :func:`pretrain_comparator` for keys).
        device: Torch device.

    Returns:
        The same comparator object with best-by-valid weights restored.
    """
    stage_name = "pretrain"
    epochs              = int(config.get("epochs", 100))
    lr                  = float(config.get("lr", 1e-4))
    batch_size          = int(config.get("batch_size", 256))
    valid_gap_threshold = float(config.get("valid_gap_threshold", 0.02))
    patience            = int(config.get("patience", 10))
    eval_every          = int(config.get("eval_every", 1))

    comparator.train()

    # ── Build train / valid pairs ─────────────────────────────────────
    train_pairs, valid_pairs = _split_seeds_and_pairs(
        seeds, task_features, valid_gap_threshold=valid_gap_threshold,
    )
    if not train_pairs:
        logger.warning("[%s] No train pairs — skipping stage.", stage_name)
        return comparator

    logger.info(
        "[%s] %d train pairs (symmetric) and %d valid pairs  | lr=%g  epochs=%d  patience=%d",
        stage_name, len(train_pairs), len(valid_pairs), lr, epochs, patience,
    )

    optimizer = torch.optim.Adam(comparator.parameters(), lr=lr)

    best_valid_loss: float = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: int = -1
    patience_counter: int = 0

    for epoch in range(epochs):
        task_groups: Dict[str, List[int]] = {}
        for k, p in enumerate(train_pairs):
            task_groups.setdefault(p["task_id"], []).append(k)

        epoch_loss = 0.0
        n_batches = 0

        for tid, group_indices in task_groups.items():
            perm = torch.randperm(len(group_indices)).tolist()
            task_feat = train_pairs[group_indices[0]]["task_feat"].to(device)

            for start in range(0, len(perm), batch_size):
                batch_idx = [group_indices[perm[j]]
                             for j in range(start, min(start + batch_size, len(perm)))]

                enc_a   = [train_pairs[i]["enc_a"]   for i in batch_idx]
                strat_a = [train_pairs[i]["strat_a"] for i in batch_idx]
                enc_b   = [train_pairs[i]["enc_b"]   for i in batch_idx]
                strat_b = [train_pairs[i]["strat_b"] for i in batch_idx]
                labels  = torch.tensor(
                    [train_pairs[i]["label"] for i in batch_idx],
                    dtype=torch.float32, device=device,
                )

                pred = comparator.forward_batch(enc_a, strat_a, enc_b, strat_b, task_feat)
                loss = F.binary_cross_entropy(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        should_eval = valid_pairs and (
            (epoch + 1) % eval_every == 0 or epoch == epochs - 1
        )
        if should_eval:
            v_loss, v_acc = _valid_loss(comparator, valid_pairs, device, batch_size)

            improved = v_loss < best_valid_loss - 1e-6
            if improved:
                best_valid_loss = v_loss
                best_state = copy.deepcopy(comparator.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or improved or epoch == 0:
                logger.info(
                    "[%s] ep %3d/%d  train=%.4f  valid=%.4f  acc=%.3f  "
                    "patience=%d/%d%s",
                    stage_name, epoch + 1, epochs, avg_train_loss, v_loss, v_acc,
                    patience_counter, patience,
                    "  *best*" if improved else "",
                )

            if patience_counter >= patience:
                logger.info(
                    "[%s] Early stop at epoch %d; best valid loss %.4f at epoch %d.",
                    stage_name, epoch + 1, best_valid_loss, best_epoch,
                )
                break
        else:
            if (epoch + 1) % 10 == 0:
                logger.info(
                    "[%s] ep %3d/%d  train=%.4f",
                    stage_name, epoch + 1, epochs, avg_train_loss,
                )

    if best_state is not None:
        comparator.load_state_dict(best_state)
        logger.info(
            "[%s] Restored best-by-valid from epoch %d (valid loss %.4f)",
            stage_name, best_epoch, best_valid_loss,
        )

    return comparator


def pretrain_comparator(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    config: Dict,
    comparator: Optional[TCLSC] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> TCLSC:
    """Pretrain T-CLSC on a single pool of seed records.

    Args:
        seeds: Seed records to train on.
        task_features: Mapping ``task_id → task-feature tensor``.  Must
            cover every ``task_id`` present in *seeds*.
        config: Training config.  Recognised keys:

            - ``epochs`` (int, 100)
            - ``lr`` (float, 1e-4)
            - ``batch_size`` (int, 256)
            - ``hidden_dim`` (int, 128): TCLSC hidden size (used only when
              a fresh comparator is constructed)
            - ``valid_gap_threshold`` (float, 0.5): in z-score units
            - ``patience`` (int, 10)
            - ``eval_every`` (int, 1)

        comparator: Optional pre-constructed comparator.  When ``None`` a
            fresh :class:`TCLSC` with ``hidden_dim`` from *config* is built.
        save_path: If given, persist the final comparator weights here.
        device: Torch device; auto-detect when ``None``.

    Returns:
        Trained :class:`TCLSC` (with best-by-valid weights loaded when a
        non-empty valid set was used).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = int(config.get("hidden_dim", 128))

    if "curriculum_levels" in config:
        logger.warning(
            "config['curriculum_levels']=%s is deprecated — the gap-magnitude "
            "curriculum was removed.  Value ignored.",
            config["curriculum_levels"],
        )

    if comparator is None:
        comparator = TCLSC(hidden_dim=hidden_dim)
    comparator = comparator.to(device)

    comparator = _train_one_stage(
        seeds, task_features, comparator, config, device,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(comparator.state_dict(), save_path)
        logger.info("Saved comparator weights to %s", save_path)

    return comparator
