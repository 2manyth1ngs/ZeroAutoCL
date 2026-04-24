"""Pretrain the T-CLSC comparator with **gap-split valid + symmetric pairs +
valid-loss early stopping + optional noisy→clean two-stage curriculum** —
ported from AutoCTS++ (``random_search.py``).

Pipeline shape
--------------
1. **Symmetric pair injection**: each unordered pair ``{A, B}`` yields BOTH
   ``(A, B, 1)`` *and* ``(B, A, 0)`` — twice as many training examples,
   teaches the comparator symmetric invariance explicitly.
2. **Gap-based train/valid split**: per task, seeds are greedily placed into
   a valid pool only when they are at least ``valid_gap_threshold`` apart
   from every already-accepted valid seed.  This forces valid pairs to have
   clearly separated labels and keeps the valid signal clean under seed
   noise.
3. **Valid-loss early stopping**: training tracks BCE on the valid pool and
   restores the best-by-valid comparator state on exit.
4. **Two-stage curriculum** (P2): when ``clean_seeds`` is provided,
   :func:`pretrain_comparator` first trains on the noisy set (broad but
   noisy labels), then fine-tunes on the clean set (narrow but reliable).
   The fine-tune stage starts from the best-by-noisy-valid weights; by
   default it uses 0.1× the noisy lr and half the noisy epochs.  This
   matches the AutoCTS++ ``noisy_seeds`` → ``clean_seeds`` protocol
   (``reference/AutoCTS_plusplus/exps/generate_seeds.py:94-103``).

The old gap-magnitude curriculum was removed in P0 — it relied on
trust-worthy absolute performance numbers, which our fixed-seed seed
records do not provide.
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


@torch.no_grad()
def _split_seeds_and_pairs(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    valid_gap_threshold: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Split seeds per task into train/valid with gap-dedup, emit symmetric pairs.

    Algorithm (per task, matches AutoCTS++ ``random_search.py:362-371``):
      - Iterate seeds in their natural order; for each seed *s*, put *s* into
        ``valid_seeds`` if no already-accepted valid seed has performance
        within ``valid_gap_threshold`` of *s*; otherwise put *s* into
        ``train_seeds``.
      - Fallback: if the greedy split leaves valid with < 2 seeds (cannot
        form any pair) but the task had ≥ 4 seeds total, force-include the
        min-performance and max-performance seeds into valid.

    Args:
        seeds: All evaluated seed records (multiple tasks allowed).
        task_features: ``task_id → task-feature tensor`` mapping.
        valid_gap_threshold: Minimum performance gap required for a seed to
            qualify for the valid pool.  The unit is whatever ``performance``
            is — for forecasting (``-MSE``) typical values are 0.01 – 0.05.

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

    for task_id, task_seeds in by_task.items():
        if task_id not in task_features:
            logger.warning("Task %s has seeds but no task_feature; skipping.", task_id)
            continue
        feat = task_features[task_id]

        # Greedy gap-based split.
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
    stage_name: str,
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    comparator: TCLSC,
    config: Dict,
    device: torch.device,
) -> TCLSC:
    """Run one training stage over *seeds* and load best-by-valid weights.

    Args:
        stage_name: Short label (e.g. ``"pretrain/noisy"`` or ``"finetune/
            clean"``) used to namespace the log output.
        seeds: All seed records for this stage.
        task_features: Shared ``task_id → feature`` mapping.
        comparator: Comparator to train in place.  Caller is responsible
            for constructing it / setting the device.
        config: Per-stage config (see :func:`pretrain_comparator` for keys).
        device: Torch device.

    Returns:
        The same comparator object with best-by-valid weights restored.
    """
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


def _derive_clean_config(noisy_config: Dict, clean_config: Optional[Dict]) -> Dict:
    """Compute an effective clean-stage config.

    Behaviour:
      - If *clean_config* is ``None``, derive from *noisy_config* with
        ``lr *= 0.1`` (gentler fine-tune) and ``epochs //= 2`` (shorter
        run on the smaller clean pool).
      - If *clean_config* is given, its keys override those inherited
        from *noisy_config*.  This lets callers specify only the keys
        they care about (e.g. ``{"lr": 1e-5}``).
    """
    base = dict(noisy_config)
    if clean_config is None:
        base["lr"]     = float(noisy_config.get("lr", 1e-4)) * 0.1
        base["epochs"] = max(1, int(noisy_config.get("epochs", 100)) // 2)
        return base
    base.update(clean_config)
    return base


def pretrain_comparator(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    config: Dict,
    comparator: Optional[TCLSC] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    clean_seeds: Optional[List[SeedRecord]] = None,
    clean_config: Optional[Dict] = None,
) -> TCLSC:
    """Pretrain T-CLSC, optionally with a two-stage noisy→clean curriculum.

    Args:
        seeds: Seed records for the (noisy) pretraining stage.  If
            *clean_seeds* is ``None`` these are the only seeds used and the
            function behaves as single-stage training.
        task_features: Mapping ``task_id → task-feature tensor``.  Must
            cover every ``task_id`` present in *seeds* and *clean_seeds*.
        config: Pretraining-stage config.  Recognised keys:

            - ``epochs`` (int, 100)
            - ``lr`` (float, 1e-4)
            - ``batch_size`` (int, 256)
            - ``hidden_dim`` (int, 128): TCLSC hidden size (used only when
              a fresh comparator is constructed)
            - ``valid_gap_threshold`` (float, 0.02)
            - ``patience`` (int, 10)
            - ``eval_every`` (int, 1)

        comparator: Optional pre-constructed comparator.  When ``None`` a
            fresh :class:`TCLSC` with ``hidden_dim`` from *config* is built.
        save_path: If given, persist the final comparator weights here
            (after the clean-stage fine-tune, when applicable).
        device: Torch device; auto-detect when ``None``.
        clean_seeds: Optional second-stage seeds (AutoCTS++ ``clean_seeds``
            analog).  When provided, training runs two stages:

              1. Pretrain on *seeds* with *config* (broad but noisy labels).
              2. Fine-tune on *clean_seeds* with *clean_config* (narrow but
                 reliable labels), starting from the best pretrain weights.

        clean_config: Overrides for the fine-tune stage.  Any missing keys
            inherit from *config*; the defaults when *clean_config* is
            ``None`` are ``lr := 0.1 * config.lr`` and ``epochs :=
            config.epochs // 2``.

    Returns:
        Trained :class:`TCLSC` (with final-stage best-by-valid weights
        loaded when a non-empty valid set was used).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim = int(config.get("hidden_dim", 128))

    if "curriculum_levels" in config:
        logger.warning(
            "config['curriculum_levels']=%s is deprecated — the gap-magnitude "
            "curriculum was removed; the intended curriculum now lives in "
            "generate_seeds(mode='noisy')→generate_seeds(mode='clean') two-stage "
            "training.  Value ignored.",
            config["curriculum_levels"],
        )

    # ── Comparator ────────────────────────────────────────────────────
    if comparator is None:
        comparator = TCLSC(hidden_dim=hidden_dim)
    comparator = comparator.to(device)

    two_stage = clean_seeds is not None and len(clean_seeds) > 0
    stage1_name = "pretrain/noisy" if two_stage else "pretrain"

    # ── Stage 1 ──────────────────────────────────────────────────────
    comparator = _train_one_stage(
        stage1_name, seeds, task_features, comparator, config, device,
    )

    # ── Stage 2 (optional, AutoCTS++ clean fine-tune) ────────────────
    if two_stage:
        effective_clean_config = _derive_clean_config(config, clean_config)
        logger.info(
            "[finetune/clean] starting from best-by-noisy-valid weights; "
            "effective config: lr=%g  epochs=%d  patience=%d",
            effective_clean_config.get("lr"),
            effective_clean_config.get("epochs"),
            effective_clean_config.get("patience", config.get("patience", 10)),
        )
        comparator = _train_one_stage(
            "finetune/clean",
            clean_seeds, task_features, comparator,
            effective_clean_config, device,
        )

    # ── Save final ────────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(comparator.state_dict(), save_path)
        logger.info("Saved comparator weights to %s", save_path)

    return comparator
