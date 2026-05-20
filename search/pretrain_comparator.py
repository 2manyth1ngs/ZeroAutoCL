"""Pretrain the T-CLSC comparator with hold-out source valid split (preferred)
or legacy per-task gap split, symmetric pair injection, and valid-loss early
stopping — ported from AutoCTS++ (``random_search.py``) and adapted for
ZeroAutoCL.

Pipeline shape
--------------
1. **Group by (task_id, stage)**: every seed record carries a ``stage`` tag
   ("clean" or "noisy"); pairs are only constructed within a single
   ``(task_id, stage)`` bucket because noisy and clean labels live on
   slightly different convergence scales (noisy under-converges) and would
   poison cross-stage comparisons.

2. **Train / valid split**:
     - **Hold-out source mode** (``valid_sources`` non-empty, default): all
       seeds whose source matches ``valid_sources`` go into valid; the rest
       go into train.  No gap-split is applied.  Each bucket forms ALL
       quadratic pairs — N seeds → N(N-1) symmetric pairs.  This is the
       AutoCTS++ regime (PEMS08 was held out as the comparator's valid in
       ``random_search.py:502``).
     - **Legacy gap-split mode** (``valid_sources`` empty): per
       ``(task_id, stage)`` z-score perfs and greedily assign seeds to valid
       only when ≥ ``valid_gap_threshold`` z-units from every accepted
       valid seed.  Kept for backwards compatibility with small-scale runs.

3. **Symmetric pair injection**: each unordered pair ``{A, B}`` yields BOTH
   ``(A, B, 1)`` and ``(B, A, 0)`` — twice as many training examples, teaches
   symmetric invariance explicitly (matches AutoCTS++
   ``generate_task_pairs``).

4. **Valid-loss early stopping with monitoring**:
     - Valid BCE tracked against the ``ln 2 ≈ 0.6931`` random-baseline floor
       and explicitly flagged on each log line so a plateau at chance level
       cannot be missed.
     - Per-source valid breakdown (BCE / acc / Spearman ρ on pair concordance)
       to spot "global learns but one source is still random" failures.
     - Per-stage train BCE breakdown (clean vs noisy) to detect noisy-label
       gradient poisoning early.

History
-------
The two-stage curriculum was previously removed because, at the ≤10k-pair
data scale, the comparator BCE never left ``ln 2`` regardless of stage mix.
The real bottleneck was pair count, not noisy label quality — the CLAUDE_ADV.md
§12.5 calibration showed Spearman ρ(noisy, clean) ≥ 0.9 on ETTh2 even with
1-epoch best-of-N labels.  Re-introducing noisy alongside hold-out source
valid lifts the per-task pair count back into the AutoCTS++ regime.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from data.dataset_slicer import parse_task_id
from models.comparator.t_clsc import TCLSC
from utils.logging_utils import get_logger
from .seed_generator import SeedRecord

# BCE of a random binary classifier on a balanced (symmetric) pair pool is
# exactly ln(2) ≈ 0.6931.  We use this as the explicit floor when judging
# whether the comparator has actually learned anything; the per-epoch logger
# prints the gap so a regression to chance-level is impossible to miss.
RANDOM_BASELINE_BCE = math.log(2.0)

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

    All *seeds* must share the same ``stage`` — the caller pre-groups
    records by ``(task_id, stage)`` so this invariant holds by construction
    and so the emitted pair dicts can carry a single unambiguous stage tag
    for per-stage monitoring.  Pair dicts also remember each side's raw
    ``performance`` so the valid-time Spearman ρ diagnostic doesn't need
    a back-reference to the original SeedRecord list.
    """
    n = len(seeds)
    if n == 0:
        return
    stage = seeds[0].stage
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
                    "stage":     stage,
                    "perf_a":    float(x.performance),
                    "perf_b":    float(y.performance),
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
            stage=s.stage,
        )
        for s in seeds
    ]


def _window_id_from_task_id(task_id: str) -> str:
    """Reduce a task_id to its window_id by stripping the ``:hgX`` suffix.

    The noisy-vs-clean Spearman ρ map persisted by
    :func:`search.seed_generator.generate_seeds` is keyed by ``sub.window_id``
    (e.g. ``"Solar:tw0:vs1"``) — i.e. one ρ per (time-window × variable-subset)
    sub-task.  Multiple seed records share that ρ when the sub-task has
    multiple horizon groups (one record per (sub-task, horizon_group), all
    with the same shared-head candidates), so the comparator's per-task_id
    filter must collapse ``task_id`` down to ``window_id`` before lookup.

    Examples::

        "Solar:tw0:vs1:hg0"  → "Solar:tw0:vs1"
        "ETTh2:tw0"          → "ETTh2:tw0"
        "AQShunyi"           → "AQShunyi"
    """
    parts = parse_task_id(task_id)
    wid = parts.base
    if parts.tw_idx is not None:
        wid += f":tw{parts.tw_idx}"
    if parts.vs_idx is not None:
        wid += f":vs{parts.vs_idx}"
    return wid


def _filter_noisy_buckets_by_rho(
    by_group: Dict[Tuple[str, str], List[SeedRecord]],
    noisy_rho_map: Dict[str, float],
    noisy_rho_threshold: float,
) -> Tuple[Dict[Tuple[str, str], List[SeedRecord]], Dict[str, int]]:
    """Drop noisy ``(task_id, "noisy")`` buckets with ρ < threshold.

    Implements the recommendation in
    ``Debug/noisy_rho_audit_2026_05_16.md §5 #1``: the noisy stage of a
    sub-task whose ρ(noisy, clean) on the shared-head overlap fell below
    ``noisy_rho_threshold`` carries near-random or actively-inverted labels,
    so its pairs poison the comparator gradient.  We drop those buckets
    entirely — clean buckets of the same sub-task stay intact, and noisy
    buckets without a recorded ρ (e.g. clean-only sub-tasks, or runs where
    ρ never computed) pass through untouched.

    Args:
        by_group: ``(task_id, stage) → records`` after the initial grouping.
        noisy_rho_map: ``{window_id: ρ}``.  Sourced from ``seeds_meta.json``
            written by :func:`scripts.merge_seed_checkpoints.main`.  Empty
            dict means "no ρ data available, do not filter".
        noisy_rho_threshold: Floor.  Buckets with ρ strictly less than this
            value are dropped.  Negative-ρ buckets (reverse correlation —
            actively poisonous) are always dropped regardless of the floor.

    Returns:
        ``(filtered_by_group, summary)`` where ``summary`` is a dict with
        keys ``kept``, ``dropped``, ``no_rho``, ``dropped_pair_count_est``,
        and ``dropped_buckets`` (a list of ``(task_id, window_id, ρ)`` tuples
        for logging).
    """
    if not noisy_rho_map:
        return by_group, {
            "kept": 0, "dropped": 0, "no_rho": 0,
            "dropped_pair_count_est": 0, "dropped_buckets": [],
        }

    kept_by_group: Dict[Tuple[str, str], List[SeedRecord]] = {}
    dropped_buckets: List[Tuple[str, str, float]] = []
    n_kept = n_dropped = n_no_rho = 0
    dropped_pair_count_est = 0
    for (task_id, stage), records in by_group.items():
        if stage != "noisy":
            kept_by_group[(task_id, stage)] = records
            continue
        wid = _window_id_from_task_id(task_id)
        if wid not in noisy_rho_map:
            kept_by_group[(task_id, stage)] = records
            n_no_rho += 1
            continue
        rho = noisy_rho_map[wid]
        if rho < noisy_rho_threshold:
            n_dropped += 1
            # Symmetric pairs → N(N-1) per bucket, doubled for the (A,B) /
            # (B,A) emission.  Use n*(n-1)*2 as the rough drop estimate.
            n = len(records)
            dropped_pair_count_est += 2 * n * max(0, n - 1)
            dropped_buckets.append((task_id, wid, rho))
        else:
            kept_by_group[(task_id, stage)] = records
            n_kept += 1

    summary = {
        "kept": n_kept,
        "dropped": n_dropped,
        "no_rho": n_no_rho,
        "dropped_pair_count_est": dropped_pair_count_est,
        "dropped_buckets": dropped_buckets,
    }
    return kept_by_group, summary


@torch.no_grad()
def _split_seeds_and_pairs(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    valid_gap_threshold: float,
    valid_sources: Optional[List[str]] = None,
    noisy_rho_map: Optional[Dict[str, float]] = None,
    noisy_rho_threshold: float = 0.0,
    noisy_only: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """Build train / valid pair pools from raw seed records.

    Two split modes (selected by ``valid_sources``):

    **Hold-out source mode** (``valid_sources`` non-empty) — preferred default.
        Seeds whose ``parse_task_id(task_id).base`` is in ``valid_sources`` go
        100% into the valid pool; the rest go 100% into train.  No gap split
        is applied — every seed in a (task_id, stage) bucket participates in
        every pair within that bucket.  This is the AutoCTS++ regime
        (``random_search.py:502-504``: PEMS08 was the held-out source for
        valid/test, never overlapping with the seven sources used for train).
        Why no gap-split?  Once valid is a *different source*, valid BCE
        already measures cross-task generalisation; carving out the easy/hard
        seeds within a task would only shrink the train pair count without
        adding signal.

    **Legacy gap-split mode** (``valid_sources`` empty / ``None``) — kept for
    backwards compatibility with old runs and small-dataset sanity checks.
        Per ``(task_id, stage)``, z-score the task's perfs and greedily place
        each seed into valid only when no already-accepted valid seed lies
        within ``valid_gap_threshold`` (z-score units).  Fallbacks force a
        non-empty valid / train when greedy split degenerates.

    In **both modes**, pairs are formed only WITHIN a ``(task_id, stage)``
    bucket.  Noisy and clean records of the same task are never paired with
    each other because their absolute perf scales differ (noisy under-
    converges) — see ``SeedRecord`` docstring for the full reasoning.

    Args:
        seeds: All evaluated seed records (multiple tasks / stages allowed).
        task_features: ``task_id → task-feature tensor`` mapping.
        valid_gap_threshold: Used only in legacy mode.  Minimum z-score gap
            for a seed to qualify for the valid pool.
        valid_sources: List of source dataset base names whose seeds become
            the valid pool.  ``None`` or empty falls back to legacy gap-split
            so existing scripts keep working unchanged.
        noisy_rho_map: Optional ``{window_id: ρ}`` map from ``seeds_meta.json``
            written by :func:`scripts.merge_seed_checkpoints.main`.  Drives
            the noisy-bucket quality filter — when present, noisy buckets
            whose sub-task ρ < ``noisy_rho_threshold`` are dropped.  Empty
            or ``None`` disables filtering.
        noisy_rho_threshold: Spearman ρ floor for noisy buckets.  Default
            ``0.0`` keeps everything (no-op).  ``0.3`` reproduces the
            "WEAK/PASS only" filter recommended in
            ``Debug/noisy_rho_audit_2026_05_16.md §5 #1``.
        noisy_only: AutoCTS++ alignment switch (rev 2026-05-20).  When
            ``True``, drop all ``(task_id, "clean")`` buckets after the
            initial grouping so the comparator trains exclusively on
            noisy seeds — mirroring AutoCTS++'s AHC pipeline, where the
            ``clean_seeds`` mode is dead code and never invoked.  Default
            ``False`` preserves legacy two-stage (clean + noisy) mixing.

    Returns:
        ``(train_pairs, valid_pairs)`` — each a list of dicts compatible
        with :meth:`TCLSC.forward_batch`.  Both lists contain the symmetric
        ``(A, B, 1)`` / ``(B, A, 0)`` examples per unordered pair.
    """
    # Group by (task_id, stage) so noisy and clean records of the same task
    # never end up in the same pair-construction bucket.  Legacy seed files
    # (no stage field) all get stage="clean" via SeedRecord.from_dict, so this
    # grouping is a no-op for backwards-compat runs.
    by_group: Dict[Tuple[str, str], List[SeedRecord]] = {}
    for s in seeds:
        by_group.setdefault((s.task_id, s.stage), []).append(s)

    # ── AutoCTS++ noisy-only alignment (rev 2026-05-20) ───────────────
    # When ``noisy_only`` is set, drop all clean buckets so pair construction
    # operates exclusively on stage="noisy" records.  This eliminates the
    # (clean, noisy) two-stage mixing that doesn't exist in AutoCTS++'s
    # pipeline (where AHC trains exclusively on noisy seeds).
    if noisy_only:
        before = len(by_group)
        by_group = {k: v for k, v in by_group.items() if k[1] == "noisy"}
        after = len(by_group)
        logger.info(
            "[split] noisy_only=True: dropped %d clean bucket(s), kept %d "
            "noisy bucket(s)",
            before - after, after,
        )

    # ── Noisy-bucket quality filter ───────────────────────────────────
    # Drop noisy ``(task_id, "noisy")`` buckets whose sub-task ρ on the
    # noisy↔clean shared-head overlap fell below ``noisy_rho_threshold``.
    # This is the "level-1 lossless fix" from
    # ``Debug/noisy_rho_audit_2026_05_16.md §5 #1``: negative or near-zero
    # ρ buckets inject reverse / random gradient into the comparator.
    # Clean buckets are NEVER filtered — only noisy.  No-op when the ρ
    # map is empty (e.g. clean-only runs, or runs predating the ρ
    # persistence change).
    if noisy_rho_map and float(noisy_rho_threshold) > 0:
        by_group, filt = _filter_noisy_buckets_by_rho(
            by_group, noisy_rho_map, float(noisy_rho_threshold),
        )
        logger.info(
            "[split] noisy-ρ filter (threshold=%.3f): kept %d noisy bucket(s), "
            "dropped %d, %d had no ρ — est. %d noisy pairs removed",
            float(noisy_rho_threshold),
            filt["kept"], filt["dropped"], filt["no_rho"],
            filt["dropped_pair_count_est"],
        )
        for task_id, wid, rho in filt["dropped_buckets"]:
            logger.info(
                "[split]   DROP noisy %s  (window=%s, ρ=%+.3f < %.3f)",
                task_id, wid, rho, float(noisy_rho_threshold),
            )

    train_pairs: List[Dict] = []
    valid_pairs: List[Dict] = []

    # ── Hold-out source path ──────────────────────────────────────────
    if valid_sources:
        valid_sources_set = {str(name) for name in valid_sources}
        # Per-bucket book-keeping for the single summary log line below.
        n_train_buckets = 0
        n_valid_buckets = 0
        n_train_pairs_total = 0
        n_valid_pairs_total = 0
        skipped_no_feat = 0

        for (task_id, stage), raw_seeds in by_group.items():
            if task_id not in task_features:
                skipped_no_feat += 1
                logger.warning(
                    "Task %s has seeds but no task_feature; skipping.", task_id,
                )
                continue
            feat = task_features[task_id]
            base = parse_task_id(task_id).base
            is_valid_source = base in valid_sources_set

            target_pool = valid_pairs if is_valid_source else train_pairs
            before = len(target_pool)
            _emit_symmetric_pairs(raw_seeds, feat, task_id, target_pool)
            added = len(target_pool) - before

            if is_valid_source:
                n_valid_buckets += 1
                n_valid_pairs_total += added
            else:
                n_train_buckets += 1
                n_train_pairs_total += added

            logger.info(
                "Task %s [stage=%s]: %d seeds → %s pool (%d pairs)",
                task_id, stage, len(raw_seeds),
                "valid" if is_valid_source else "train", added,
            )

        logger.info(
            "[split] hold-out-source mode: valid_sources=%s  | "
            "train buckets=%d (%d pairs)  | valid buckets=%d (%d pairs)  | "
            "skipped (no task_feat)=%d",
            sorted(valid_sources_set),
            n_train_buckets, n_train_pairs_total,
            n_valid_buckets, n_valid_pairs_total,
            skipped_no_feat,
        )
        return train_pairs, valid_pairs

    # ── Legacy gap-split path ─────────────────────────────────────────
    for (task_id, stage), raw_seeds in by_group.items():
        if task_id not in task_features:
            logger.warning("Task %s has seeds but no task_feature; skipping.", task_id)
            continue
        feat = task_features[task_id]
        task_seeds = _zscore_seeds(raw_seeds)

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

        if len(valid_seeds) < 2 and len(task_seeds) >= 4:
            sorted_by_perf = sorted(task_seeds, key=lambda r: r.performance)
            forced_valid = [sorted_by_perf[0], sorted_by_perf[-1]]
            valid_seeds = forced_valid
            train_seeds = [s for s in task_seeds if s not in forced_valid]
            logger.warning(
                "Task %s [stage=%s]: greedy split left < 2 valid seeds; "
                "forced min+max split.",
                task_id, stage,
            )

        if len(train_seeds) < 2 and len(task_seeds) >= 4:
            all_sorted = sorted(task_seeds, key=lambda r: r.performance)
            valid_seeds = all_sorted[::2]
            train_seeds = all_sorted[1::2]
            logger.warning(
                "Task %s [stage=%s]: train pool had %d seeds (threshold=%s, "
                "valid=%d); rebalancing 50/50 across %d total.",
                task_id, stage, len(train_seeds), valid_gap_threshold,
                len(valid_seeds), len(task_seeds),
            )

        _emit_symmetric_pairs(train_seeds, feat, task_id, train_pairs)
        _emit_symmetric_pairs(valid_seeds, feat, task_id, valid_pairs)

        logger.info(
            "Task %s [stage=%s]: %d seeds → %d train (%d pairs) + %d valid (%d pairs)",
            task_id, stage, len(task_seeds),
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
) -> Tuple[float, float, Dict]:
    """Compute valid-pool BCE + accuracy with per-source / per-stage breakdown.

    P0-1: uses ``forward_batch_multitask`` so each pair's own ``task_feat``
    flows through the task encoder.  Pairs are NOT pre-grouped by task —
    we batch directly through the full pair list so per-pair scoring is
    consistent with training.

    Returns a ``(mean_loss, accuracy, breakdown)`` triple.  ``breakdown`` is
    a dict of optional diagnostic sub-dicts whose keys may include:

    - ``per_source``: ``{source_base: {"bce": float, "acc": float, "n": int,
      "spearman": float}}`` — per held-out source, also includes a per-task
      Spearman ρ averaged over that source's sub-tasks (gives a fine-grained
      "is this source learnable at all?" signal that BCE alone hides on
      degenerate buckets).
    - ``per_stage``: ``{"clean"|"noisy": {"bce", "acc", "n"}}`` — present
      when valid contains both stages.

    The per-source breakdown is most useful when ``valid_sources`` covers a
    single hold-out — it surfaces "this single source's BCE is fine but its
    ranking is random" failure modes that aggregate BCE can hide.
    """
    if not valid_pairs:
        return float("nan"), float("nan"), {}

    comparator.eval()

    total_weighted_loss = 0.0
    correct = 0
    n_examples = 0

    # Per-source aggregates (keyed by parse_task_id(...).base).
    per_source: Dict[str, Dict[str, float]] = {}
    # Per-stage aggregates.
    per_stage: Dict[str, Dict[str, float]] = {}
    # Per-task Spearman accumulators: collect (pred, label) lists per task so
    # we can compute a ranking-quality measure once at the end.
    per_task_signals: Dict[str, Dict[str, list]] = {}

    # Mixed-task batched eval.  We don't shuffle valid (eval is deterministic),
    # but we DO walk the full pair list in batch chunks so each batch carries
    # mixed task_feats — the same forward path training uses.
    n_pairs = len(valid_pairs)
    for start in range(0, n_pairs, batch_size):
        batch_idx = list(range(start, min(start + batch_size, n_pairs)))

        enc_a       = [valid_pairs[i]["enc_a"]     for i in batch_idx]
        strat_a     = [valid_pairs[i]["strat_a"]   for i in batch_idx]
        enc_b       = [valid_pairs[i]["enc_b"]     for i in batch_idx]
        strat_b     = [valid_pairs[i]["strat_b"]   for i in batch_idx]
        task_feats  = [valid_pairs[i]["task_feat"] for i in batch_idx]
        task_ids    = [valid_pairs[i]["task_id"]   for i in batch_idx]
        labels      = torch.tensor(
            [valid_pairs[i]["label"] for i in batch_idx],
            dtype=torch.float32, device=device,
        )

        pred = comparator.forward_batch_multitask(
            enc_a, strat_a, enc_b, strat_b, task_feats, task_ids=task_ids,
        )
        loss = F.binary_cross_entropy(pred, labels)
        batch_correct = int(((pred > 0.5).float() == labels).sum().item())
        n_batch = len(batch_idx)

        total_weighted_loss += loss.item() * n_batch
        correct += batch_correct
        n_examples += n_batch

        # Per-source / per-stage accumulators.  Each pair contributes its own
        # source/stage independently — no batch-level grouping assumption.
        for k_local, i in enumerate(batch_idx):
            tid = valid_pairs[i]["task_id"]
            base = parse_task_id(tid).base
            stg = valid_pairs[i].get("stage", "clean")
            pred_val = float(pred[k_local].item())
            lab_val  = float(labels[k_local].item())
            # Numerically-stable per-example BCE: -[y log p + (1-y) log(1-p)].
            # Exact (not batch-mean approximated) so per-source / per-stage
            # numbers can be trusted at fine granularity.
            p_clip = min(max(pred_val, 1e-7), 1.0 - 1e-7)
            ex_bce = -(lab_val * math.log(p_clip)
                       + (1.0 - lab_val) * math.log(1.0 - p_clip))
            is_correct = int((pred_val > 0.5) == (lab_val > 0.5))

            src = per_source.setdefault(base, {"bce_sum": 0.0, "correct": 0, "n": 0})
            src["bce_sum"] += ex_bce
            src["correct"] += is_correct
            src["n"] += 1

            bk = per_stage.setdefault(stg, {"bce_sum": 0.0, "correct": 0, "n": 0})
            bk["bce_sum"] += ex_bce
            bk["correct"] += is_correct
            bk["n"] += 1

            sig = per_task_signals.setdefault(
                tid, {"pred": [], "label": [], "base": base},
            )
            sig["pred"].append(pred_val)
            sig["label"].append(lab_val)

    comparator.train()

    mean_loss = total_weighted_loss / max(n_examples, 1)
    acc = correct / max(n_examples, 1)

    # Per-task Spearman ρ: rank-correlation of pred-prob vs label across the
    # task's valid pairs.  Each unordered seed pair contributes two examples
    # (A>B,1) and (B>A,0); a perfect comparator outputs pred>0.5 exactly when
    # label=1, so ρ on (pred, label) is monotone in pair concordance.  We
    # roll up by source for the breakdown.
    per_source_spearman: Dict[str, list] = {}
    for tid, sig in per_task_signals.items():
        if len(sig["pred"]) < 4:
            continue
        try:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(sig["pred"], sig["label"])
            if rho == rho:  # not NaN
                per_source_spearman.setdefault(sig["base"], []).append(float(rho))
        except ImportError:                                       # pragma: no cover
            pass

    # Materialise the breakdown dict.
    breakdown: Dict = {}
    if per_source:
        breakdown["per_source"] = {
            base: {
                "bce": v["bce_sum"] / max(v["n"], 1),
                "acc": v["correct"] / max(v["n"], 1),
                "n":   int(v["n"]),
                "spearman_mean": (
                    sum(per_source_spearman[base]) / len(per_source_spearman[base])
                    if base in per_source_spearman else float("nan")
                ),
            }
            for base, v in per_source.items()
        }
    if len(per_stage) > 1:
        # Only emit per-stage breakdown when it actually splits — single-stage
        # valid pools (the common case) just duplicate the global numbers.
        breakdown["per_stage"] = {
            stg: {
                "bce": v["bce_sum"] / max(v["n"], 1),
                "n":   int(v["n"]),
            }
            for stg, v in per_stage.items()
        }
    return mean_loss, acc, breakdown


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_stage(
    seeds: List[SeedRecord],
    task_features: Dict[str, Tensor],
    comparator: TCLSC,
    config: Dict,
    device: torch.device,
    save_path: Optional[str] = None,
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
    # Rev 2026-05-16 — L2 regularisation added after run 44320 showed
    # textbook overfitting (train BCE 0.56→0.27 while valid BCE 0.92→1.35
    # over 10 epochs).  Default 1e-3 is the middle of Adam's typical
    # range (1e-4 to 1e-2).  Set to 0.0 to disable.
    weight_decay        = float(config.get("weight_decay", 1e-3))
    batch_size          = int(config.get("batch_size", 256))
    valid_gap_threshold = float(config.get("valid_gap_threshold", 0.02))
    patience            = int(config.get("patience", 10))
    eval_every          = int(config.get("eval_every", 1))
    valid_sources       = config.get("valid_sources") or []
    # Diagnostic flag: filter valid pool down to stage=="clean" only.  Train
    # pool is left untouched.  Use this when the noisy stage's noisy↔clean
    # Spearman ρ is so low on the hold-out source that the noisy valid
    # labels become near-random — in that case noisy-valid BCE plateaus at
    # ln(2) by construction (random labels), drowning the clean-valid
    # signal and triggering the P0-3 sanity gate even when the comparator
    # IS learning on the clean portion.  See result/zac_array_seedgen_43760_4
    # for the ExchangeRate noisy ρ stats that motivated this flag.
    valid_clean_only    = bool(config.get("valid_clean_only", True))
    # The "is comparator stuck at random?" log threshold.  At chance level
    # the BCE on a balanced symmetric pair set is exactly ln(2) ≈ 0.693.
    # We flag epochs whose valid BCE sits inside [ln2 - eps, ln2 + eps].
    plateau_eps         = float(config.get("plateau_eps", 0.02))
    # Noisy-bucket quality filter knobs (see _filter_noisy_buckets_by_rho).
    # ``noisy_rho_map`` is the {window_id: ρ} dict loaded from
    # ``seeds_meta.json`` by ``run_pretrain_comparator.py``.  Empty / 0.0
    # threshold ⇒ no filtering (preserves old behaviour).
    noisy_rho_map       = config.get("noisy_rho_map") or {}
    noisy_rho_threshold = float(config.get("noisy_rho_threshold", 0.0))
    # AutoCTS++ noisy-only alignment switch (rev 2026-05-20).  When True,
    # drop all stage="clean" buckets from pair construction.
    noisy_only          = bool(config.get("noisy_only", False))

    comparator.train()

    # ── Build train / valid pairs ─────────────────────────────────────
    train_pairs, valid_pairs = _split_seeds_and_pairs(
        seeds, task_features,
        valid_gap_threshold=valid_gap_threshold,
        valid_sources=list(valid_sources) if valid_sources else None,
        noisy_rho_map=noisy_rho_map,
        noisy_rho_threshold=noisy_rho_threshold,
        noisy_only=noisy_only,
    )
    if not train_pairs:
        logger.warning("[%s] No train pairs — skipping stage.", stage_name)
        return comparator

    # Optional valid-only clean filter (does not touch train).  Applied here
    # so all downstream consumers — the summary log line below, _valid_loss,
    # early-stop, and the P0-3 sanity gate — see a consistent valid pool.
    if valid_clean_only and valid_pairs:
        n_before = len(valid_pairs)
        valid_pairs = [p for p in valid_pairs if p.get("stage", "clean") == "clean"]
        n_dropped = n_before - len(valid_pairs)
        logger.info(
            "[%s] valid_clean_only=True: dropped %d noisy valid pair(s); "
            "%d clean valid pair(s) remain.",
            stage_name, n_dropped, len(valid_pairs),
        )

    n_train_clean = sum(1 for p in train_pairs if p.get("stage") == "clean")
    n_train_noisy = sum(1 for p in train_pairs if p.get("stage") == "noisy")
    n_valid_clean = sum(1 for p in valid_pairs if p.get("stage") == "clean")
    n_valid_noisy = sum(1 for p in valid_pairs if p.get("stage") == "noisy")
    logger.info(
        "[%s] train=%d (clean=%d, noisy=%d)  valid=%d (clean=%d, noisy=%d)  "
        "| lr=%g  wd=%g  epochs=%d  patience=%d  plateau_eps=%.3f  "
        "random-baseline BCE=%.4f",
        stage_name, len(train_pairs), n_train_clean, n_train_noisy,
        len(valid_pairs), n_valid_clean, n_valid_noisy,
        lr, weight_decay, epochs, patience, plateau_eps, RANDOM_BASELINE_BCE,
    )

    optimizer = torch.optim.Adam(
        comparator.parameters(), lr=lr, weight_decay=weight_decay,
    )

    best_valid_loss: float = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: int = -1
    patience_counter: int = 0

    for epoch in range(epochs):
        # P0-1 fix: global shuffle across ALL train pairs (mixed-task batches).
        # Each batch now contains pairs from many different tasks, so the
        # ``z_task`` term in the comparison head MUST be task-discriminative
        # to fit the per-pair labels — task_encoder finally gets a meaningful
        # gradient signal.  See Debug/comparator_bug_2026_05_12.md §3.
        n_train = len(train_pairs)
        perm = torch.randperm(n_train).tolist()

        epoch_loss = 0.0
        n_batches = 0
        # Per-stage train BCE accumulators for the per-epoch monitor.  Use
        # per-example BCE so the accumulator is exact even when a batch mixes
        # stages (which it now does under mixed-task batching).
        per_stage_loss = {"clean": 0.0, "noisy": 0.0}
        per_stage_n    = {"clean": 0,   "noisy": 0}

        for start in range(0, n_train, batch_size):
            batch_idx = [perm[j] for j in range(start, min(start + batch_size, n_train))]

            enc_a       = [train_pairs[i]["enc_a"]     for i in batch_idx]
            strat_a     = [train_pairs[i]["strat_a"]   for i in batch_idx]
            enc_b       = [train_pairs[i]["enc_b"]     for i in batch_idx]
            strat_b     = [train_pairs[i]["strat_b"]   for i in batch_idx]
            task_feats  = [train_pairs[i]["task_feat"] for i in batch_idx]
            task_ids    = [train_pairs[i]["task_id"]   for i in batch_idx]
            labels      = torch.tensor(
                [train_pairs[i]["label"] for i in batch_idx],
                dtype=torch.float32, device=device,
            )

            pred = comparator.forward_batch_multitask(
                enc_a, strat_a, enc_b, strat_b, task_feats, task_ids=task_ids,
            )
            loss = F.binary_cross_entropy(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Per-stage roll-up: each pair contributes its own stage to the
            # accumulator (a single batch may now mix stages because we don't
            # group by task_id any more).  Approximation: use the batch's mean
            # loss for every example in the batch — this loses per-example
            # signal but matches what the optimizer actually saw.
            mean_loss_item = loss.item()
            for i in batch_idx:
                stg = train_pairs[i].get("stage", "clean")
                if stg not in per_stage_loss:
                    stg = "clean"
                per_stage_loss[stg] += mean_loss_item
                per_stage_n[stg]    += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        clean_train_bce = (
            per_stage_loss["clean"] / per_stage_n["clean"]
            if per_stage_n["clean"] > 0 else float("nan")
        )
        noisy_train_bce = (
            per_stage_loss["noisy"] / per_stage_n["noisy"]
            if per_stage_n["noisy"] > 0 else float("nan")
        )

        should_eval = valid_pairs and (
            (epoch + 1) % eval_every == 0 or epoch == epochs - 1
        )
        if should_eval:
            v_loss, v_acc, breakdown = _valid_loss(
                comparator, valid_pairs, device, batch_size,
            )

            improved = v_loss < best_valid_loss - 1e-6
            if improved:
                best_valid_loss = v_loss
                best_state = copy.deepcopy(comparator.state_dict())
                best_epoch = epoch + 1
                patience_counter = 0
                # Rev 2026-05-19 — persist best weights to disk on every
                # improvement.  Previously the only save was at the end of
                # ``pretrain_comparator`` after the P0-3 sanity gate, so a
                # run that never beat ln(2) lost all weights even though
                # the in-memory best_state was perfectly usable for
                # diagnostics (z_task_cosine_verify, Phase 4 trial runs).
                # Saving here means save_path always contains the current
                # best — newer bests overwrite older ones.
                if save_path is not None:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    torch.save(best_state, save_path)
                    logger.info(
                        "[%s] ep %3d  saved best weights to %s "
                        "(valid loss %.4f)",
                        stage_name, epoch + 1, save_path, v_loss,
                    )
            else:
                patience_counter += 1

            # Random-baseline marker: explicit ↑/↓ relative to ln(2) so a
            # plateau at chance level is impossible to overlook in the log.
            gap = v_loss - RANDOM_BASELINE_BCE
            if abs(gap) <= plateau_eps:
                plateau_tag = "≈ random"
            elif gap < 0:
                plateau_tag = f"↓ random by {-gap:.3f}"
            else:
                plateau_tag = f"↑ random by {gap:.3f}  ⚠"

            # Rev 2026-05-16 — main per-epoch line is now ALWAYS printed.
            # Previously gated by ``verbose`` so silent runs of 9+ epochs
            # were possible when nothing improved (since ep 2-9 are
            # neither improved, nor 10-divisible, nor plateau-near in the
            # typical "stuck above ln(2)" regime).  That made the log
            # indistinguishable from a hung job.  Per-source / per-stage
            # breakdowns are still throttled to improved / 10-multiple
            # epochs to keep the log scannable.
            logger.info(
                "[%s] ep %3d/%d  train=%.4f (clean=%.4f noisy=%s)  "
                "valid=%.4f  acc=%.3f  [%s]  patience=%d/%d%s",
                stage_name, epoch + 1, epochs, avg_train_loss,
                clean_train_bce,
                "n/a" if math.isnan(noisy_train_bce) else f"{noisy_train_bce:.4f}",
                v_loss, v_acc, plateau_tag,
                patience_counter, patience,
                "  *best*" if improved else "",
            )

            # Per-source breakdown — only on improvement or every 10
            # epochs to keep the log readable.  Helps spot "comparator
            # learns globally but is random on one specific source".
            if improved or (epoch + 1) % 10 == 0:
                src_bd = breakdown.get("per_source") if breakdown else None
                if src_bd:
                    for base in sorted(src_bd):
                        m = src_bd[base]
                        rho_str = (
                            f"  ρ={m['spearman_mean']:+.3f}"
                            if not math.isnan(m["spearman_mean"]) else ""
                        )
                        logger.info(
                            "[%s] ep %3d   source=%s  bce=%.4f  acc=%.3f  "
                            "n=%d%s",
                            stage_name, epoch + 1, base,
                            m["bce"], m["acc"], m["n"], rho_str,
                        )
                stg_bd = breakdown.get("per_stage") if breakdown else None
                if stg_bd and (improved or (epoch + 1) % 10 == 0):
                    parts = [
                        f"{s}={v['bce']:.4f} (n={v['n']})"
                        for s, v in sorted(stg_bd.items())
                    ]
                    logger.info(
                        "[%s] ep %3d   valid per-stage: %s",
                        stage_name, epoch + 1, "  ".join(parts),
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
                    "[%s] ep %3d/%d  train=%.4f (clean=%.4f noisy=%s)",
                    stage_name, epoch + 1, epochs, avg_train_loss,
                    clean_train_bce,
                    "n/a" if math.isnan(noisy_train_bce) else f"{noisy_train_bce:.4f}",
                )

    if best_state is not None:
        comparator.load_state_dict(best_state)
        logger.info(
            "[%s] Restored best-by-valid from epoch %d (valid loss %.4f)",
            stage_name, best_epoch, best_valid_loss,
        )

    # Surface best_valid_loss to the caller (pretrain_comparator) so the
    # P0-3 sanity gate can decide whether to save the weights.  Stashed as
    # an attribute to avoid breaking the public return type.
    comparator._best_valid_loss = float(best_valid_loss)  # noqa: SLF001
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
            - ``valid_sources`` (list[str], []): source dataset base names
              held out for valid (preferred).  Non-empty → no gap-split.
            - ``valid_gap_threshold`` (float, 0.5): in z-score units; used
              only when ``valid_sources`` is empty.
            - ``patience`` (int, 10)
            - ``eval_every`` (int, 1)
            - ``plateau_eps`` (float, 0.02): half-width of the random-
              baseline plateau band for log flagging.
            - ``sanity_eps`` (float, 0.02): P0-3 sanity gate margin.
              Saving fails when ``best_valid_loss >= ln(2) - sanity_eps``.
            - ``sanity_skip`` (bool, False): bypass the P0-3 gate (only
              for ablation runs that intentionally want to save a
              chance-level comparator).
            - ``noisy_rho_threshold`` (float, 0.0): noisy-bucket quality
              floor.  Noisy ``(task_id, "noisy")`` buckets whose sub-task
              ρ(noisy, clean) on the shared-head overlap < this value are
              dropped from training pairs.  0.0 disables (no-op);
              ``Debug/noisy_rho_audit_2026_05_16.md §5 #1`` recommends 0.3.
            - ``noisy_rho_map`` (dict[str, float], {}): ``{window_id: ρ}``
              read from ``seeds_meta.json``.  Required for the filter to
              do anything; empty disables.
            - ``noisy_only`` (bool, False): AutoCTS++ alignment switch.
              When True, drop all stage="clean" buckets from pair
              construction so the comparator trains exclusively on noisy
              seeds (mirrors AutoCTS++ AHC pipeline).

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
        save_path=save_path,
    )

    # ── P0-3 sanity gate ─────────────────────────────────────────────
    # A comparator whose best valid BCE never escaped the ln(2) random-
    # baseline plateau is unreliable for downstream zero-shot search.
    # Rev 2026-05-19 — softened from hard RuntimeError to a loud WARNING.
    # Rationale: ``_train_one_stage`` now persists best-by-valid weights to
    # ``save_path`` on every improvement (see the new ``torch.save`` block
    # inside the train loop), so the best comparator is always on disk
    # regardless of whether it crossed the sanity threshold.  Refusing to
    # save here would no longer prevent garbage weights from being used
    # downstream — it would just delete the only diagnostic artefact we
    # have for figuring out *why* a run failed to learn.
    #
    # Override via ``config["sanity_skip"] = True`` to silence the warning
    # entirely (e.g. for ablation runs that intentionally save a
    # chance-level comparator).
    sanity_eps  = float(config.get("sanity_eps", 0.02))
    sanity_skip = bool(config.get("sanity_skip", False))
    best_valid  = getattr(comparator, "_best_valid_loss", float("inf"))
    sanity_failed = (
        not sanity_skip
        and best_valid >= RANDOM_BASELINE_BCE - sanity_eps
    )
    if sanity_failed:
        logger.warning(
            "[P0-3 sanity gate] Comparator never beat random: "
            "best_valid_loss = %.4f >= ln(2) - %.3f = %.4f.  Weights have "
            "still been saved (see the per-epoch '[pretrain] ep N saved "
            "best weights' lines) so you can diagnose via "
            "Debug/p2_7_z_task_cosine_verify.py or run a trial Phase 4, "
            "but treat any downstream ranking as untrustworthy until the "
            "training log shows valid BCE < %.4f.",
            best_valid, sanity_eps, RANDOM_BASELINE_BCE - sanity_eps,
            RANDOM_BASELINE_BCE - sanity_eps,
        )

    if save_path is not None:
        # The per-epoch save inside ``_train_one_stage`` already wrote the
        # best weights to ``save_path``; this end-of-run write is a no-op
        # in the common case (same bytes) but kept for the edge case where
        # the train loop exited before any improvement was recorded (in
        # which case the in-training save never fired and we still want
        # *something* on disk for inspection).
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(comparator.state_dict(), save_path)
        logger.info(
            "Saved comparator weights to %s  (best_valid_loss=%.4f, "
            "gap to ln 2 = %+.4f%s)",
            save_path, best_valid, best_valid - RANDOM_BASELINE_BCE,
            "  [SANITY GATE TRIPPED — see warning above]" if sanity_failed else "",
        )

    return comparator
