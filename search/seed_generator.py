"""Generate seed data by evaluating sampled candidates on source datasets.

For each source dataset, a pool of (encoder_config, cl_strategy) pairs is
sampled, trained via contrastive pretraining, and evaluated on a held-out
validation set.  The resulting :class:`SeedRecord` objects are used to train
the T-CLSC comparator.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TimeSeriesDataset, load_dataset
from data.dataset_slicer import (
    ForecastingSubTask,
    build_task_id,
    make_forecasting_subtasks,
)
from models.encoder.encoder_config import EncoderConfig
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

from .sampler import batch_sample_candidates, batch_sample_strategies

# Bug fix: previously used logging.getLogger(__name__) which had no handler
# attached, causing every progress line in this module to be silently dropped
# under SLURM.  Route through utils.logging_utils.get_logger so that INFO
# messages reach stdout like the rest of the pipeline.
logger = get_logger(__name__)


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS (or M:SS for < 1h)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Seed record
# ---------------------------------------------------------------------------

@dataclass
class SeedRecord:
    """One evaluated (encoder_config, cl_strategy) on a specific task."""

    encoder_config: Dict[str, int]
    strategy: Dict
    task_id: str
    performance: float  # primary metric (higher is better)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SeedRecord":
        return cls(**d)


# ---------------------------------------------------------------------------
# Single candidate evaluation
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    encoder_config: Dict[str, int],
    strategy: Dict,
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    task_type: str,
    pretrain_epochs: int,
    pretrain_lr: float,
    batch_size: int,
    device: torch.device,
    pretrain_iters: int = 0,
    horizon_groups: Optional[List[Optional[List[int]]]] = None,
    eval_horizons: Optional[List[int]] = None,
) -> List[float]:
    """Train one candidate and return its validation performance(s).

    Delegates to :func:`train.pretrain.contrastive_pretrain` so that the
    iter-budget mechanism (Bug #003a) and the task-aware val-best gating
    are applied uniformly with the rest of the pipeline.

    Args:
        horizon_groups: When the task is forecasting, evaluate the trained
            encoder once per horizon list and return a perf score per group.
            ``None`` (or ``[None]``) keeps the legacy single-eval behaviour.

    Returns:
        List of perf values, one per horizon group (always length 1 for
        non-forecasting tasks).  Returns ``[-1e9, ...]`` of the same length
        on training failure.
    """
    from train.pretrain import contrastive_pretrain

    if not horizon_groups:
        horizon_groups = [None]
    n_groups = len(horizon_groups)

    input_dim = train_dataset.n_channels

    encoder = DilatedCNNEncoder.from_config_dict(input_dim, encoder_config).to(device)
    pipeline = CLPipeline(encoder, strategy).to(device)

    cfg: Dict = {
        "pretrain_epochs": pretrain_epochs,
        "pretrain_iters":  pretrain_iters,
        "pretrain_lr":     pretrain_lr,
        "batch_size":      batch_size,
    }

    try:
        contrastive_pretrain(
            encoder=encoder,
            cl_pipeline=pipeline,
            train_data=train_dataset,
            config=cfg,
            device=device,
            task_type=task_type,
            val_data=None,
            horizons=eval_horizons,
        )
    except Exception as exc:                           # pragma: no cover
        logger.warning("contrastive_pretrain failed for candidate: %s", exc)
        return [-1e9] * n_groups

    # When the dataset has a per-source ``eval_horizons`` override (e.g.
    # PEMS07's [24, 48, 168] to dodge the H=720 CPU OOM), substitute it for
    # the default-sentinel ``None`` group.  Explicit horizon groups from
    # ``forecasting_task_variants.horizon_groups`` win — the user's request
    # is always honoured.
    encoder.eval()
    perfs: List[float] = []
    for hg in horizon_groups:
        eval_hg = hg if hg is not None else eval_horizons
        perf = _quick_eval(
            encoder, train_dataset, val_dataset, task_type, device,
            horizons=eval_hg,
        )
        perfs.append(perf)
    return perfs


def _quick_eval(
    encoder: DilatedCNNEncoder,
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    task_type: str,
    device: torch.device,
    horizons: Optional[List[int]] = None,
) -> float:
    """Lightweight downstream evaluation for seed generation.

    Task-wise metrics (all return "higher = better"):

    - ``classification``: SVM accuracy on time-pooled embeddings.
    - ``forecasting``: negative mean MAE across the supplied horizon set
      (``None`` → canonical ``[24, 48, 168, 336, 720]``) under the
      TS2Vec-aligned protocol (causal sliding encode + multi-step Ridge
      with α picked on a val tail).  P1-A: switched from H=24 MSE to
      all-horizons MAE so the comparator supervision reflects both short
      and long-range forecasting quality and has less heavy-tail noise.
      Horizons that do not fit the series length are skipped automatically
      by :func:`eval_forecasting`.
    - ``anomaly_detection``: mean embedding-std as a proxy (higher = more
      structured representation).

    Args:
        horizons: Optional explicit horizon list for forecasting eval.
            ``None`` keeps the legacy canonical set.  Used by the horizon-
            variation pathway (one CL pre-train, multiple downstream evals).
    """
    from sklearn.svm import SVC
    import numpy as np

    def encode_pool(ds: TimeSeriesDataset) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                x = ds.data[i : i + 64].to(device)
                h = encoder(x).mean(dim=1).cpu().numpy()
                parts.append(h)
        return np.concatenate(parts, axis=0)

    try:
        if task_type == "classification":
            tr_repr = encode_pool(train_dataset)
            va_repr = encode_pool(val_dataset)
            tr_y = train_dataset.labels.numpy()
            va_y = val_dataset.labels.numpy()
            if len(np.unique(tr_y)) < 2:
                return 0.0
            svm = SVC(kernel="rbf", max_iter=2000)
            svm.fit(tr_repr, tr_y)
            return float(svm.score(va_repr, va_y))

        elif task_type == "forecasting":
            # Delegate to the shared TS2Vec-protocol evaluator so the seed
            # labels use exactly the same pipeline as Phase 4 evaluation.
            # val_dataset is passed as ``test_data`` — Ridge's α is then
            # picked from a train tail (default behaviour inside
            # :func:`eval_forecasting`).
            from train.evaluate import eval_forecasting
            m = eval_forecasting(
                encoder=encoder,
                train_data=train_dataset,
                test_data=val_dataset,
                horizons=horizons,        # None → canonical [24,48,168,336,720]
                device=device,
            )
            if not m:
                return 0.0
            maes = [v["mae"] for v in m.values()]
            return -float(sum(maes) / len(maes))

        elif task_type == "anomaly_detection":
            # Encode, compute neighbour distance → anomaly score.
            va_repr = encode_pool(val_dataset)  # (N, D)
            # Use std of embeddings as a proxy metric (higher variance = better repr).
            return float(np.mean(np.std(va_repr, axis=0)))
    except Exception as exc:
        logger.warning("_quick_eval failed: %s", exc)
        return -1e9

    return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_seeds(
    source_datasets: List[str],
    data_dir: str,
    n_per_dataset: int = 200,
    n_shared: int = 0,
    source_global_idx_offset: int = 0,
    pretrain_epochs: int = 40,
    pretrain_lr: float = 1e-3,
    batch_size: int = 64,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    dataset_budgets: Optional[Dict[str, Dict[str, int]]] = None,
    fixed_encoders: Optional[List[Dict[str, int]]] = None,
    crop_len: Optional[int] = None,
    n_time_windows: int = 1,
    horizon_groups: Optional[List[Optional[List[int]]]] = None,
    min_window_len: int = 1000,
    n_variable_subsets: int = 1,
    var_size_rates: Optional[List[float]] = None,
    min_var_count: int = 4,
) -> List[SeedRecord]:
    """Generate seed data across source datasets.

    Each candidate is fully trained under the per-dataset iter/epoch budget
    and evaluated via :func:`_quick_eval` on the held-out validation split,
    yielding a single low-variance label per (sub-task, candidate, horizon
    group).  This is the ``clean_seeds`` protocol of AutoCTS++; the noisy
    early-stop variant has been removed because the original noisy/clean
    two-stage curriculum failed to learn beyond the random-guess baseline
    on the ZeroAutoCL data scale (see CLAUDE_DEBUG.md).

    Args:
        source_datasets: Names of datasets to use (e.g. ``['HAR', 'Yahoo', 'ETTh1']``).
        data_dir: Root data directory.
        n_per_dataset: Number of candidates to evaluate per (source × sub-task).
        n_shared: Cross-source L-share pool size (AutoCTS++ generate_seeds.py
            L65-103 ``use_seed=True`` mechanism).  When ``n_shared > 0``, the
            first ``n_shared`` candidates per source are drawn from a single
            pool that is **identical across every source**; the remaining
            ``n_per_dataset - n_shared`` are sampled fresh per source.  This
            gives the comparator dense task-conditional supervision: each
            shared config appears in ``len(source_datasets) × n_sub_tasks``
            (task, perf) cells, training the task-feature head to recognise
            how the same config's rank shifts across tasks.  Clamped to
            ``[0, n_per_dataset]``.  ``0`` (default) preserves the old
            independent per-source pool behaviour.
        source_global_idx_offset: Per-source ds-index offset for SLURM
            job-array fan-out.  When seed generation is split across an
            array (one source per task), each task's local ``ds_idx`` is
            always 0; pass the array task ID here so the per-source random
            pool seed (``seed + 10_000 + effective_ds_idx``) and the per-
            candidate seed (``seed + (effective_ds_idx * n_max_subtasks +
            sub_idx) * n_per_dataset + i``) stay distinct across the array
            and identical to the sequential single-job ordering.  Default
            ``0`` keeps sequential semantics.
        pretrain_epochs: CL pretraining epochs per candidate.
        pretrain_lr: Pretraining learning rate.
        batch_size: Training batch size.
        save_dir: If given, serialise seed records to
            ``{save_dir}/seeds.json``.
        device: Torch device.  ``None`` → auto-detect.
        seed: Global random seed; also used as the base for per-candidate
            deterministic seeds (``seed + (effective_ds_idx * n_max_subtasks
            + sub_idx) * n_per_dataset + i``, where ``effective_ds_idx =
            source_global_idx_offset + ds_idx`` and ``n_max_subtasks =
            n_time_windows × n_variable_subsets``).
        fixed_encoders: Optional list of encoder configs to restrict the
            encoder sub-space to (Plan B Stage B). When given, candidates are
            sampled via :func:`batch_sample_strategies` rather than the full
            joint sampler — every seed record's ``encoder_config`` field will
            be drawn from this list.
        crop_len: Optional sliding-window crop length override for
            forecasting training splits.
        n_time_windows: Number of non-overlapping temporal windows to carve
            out of each forecasting source dataset (AutoCTS++-style
            subset enrichment).  ``1`` preserves the original
            single-task-per-dataset behaviour.  ``>1`` multiplies the seed
            count linearly because each window re-runs CL pretraining.
        horizon_groups: List of horizon sets used at downstream eval time
            (one seed record per group).  ``None`` or ``[None]`` keeps the
            canonical ``[24,48,168,336,720]`` eval.  Horizon groups share
            the CL pretrain (one fit, ``len(horizon_groups)`` evals) so
            this axis is essentially free in compute.
        min_window_len: Minimum per-window length required to enable
            time-window slicing — falls back to a single window if the
            source dataset is too short.
        n_variable_subsets: AutoCTS++-style variable subsampling — number of
            random subsets of raw-variable channels per (window).  ``1``
            disables; ``≥2`` engages stratified bucket sampling.
            Datasets with fewer than ``min_var_count`` raw variables (e.g.
            ETT in univariate mode) silently bypass this axis.  Composes
            multiplicatively with ``n_time_windows`` — a source produces
            up to ``n_time_windows × n_variable_subsets`` sub-tasks.
        var_size_rates: Fractional bucket centres for variable-subset sizing
            (defaults to ``[0.25, 0.5, 0.75]`` inside
            :func:`make_forecasting_subtasks`).
        min_var_count: Minimum raw-variable channel count required to
            enable variable subsampling for a given source.

    Returns:
        List of all :class:`SeedRecord` objects.  Length is
        ``len(source_datasets) × n_per_dataset × n_time_windows ×
        n_variable_subsets × len(horizon_groups)`` (modulo sub-tasks
        skipped for being too short or having too few variables).
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve task-variant axes once so logging and ID building stay consistent.
    if not horizon_groups:
        horizon_groups = [None]
    has_windows = n_time_windows > 1
    has_var_subsets = n_variable_subsets > 1
    has_hg = not (len(horizon_groups) == 1 and horizon_groups[0] is None)

    logger.info(
        "[generate_seeds] n_per_dataset=%d  n_time_windows=%d  "
        "n_variable_subsets=%d  n_horizon_groups=%d",
        n_per_dataset, n_time_windows, n_variable_subsets, len(horizon_groups),
    )
    if has_hg:
        logger.info("[generate_seeds] horizon_groups=%s", horizon_groups)
    if has_var_subsets:
        logger.info(
            "[generate_seeds] var_size_rates=%s  min_var_count=%d",
            var_size_rates, min_var_count,
        )

    if fixed_encoders is not None:
        logger.info(
            "[plan-B] fixed_encoders mode: %d encoder(s) -> %s",
            len(fixed_encoders), fixed_encoders,
        )

    # Hoisted from the persist block at the bottom: per-dataset checkpoint
    # files live in ``save_dir`` and need it to exist before the first
    # source completes.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ── Cross-source L-share pool ────────────────────────────────────────
    # Sampled ONCE before the source loop so every source sees an identical
    # head of the candidate list.  Re-seeding off the base ``seed`` makes
    # the shared pool reproducible regardless of whether downstream
    # per-source pools or per-candidate seeds advance the global RNG.
    n_shared = max(0, min(int(n_shared), int(n_per_dataset)))
    n_random = n_per_dataset - n_shared
    if n_shared > 0:
        set_seed(seed)
        if fixed_encoders is not None:
            shared_candidates = batch_sample_strategies(n_shared, fixed_encoders)
        else:
            shared_candidates = batch_sample_candidates(n_shared)
        logger.info(
            "[generate_seeds] L-share enabled: %d shared + %d random per source",
            n_shared, n_random,
        )
    else:
        shared_candidates = []

    # Per-candidate seed (``seed + (effective_ds_idx * n_max_subtasks +
    # sub_idx) * n_per_dataset + i``) uses this upper bound on per-source
    # sub-task count so the formula is independent of source-iteration
    # order.  That makes seeds line up identically between sequential mode
    # and SLURM job-array fan-out (each task processes one source at its
    # own ``source_global_idx_offset``).
    n_max_subtasks = max(1, int(n_time_windows) * int(n_variable_subsets))
    if int(source_global_idx_offset) != 0:
        logger.info(
            "[generate_seeds] source_global_idx_offset=%d  n_max_subtasks=%d",
            int(source_global_idx_offset), n_max_subtasks,
        )

    all_seeds: List[SeedRecord] = []
    n_datasets = len(source_datasets)
    overall_start = time.time()

    for ds_idx, ds_name in enumerate(source_datasets):
        logger.info(
            "[dataset %d/%d] Generating seeds for: %s",
            ds_idx + 1, n_datasets, ds_name,
        )

        # ── Per-dataset checkpoint resume ──────────────────────────────
        # On crash mid-loop (e.g. PEMS07 OOM) we lose ``all_seeds`` because
        # the bottom-of-function ``seeds.json`` write is unreached.  A
        # sidecar ``_seeds_<ds>.json`` is written after each source
        # completes; on the next invocation we replay it instead of re-
        # training the source.  Per-candidate seeds now derive from
        # ``effective_ds_idx`` (independent of iteration order), so cache
        # resume needs no counter bookkeeping.
        ds_ckpt = (
            os.path.join(save_dir, f"_seeds_{ds_name}.json")
            if save_dir is not None else None
        )
        if ds_ckpt and os.path.exists(ds_ckpt) and os.path.getsize(ds_ckpt) > 0:
            try:
                with open(ds_ckpt, encoding="utf-8") as f:
                    payload = json.load(f)
                cached = [SeedRecord.from_dict(d) for d in payload["records"]]
                n_sub_cached = int(payload.get("n_subtasks", 0))
                logger.info(
                    "[dataset %d/%d] %s — checkpoint hit (%d records, "
                    "%d sub-tasks); skipping re-training.",
                    ds_idx + 1, n_datasets, ds_name,
                    len(cached), n_sub_cached,
                )
                all_seeds.extend(cached)
                continue
            except Exception as exc:                            # pragma: no cover
                logger.warning(
                    "Checkpoint %s unreadable (%s); re-running %s from scratch.",
                    ds_ckpt, exc, ds_name,
                )

        # Per-dataset budget override (Bug #003a).  Sub-tasks of the same
        # source share the same budget — they are different windows of the
        # same time series, not different datasets.
        ds_budget = (dataset_budgets or {}).get(ds_name, {}) or {}
        # Per-dataset crop_len override.  Wide sources like PEMS07 (883 ch)
        # blow up VRAM at the global crop_len — half the window is enough
        # for relative ranking (the bias is uniform across candidates).
        ds_crop_len = ds_budget.get("crop_len", crop_len)
        if ds_crop_len != crop_len:
            logger.info(
                "  per-dataset crop_len override: %s → %s", crop_len, ds_crop_len,
            )
        # Per-dataset forecasting eval horizons override.  Wide sources blow
        # up CPU memory in ``eval_forecasting`` — each horizon allocates an
        # (N, H × C_raw) numpy target; at H=720, C_raw=883 that's ~43 GB.
        # Setting ``eval_horizons`` per-dataset lets PEMS07 drop H=336/720
        # and stay inside a 96 GB SLURM mem allocation.  ``None`` keeps the
        # default [24, 48, 168, 336, 720].
        ds_eval_horizons = ds_budget.get("eval_horizons")
        if ds_eval_horizons is not None:
            logger.info(
                "  per-dataset eval_horizons override: %s", ds_eval_horizons,
            )

        # Expand the source into one or more sub-tasks (time-window axis
        # × variable-subset axis).  Horizon-group axis is handled inside
        # ``_evaluate_candidate`` (one CL pre-train, multiple Ridge evals).
        sub_tasks: List[ForecastingSubTask] = make_forecasting_subtasks(
            ds_name,
            data_dir,
            n_time_windows=n_time_windows,
            horizon_groups=horizon_groups,
            crop_len=ds_crop_len,
            min_window_len=min_window_len,
            n_variable_subsets=n_variable_subsets,
            var_size_rates=var_size_rates,
            min_var_count=min_var_count,
            var_subset_seed=seed,
        )
        logger.info(
            "[dataset %d/%d] %s expanded into %d sub-task(s)",
            ds_idx + 1, n_datasets, ds_name, len(sub_tasks),
        )

        ds_iters  = int(ds_budget.get("pretrain_iters", 0))
        ds_epochs = int(ds_budget.get("pretrain_epochs", pretrain_epochs))
        if ds_iters > 0:
            logger.info("  budget: pretrain_iters=%d", ds_iters)
        else:
            logger.info("  budget: pretrain_epochs=%d", ds_epochs)

        # Per-source candidate pool: cross-source L-share head (sampled
        # once before the loop, identical across every source) plus a
        # per-source random tail.  Within a source the pool is reused
        # across all sub-tasks so the comparator sees the same config
        # ranked differently across windows / horizons.  ``effective_ds_idx``
        # combines the local iteration index with ``source_global_idx_offset``
        # so SLURM job-array fan-out (one source per task) gets the same
        # per-source seeds as a sequential single-job run.
        effective_ds_idx = int(source_global_idx_offset) + ds_idx
        if n_random > 0:
            set_seed(seed + 10_000 + effective_ds_idx)
            if fixed_encoders is not None:
                random_candidates = batch_sample_strategies(n_random, fixed_encoders)
            else:
                random_candidates = batch_sample_candidates(n_random)
        else:
            random_candidates = []
        candidates = list(shared_candidates) + random_candidates

        # Records produced by THIS source.  Kept separate from ``all_seeds``
        # so we can write a per-dataset checkpoint atomically once the
        # source finishes (see end-of-loop block).
        ds_seeds: List[SeedRecord] = []

        for sub_idx, sub in enumerate(sub_tasks):
            task_type = sub.train.task_type
            sub_start = time.time()
            logger.info(
                "  [%s] sub-task %d/%d  window_id=%s  horizon_groups=%d",
                ds_name, sub_idx + 1, len(sub_tasks),
                sub.window_id, len(sub.horizon_groups),
            )

            for i, (enc_cfg, strat_cfg) in enumerate(candidates):
                cand_start = time.time()

                per_cand_seed = (
                    seed
                    + (effective_ds_idx * n_max_subtasks + sub_idx) * n_per_dataset
                    + i
                )
                set_seed(per_cand_seed)

                perfs = _evaluate_candidate(
                    enc_cfg, strat_cfg,
                    sub.train, sub.val,
                    task_type,
                    ds_epochs, pretrain_lr, batch_size,
                    device,
                    pretrain_iters=ds_iters,
                    horizon_groups=sub.horizon_groups,
                    eval_horizons=ds_eval_horizons,
                )
                cand_elapsed = time.time() - cand_start

                # One seed record per horizon group.  Encoder pre-train
                # cost is paid once per (sub-task, candidate); horizon
                # variation just re-runs the cheap Ridge eval.  The
                # (tw_idx, vs_idx) come from the sub-task itself — they
                # may be ``None`` even when the parent run enables the
                # axis (e.g. a too-narrow source falls back to vs_idx=None).
                for hg_idx, perf in enumerate(perfs):
                    tid = build_task_id(
                        ds_name,
                        tw_idx=sub.tw_idx,
                        vs_idx=sub.vs_idx,
                        hg_idx=hg_idx if has_hg else None,
                        has_windows=has_windows,
                        has_variable_subsets=has_var_subsets,
                        has_horizon_groups=has_hg,
                    )
                    ds_seeds.append(
                        SeedRecord(
                            encoder_config=enc_cfg,
                            strategy=strat_cfg,
                            task_id=tid,
                            performance=perf,
                        )
                    )

                # Progress / ETA reporting at the sub-task granularity.
                done = i + 1
                avg_per_cand = (time.time() - sub_start) / done
                sub_eta = avg_per_cand * (n_per_dataset - done)
                logger.info(
                    "    [%s] %d/%d  perfs=[%s]  (enc L%d H%d O%d)  "
                    "took %s  avg %s/cand  sub-ETA %s",
                    sub.window_id, done, n_per_dataset,
                    ", ".join(f"{p:.4f}" for p in perfs),
                    enc_cfg["n_layers"], enc_cfg["hidden_dim"], enc_cfg["output_dim"],
                    _fmt_hms(cand_elapsed),
                    _fmt_hms(avg_per_cand),
                    _fmt_hms(sub_eta),
                )

            sub_total = time.time() - sub_start
            logger.info(
                "  [%s] sub-task %d/%d done in %s",
                ds_name, sub_idx + 1, len(sub_tasks), _fmt_hms(sub_total),
            )

        # Promote this source's records into the global list and persist a
        # sidecar checkpoint so a later crash on a different source doesn't
        # cost us this work.  ``n_subtasks`` is logged for diagnostics; the
        # resume path no longer needs it for seed-counter bookkeeping.
        all_seeds.extend(ds_seeds)
        if ds_ckpt is not None:
            payload = {
                "ds_name": ds_name,
                "n_subtasks": len(sub_tasks),
                "records": [s.to_dict() for s in ds_seeds],
            }
            with open(ds_ckpt, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(
                "  [%s] checkpoint saved: %d records → %s",
                ds_name, len(ds_seeds), ds_ckpt,
            )

        ds_total = time.time() - overall_start
        # Coarse overall ETA: assume remaining datasets cost like the
        # current one's average per-sub-task wall time × remaining sub-tasks.
        remaining_ds = n_datasets - (ds_idx + 1)
        overall_eta = (ds_total / (ds_idx + 1)) * remaining_ds
        logger.info(
            "[dataset %d/%d] %s done  | remaining datasets: %d  | "
            "rough overall-ETA: %s",
            ds_idx + 1, n_datasets, ds_name,
            remaining_ds, _fmt_hms(overall_eta),
        )

    logger.info(
        "All datasets done in %s", _fmt_hms(time.time() - overall_start),
    )

    # ── Persist ────────────────────────────────────────────────────────
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "seeds.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in all_seeds], f, indent=2)
        logger.info("Saved %d seed records to %s", len(all_seeds), path)

    return all_seeds
