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


def _spearman_rho(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation between two equal-length sequences.

    Returns ``float('nan')`` when there's insufficient data (n < 3) or both
    inputs are constant (rank-correlation is undefined).  Used as a noisy-
    vs-clean label-quality diagnostic on the shared-head overlap of each
    sub-task's two candidate pools.
    """
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return float("nan")
    try:
        from scipy.stats import spearmanr
    except ImportError:                                             # pragma: no cover
        return float("nan")
    rho, _ = spearmanr(xs, ys)
    return float(rho)


# ---------------------------------------------------------------------------
# Seed record
# ---------------------------------------------------------------------------

@dataclass
class SeedRecord:
    """One evaluated (encoder_config, cl_strategy) on a specific task.

    ``stage`` distinguishes the two seed-generation budgets:

    - ``"clean"`` — full per-source CL pretrain budget (e.g. 600 iters).  The
      label is reliable enough to use as ground truth for ranking.
    - ``"noisy"`` — cheap CL pretrain budget (e.g. 100 iters).  Labels carry
      higher variance but the cost-per-record is ~6× lower, so we can afford
      far more candidates and lift the per-task pair count into the regime
      where the comparator's task-conditional head can actually learn.

    The pair-construction logic in ``pretrain_comparator._split_seeds_and_pairs``
    only pairs records that share BOTH ``task_id`` AND ``stage`` — noisy and
    clean labels live on slightly different convergence scales and shouldn't
    be mixed in the same comparison.  Default ``"clean"`` preserves backwards
    compatibility with seed files generated before this field existed.
    """

    encoder_config: Dict[str, int]
    strategy: Dict
    task_id: str
    performance: float  # primary metric (higher is better)
    stage: str = "clean"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SeedRecord":
        # Tolerate seed files written before ``stage`` existed — treat them as
        # clean (the conservative interpretation: assume those records had a
        # full training budget).
        d = dict(d)
        d.setdefault("stage", "clean")
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
    n_noisy_per_dataset: int = 0,
    noisy_pretrain_iters: int = 100,
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
        n_noisy_per_dataset: Per-sub-task count for the cheap "noisy" seed
            pass (AutoCTS++ ``noisy_seeds`` mode).  Default ``0`` disables
            noisy seeds entirely.  When ``>0``, each sub-task runs an
            additional N-candidate pass with the ``noisy_pretrain_iters``
            budget, producing ``SeedRecord(stage="noisy")`` entries.  Noisy
            and clean records share their first ``n_shared`` candidates so
            the per-sub-task Spearman ρ between the two label sources can
            be measured (logged at sub-task end) as a label-quality check.
            Pair construction in the comparator only pairs records sharing
            the same ``(task_id, stage)``; noisy and clean are never mixed
            in the same pair because their absolute perf scales differ
            (noisy under-converges).
        noisy_pretrain_iters: Iter budget used for each noisy candidate.
            Default ``100`` is ~1/6 of the canonical 600-iter clean budget
            and matches the AutoCTS++ noisy/clean wall-clock ratio.  Ignored
            when ``n_noisy_per_dataset == 0``.

    Returns:
        List of all :class:`SeedRecord` objects.  Length is
        ``len(source_datasets) × (n_per_dataset + n_noisy_per_dataset) ×
        n_time_windows × n_variable_subsets × len(horizon_groups)`` (modulo
        sub-tasks skipped for being too short or having too few variables).
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve task-variant axes once so logging and ID building stay consistent.
    # Bug A1 (2026-05-10): ``has_windows`` / ``has_var_subsets`` used to be
    # computed from the *global* ``n_time_windows`` / ``n_variable_subsets``
    # and reused for every source.  When global=1 but a per-source override
    # turned an axis on (e.g. global ``n_time_windows: 1`` + ETTh2 override
    # to 3), the global flag stayed ``False`` and ``build_task_id`` suppressed
    # the ``:twX`` suffix, collapsing all of ETTh2's sub-task IDs to the bare
    # base name and silently merging their seed pools under one task feature.
    # The fix moves the *axis-on* decision inside the per-source loop where
    # the resolved ``ds_n_time_windows`` / ``ds_n_variable_subsets`` are
    # available.  ``has_hg`` is genuinely a run-level decision (horizon
    # groups are not per-source-overridable) so it stays global.
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
    #
    # Per-source overrides (``dataset_budgets[ds].n_time_windows`` /
    # ``n_variable_subsets``) can push a single source above the run-level
    # global, e.g. ETTh2 boosted to ``n_time_windows=9`` while the global
    # default stays at 3.  Take the max across all sources so per-candidate
    # seed slots never collide between consecutive ``effective_ds_idx``.
    def _resolve_subtask_upper(name: str) -> int:
        b = (dataset_budgets or {}).get(name, {}) or {}
        ntw = int(b.get("n_time_windows", n_time_windows))
        nvs = int(b.get("n_variable_subsets", n_variable_subsets))
        return max(1, ntw * nvs)
    n_max_subtasks = max(
        max(1, int(n_time_windows) * int(n_variable_subsets)),
        max((_resolve_subtask_upper(n) for n in source_datasets), default=1),
    )
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

        # Per-dataset slicing axes override (AutoCTS++ §4.1.1: under-represented
        # domains need more sub-tasks to cover their task-feature cluster).
        # Used to bump ETTh2's sub-task count from 3 (univariate → variable
        # axis silently disabled) up to 9, matching wide sources whose 3×3
        # = 9 sub-tasks come from time × variable axes.  Without this the
        # comparator's ETT cluster gets ~3% of the seed pool and degrades to
        # the dominant traffic-source preference.
        ds_n_time_windows = int(ds_budget.get("n_time_windows", n_time_windows))
        ds_n_variable_subsets = int(
            ds_budget.get("n_variable_subsets", n_variable_subsets),
        )
        if ds_n_time_windows != n_time_windows:
            logger.info(
                "  per-dataset n_time_windows override: %d → %d",
                n_time_windows, ds_n_time_windows,
            )
        if ds_n_variable_subsets != n_variable_subsets:
            logger.info(
                "  per-dataset n_variable_subsets override: %d → %d",
                n_variable_subsets, ds_n_variable_subsets,
            )
        # Per-source axis-on flags (Bug A1 fix): use the *resolved* per-source
        # subtask counts so a global-off + per-source-on override (or the
        # reverse) doesn't collapse IDs.  See axis-decision comment at the
        # top of generate_seeds().
        ds_has_windows     = ds_n_time_windows     > 1
        ds_has_var_subsets = ds_n_variable_subsets > 1

        # Expand the source into one or more sub-tasks (time-window axis
        # × variable-subset axis).  Horizon-group axis is handled inside
        # ``_evaluate_candidate`` (one CL pre-train, multiple Ridge evals).
        # Built BEFORE the checkpoint resume so the staleness check below can
        # compare cached ``n_subtasks`` against the actual ``len(sub_tasks)``.
        sub_tasks: List[ForecastingSubTask] = make_forecasting_subtasks(
            ds_name,
            data_dir,
            n_time_windows=ds_n_time_windows,
            horizon_groups=horizon_groups,
            crop_len=ds_crop_len,
            min_window_len=min_window_len,
            n_variable_subsets=ds_n_variable_subsets,
            var_size_rates=var_size_rates,
            min_var_count=min_var_count,
            var_subset_seed=seed,
        )
        logger.info(
            "[dataset %d/%d] %s expanded into %d sub-task(s)",
            ds_idx + 1, n_datasets, ds_name, len(sub_tasks),
        )

        # ── Per-(sub_idx, stage) checkpoint resume (rev 2026-05-12) ────
        # Schema:
        #   { ds_name, n_subtasks, n_per_dataset, n_noisy_per_dataset,
        #     completed_stages: ["sub{i}:clean", "sub{i}:noisy", ...],
        #     records: [...all SeedRecords so far...] }
        #
        # Granularity matters because noisy adds 30-50 min on top of clean's
        # 60-90 min per source.  On spot pre-emption (gpu_spot partition) the
        # old whole-source checkpoint forced re-running everything; this lets
        # us resume after either stage of any sub-task.
        #
        # Staleness: any change in n_subtasks / n_per_dataset /
        # n_noisy_per_dataset between runs forces a full re-generation so
        # the seed pool stays consistent with the current config.
        #
        # Backward compat: legacy checkpoints (no ``completed_stages`` key)
        # are treated as fully done — preserves the old whole-source skip.
        ds_ckpt = (
            os.path.join(save_dir, f"_seeds_{ds_name}.json")
            if save_dir is not None else None
        )

        completed_stages: set = set()
        ds_seeds: List[SeedRecord] = []
        legacy_skip = False

        if ds_ckpt and os.path.exists(ds_ckpt) and os.path.getsize(ds_ckpt) > 0:
            try:
                with open(ds_ckpt, encoding="utf-8") as f:
                    payload = json.load(f)
                cached_records = [
                    SeedRecord.from_dict(d) for d in payload.get("records", [])
                ]
                cached_n_sub   = int(payload.get("n_subtasks", 0))
                cached_n_per   = payload.get("n_per_dataset")
                cached_n_noisy = payload.get("n_noisy_per_dataset")
                cached_stages  = payload.get("completed_stages")  # None on legacy

                # Staleness checks — discard cache on any knob mismatch.
                reason: Optional[str] = None
                if cached_n_sub != 0 and cached_n_sub != len(sub_tasks):
                    reason = (
                        f"n_subtasks changed {cached_n_sub} -> {len(sub_tasks)}"
                    )
                elif (cached_n_per is not None
                      and int(cached_n_per) != int(n_per_dataset)):
                    reason = (
                        f"n_per_dataset changed {cached_n_per} -> {n_per_dataset}"
                    )
                elif (cached_n_noisy is not None
                      and int(cached_n_noisy) != int(n_noisy_per_dataset)):
                    reason = (
                        f"n_noisy_per_dataset changed {cached_n_noisy} -> "
                        f"{n_noisy_per_dataset}"
                    )

                if reason is not None:
                    logger.warning(
                        "[%s] discarding stale cache (%s); delete %s manually "
                        "to keep.",
                        ds_name, reason, ds_ckpt,
                    )
                elif cached_stages is None:
                    # Legacy whole-source checkpoint: treat as fully done.
                    logger.info(
                        "[dataset %d/%d] %s — legacy checkpoint hit (%d "
                        "records, %d sub-tasks); skipping re-training.",
                        ds_idx + 1, n_datasets, ds_name,
                        len(cached_records), cached_n_sub,
                    )
                    all_seeds.extend(cached_records)
                    legacy_skip = True
                else:
                    # Partial resume.  Carry forward already-completed stages.
                    completed_stages = set(cached_stages)
                    ds_seeds = list(cached_records)
                    logger.info(
                        "[%s] partial checkpoint hit: %d stages done "
                        "(%s), %d records cached — resuming.",
                        ds_name, len(completed_stages),
                        ", ".join(sorted(completed_stages)[:8])
                        + ("..." if len(completed_stages) > 8 else ""),
                        len(cached_records),
                    )
            except Exception as exc:                            # pragma: no cover
                logger.warning(
                    "Checkpoint %s unreadable (%s); re-running %s from scratch.",
                    ds_ckpt, exc, ds_name,
                )

        if legacy_skip:
            continue

        # Atomic checkpoint writer — called after each (sub_idx, stage)
        # finishes so a mid-source crash never wastes more than one
        # sub-task × one stage's worth of work.  Uses ``os.replace`` (atomic
        # on POSIX and on Windows when target exists) to avoid leaving a
        # corrupt half-written checkpoint on disk.
        def _persist_ckpt() -> None:
            if ds_ckpt is None:
                return
            payload = {
                "ds_name":             ds_name,
                "n_subtasks":          len(sub_tasks),
                "n_per_dataset":       int(n_per_dataset),
                "n_noisy_per_dataset": int(n_noisy_per_dataset),
                "completed_stages":    sorted(completed_stages),
                "records":             [s.to_dict() for s in ds_seeds],
            }
            tmp = ds_ckpt + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, ds_ckpt)

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

        # ``ds_seeds`` was initialised at cache-load time above — either
        # empty (fresh) or pre-populated from the partial checkpoint.  Do
        # NOT reset it here or cached records get nuked.

        for sub_idx, sub in enumerate(sub_tasks):
            task_type = sub.train.task_type
            sub_start = time.time()
            logger.info(
                "  [%s] sub-task %d/%d  window_id=%s  horizon_groups=%d",
                ds_name, sub_idx + 1, len(sub_tasks),
                sub.window_id, len(sub.horizon_groups),
            )

            key_clean = f"sub{sub_idx}:clean"
            key_noisy = f"sub{sub_idx}:noisy"
            clean_ran_this_invocation = False
            clean_total = 0.0

            # First-horizon clean perfs of the shared-head candidates.  Used
            # *only* for the per-sub-task noisy-vs-clean Spearman diagnostic
            # below (does not affect seed records or downstream training).
            diag_clean_perfs: List[float] = []

            if key_clean in completed_stages:
                logger.info(
                    "  [%s] sub-task %d/%d  clean: cached, skipping (records "
                    "already in ds_seeds)",
                    ds_name, sub_idx + 1, len(sub_tasks),
                )
            else:
                clean_ran_this_invocation = True
                clean_start = time.time()
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

                    # Capture the shared-head perf for the noisy diagnostic.
                    # Only the first horizon group is used so the comparison
                    # is well-defined (every shared candidate produces ≥ 1
                    # perf value, by construction).
                    if i < n_shared and perfs:
                        diag_clean_perfs.append(perfs[0])

                    # One seed record per horizon group.  Encoder pre-train
                    # cost is paid once per (sub-task, candidate); horizon
                    # variation just re-runs the cheap Ridge eval.  The
                    # (tw_idx, vs_idx) come from the sub-task itself — they
                    # may be ``None`` even when the parent run enables the
                    # axis (e.g. too-narrow source falls back to vs_idx=None).
                    for hg_idx, perf in enumerate(perfs):
                        tid = build_task_id(
                            ds_name,
                            tw_idx=sub.tw_idx,
                            vs_idx=sub.vs_idx,
                            hg_idx=hg_idx if has_hg else None,
                            has_windows=ds_has_windows,
                            has_variable_subsets=ds_has_var_subsets,
                            has_horizon_groups=has_hg,
                        )
                        ds_seeds.append(
                            SeedRecord(
                                encoder_config=enc_cfg,
                                strategy=strat_cfg,
                                task_id=tid,
                                performance=perf,
                                stage="clean",
                            )
                        )

                    # Progress / ETA reporting at the sub-task granularity.
                    done = i + 1
                    avg_per_cand = (time.time() - clean_start) / done
                    sub_eta = avg_per_cand * (n_per_dataset - done)
                    logger.info(
                        "    [%s] %d/%d  perfs=[%s]  (enc L%d H%d O%d)  "
                        "took %s  avg %s/cand  sub-ETA %s",
                        sub.window_id, done, n_per_dataset,
                        ", ".join(f"{p:.4f}" for p in perfs),
                        enc_cfg["n_layers"], enc_cfg["hidden_dim"],
                        enc_cfg["output_dim"],
                        _fmt_hms(cand_elapsed),
                        _fmt_hms(avg_per_cand),
                        _fmt_hms(sub_eta),
                    )

                clean_total = time.time() - clean_start

                # Stage done — checkpoint atomically so a crash before the
                # noisy stage finishes doesn't waste the clean records.
                completed_stages.add(key_clean)
                _persist_ckpt()
                logger.info(
                    "  [%s] sub-task %d/%d  clean stage done in %s "
                    "(checkpoint written)",
                    ds_name, sub_idx + 1, len(sub_tasks), _fmt_hms(clean_total),
                )

            # ── Noisy stage (per sub-task) ──────────────────────────────
            # Same shared-head as clean (so diagnostic ρ on first n_shared
            # candidates is well-defined), plus a fresh random tail seeded
            # from a different namespace than the clean random pool.
            if int(n_noisy_per_dataset) > 0:
                if key_noisy in completed_stages:
                    logger.info(
                        "  [%s] sub-task %d/%d  noisy: cached, skipping",
                        ds_name, sub_idx + 1, len(sub_tasks),
                    )
                else:
                    noisy_n = int(n_noisy_per_dataset)
                    n_noisy_random = max(0, noisy_n - n_shared)
                    if n_noisy_random > 0:
                        set_seed(seed + 20_000 + effective_ds_idx)
                        if fixed_encoders is not None:
                            noisy_random_candidates = batch_sample_strategies(
                                n_noisy_random, fixed_encoders,
                            )
                        else:
                            noisy_random_candidates = batch_sample_candidates(
                                n_noisy_random,
                            )
                    else:
                        noisy_random_candidates = []
                    noisy_candidates = (
                        list(shared_candidates) + noisy_random_candidates
                    )
                    noisy_n = len(noisy_candidates)

                    noisy_start = time.time()
                    diag_noisy_perfs: List[float] = []
                    logger.info(
                        "  [%s] sub-task %d/%d  noisy stage: %d candidates × "
                        "%d iters (shared=%d + random=%d)",
                        ds_name, sub_idx + 1, len(sub_tasks), noisy_n,
                        int(noisy_pretrain_iters), n_shared, n_noisy_random,
                    )

                    for i, (enc_cfg, strat_cfg) in enumerate(noisy_candidates):
                        cand_start = time.time()

                        # Distinct seed namespace from clean (offset
                        # +1_000_000) so noisy reproducibility doesn't collide.
                        per_cand_seed = (
                            seed
                            + 1_000_000
                            + (effective_ds_idx * n_max_subtasks + sub_idx)
                              * noisy_n
                            + i
                        )
                        set_seed(per_cand_seed)

                        perfs = _evaluate_candidate(
                            enc_cfg, strat_cfg,
                            sub.train, sub.val,
                            task_type,
                            ds_epochs, pretrain_lr, batch_size,
                            device,
                            pretrain_iters=int(noisy_pretrain_iters),
                            horizon_groups=sub.horizon_groups,
                            eval_horizons=ds_eval_horizons,
                        )
                        cand_elapsed = time.time() - cand_start

                        if i < n_shared and perfs:
                            diag_noisy_perfs.append(perfs[0])

                        for hg_idx, perf in enumerate(perfs):
                            tid = build_task_id(
                                ds_name,
                                tw_idx=sub.tw_idx,
                                vs_idx=sub.vs_idx,
                                hg_idx=hg_idx if has_hg else None,
                                has_windows=ds_has_windows,
                                has_variable_subsets=ds_has_var_subsets,
                                has_horizon_groups=has_hg,
                            )
                            ds_seeds.append(
                                SeedRecord(
                                    encoder_config=enc_cfg,
                                    strategy=strat_cfg,
                                    task_id=tid,
                                    performance=perf,
                                    stage="noisy",
                                )
                            )

                        done = i + 1
                        if done % 10 == 0 or done == noisy_n:
                            avg_per_cand = (time.time() - noisy_start) / done
                            noisy_eta = avg_per_cand * (noisy_n - done)
                            logger.info(
                                "    [%s] noisy %d/%d  perfs=[%s]  took %s  "
                                "avg %s/cand  noisy-ETA %s",
                                sub.window_id, done, noisy_n,
                                ", ".join(f"{p:.4f}" for p in perfs),
                                _fmt_hms(cand_elapsed),
                                _fmt_hms(avg_per_cand),
                                _fmt_hms(noisy_eta),
                            )

                    noisy_total = time.time() - noisy_start

                    # Label-quality diagnostic — only meaningful when both
                    # clean and noisy ran in THIS invocation (so the per-
                    # candidate perfs are both present in memory).  When
                    # clean was loaded from cache we skip with a note.
                    if clean_ran_this_invocation and \
                       len(diag_clean_perfs) >= 3 and \
                       len(diag_clean_perfs) == len(diag_noisy_perfs):
                        rho = _spearman_rho(diag_clean_perfs, diag_noisy_perfs)
                        if rho == rho:  # not NaN
                            if rho >= 0.5:
                                verdict = "[PASS] usable"
                            elif rho >= 0.3:
                                verdict = "[WEAK]"
                            else:
                                verdict = "[FAIL] unreliable"
                        else:
                            verdict = "n/a (constant perf)"
                            rho = float("nan")
                        logger.info(
                            "  [%s] sub-task %d/%d  noisy-vs-clean Spearman "
                            "rho on %d shared candidates = %+.3f  %s",
                            ds_name, sub_idx + 1, len(sub_tasks),
                            len(diag_clean_perfs), rho, verdict,
                        )
                    elif not clean_ran_this_invocation:
                        logger.info(
                            "  [%s] sub-task %d/%d  noisy-vs-clean rho "
                            "diagnostic skipped (clean was loaded from cache)",
                            ds_name, sub_idx + 1, len(sub_tasks),
                        )

                    completed_stages.add(key_noisy)
                    _persist_ckpt()
                    logger.info(
                        "  [%s] sub-task %d/%d  noisy stage done in %s "
                        "(checkpoint written)",
                        ds_name, sub_idx + 1, len(sub_tasks),
                        _fmt_hms(noisy_total),
                    )

            sub_total = time.time() - sub_start
            logger.info(
                "  [%s] sub-task %d/%d done in %s",
                ds_name, sub_idx + 1, len(sub_tasks), _fmt_hms(sub_total),
            )

        # Promote this source's records into the global list.  The per-
        # source checkpoint was already written incrementally by
        # ``_persist_ckpt()`` after each (sub_idx, stage), so there's nothing
        # to write here — just a final log line for visibility.
        all_seeds.extend(ds_seeds)
        if ds_ckpt is not None:
            # Final tidy write so the on-disk checkpoint reflects the
            # post-source state exactly (defensive — every prior _persist_ckpt
            # call already did this, but a single final pass guarantees
            # consistency if any per-stage write was skipped).
            _persist_ckpt()
            logger.info(
                "  [%s] all stages done: %d records  | %d/%d stages "
                "(checkpoint: %s)",
                ds_name, len(ds_seeds), len(completed_stages),
                len(sub_tasks) * (2 if int(n_noisy_per_dataset) > 0 else 1),
                ds_ckpt,
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
