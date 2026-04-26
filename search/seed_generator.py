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
from typing import Dict, List, Optional, Tuple

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


def _resolve_mode_budget(
    ds_budget: Dict,
    mode: str,
    default_epochs: int,
) -> Tuple[int, int]:
    """Pick the (pretrain_iters, pretrain_epochs) pair for a (dataset, mode).

    AutoCTS++'s ``noisy_seeds`` stage runs a 20× shorter training schedule
    than ``clean_seeds`` (5 vs 100 epochs, ``reference/AutoCTS_plusplus/
    exps/generate_seeds.py:94-103``).  ZeroAutoCL mirrors this via two
    optional keys in the per-dataset budget dict:

      - ``noisy_pretrain_iters``:  iter budget when ``mode == 'noisy'``
      - ``noisy_pretrain_epochs``: epoch budget when ``mode == 'noisy'``

    Resolution rule (in order):

      1. If ``mode != 'noisy'`` → use base ``pretrain_iters`` /
         ``pretrain_epochs``.
      2. If ``mode == 'noisy'`` AND at least one ``noisy_*`` key is set →
         compose the noisy budget.  ``noisy_pretrain_iters`` (when set)
         drives iter mode; ``noisy_pretrain_epochs`` (when set) is the
         epoch fallback.  Either alone is sufficient — the missing axis
         falls back to its base counterpart so downstream
         ``contrastive_pretrain`` still has a meaningful epoch count for
         logging even in iter mode.
      3. ``mode == 'noisy'`` with no ``noisy_*`` keys → fall back to base
         budget (preserves backward compatibility with old YAML files).

    Args:
        ds_budget: One source's budget dict (may contain ``pretrain_iters``,
            ``pretrain_epochs``, ``noisy_pretrain_iters``,
            ``noisy_pretrain_epochs``).
        mode: ``'noisy'`` or ``'clean'``.
        default_epochs: Fallback epoch count (the seed_generator-level
            default) when neither base nor mode-specific epochs are given.

    Returns:
        ``(pretrain_iters, pretrain_epochs)`` — pass directly to
        :func:`_evaluate_candidate`.  ``pretrain_iters > 0`` engages
        iter-budget mode inside :func:`contrastive_pretrain`; otherwise
        the call runs ``pretrain_epochs`` epochs.
    """
    base_iters  = int(ds_budget.get("pretrain_iters", 0))
    base_epochs = int(ds_budget.get("pretrain_epochs", default_epochs))

    if mode != "noisy":
        return base_iters, base_epochs

    has_noisy = (
        "noisy_pretrain_iters"  in ds_budget
        or "noisy_pretrain_epochs" in ds_budget
    )
    if not has_noisy:
        return base_iters, base_epochs

    noisy_iters  = int(ds_budget.get("noisy_pretrain_iters",  0))
    noisy_epochs = int(ds_budget.get("noisy_pretrain_epochs", base_epochs))
    return noisy_iters, noisy_epochs


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
    mode: str = "clean",
    horizon_groups: Optional[List[Optional[List[int]]]] = None,
) -> List[float]:
    """Train one candidate and return its validation performance(s).

    Delegates to :func:`train.pretrain.contrastive_pretrain` so that the
    iter-budget mechanism (Bug #003a) and the task-aware val-best gating
    are applied uniformly with the rest of the pipeline.

    Modes (``mode`` parameter) — AutoCTS++-style seed generation:

    - ``"clean"``  (default): the original protocol.  Run full training,
      evaluate the final encoder via :func:`_quick_eval`.  Low noise but
      expensive; used for the comparator's fine-tuning stage.

    - ``"noisy"``: short training (budget set by caller) with a val eval
      after every epoch.  Performance = ``max(val_score_per_epoch)``
      (equivalent to AutoCTS++'s ``min(MAE)`` over the 5 noisy epochs at
      ``reference/AutoCTS_plusplus/exps/random_search.py:285``).  Cheap
      best-of-N estimator that naturally attenuates per-epoch seed noise;
      used for the comparator's pretraining stage.

    Args:
        horizon_groups: When the task is forecasting and clean mode is
            active, evaluate the trained encoder once per horizon list and
            return a perf score per group.  ``None`` (or ``[None]``) keeps
            the legacy single-eval behaviour.  In noisy mode the parameter
            is ignored and a single value is returned (per-epoch eval inside
            ``contrastive_pretrain`` already costs a horizon set; supporting
            multi-group there would re-run the whole CL loop per group).

    Returns:
        List of perf values, one per horizon group (always length 1 for
        non-forecasting tasks or noisy mode).  Returns ``[-1e9, ...]`` of
        the same length on training failure.
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

    # In noisy mode we need per-epoch val scores so we can return the best
    # one as the "best-of-N" performance estimate.  val_best=False so we
    # don't restore checkpoints (the encoder is discarded after this call).
    history: Optional[List[Dict]] = None
    if mode == "noisy":
        cfg["eval_every"] = 1
        cfg["val_best"]   = False
        history = []

    try:
        contrastive_pretrain(
            encoder=encoder,
            cl_pipeline=pipeline,
            train_data=train_dataset,
            config=cfg,
            device=device,
            task_type=task_type,
            val_data=(val_dataset if mode == "noisy" else None),
            # P1-A: noisy mode uses full horizons (None → default
            # [24, 48, 168, 336, 720]) so the best-of-N val_score reflects
            # end-to-end forecasting quality, not short-horizon bias.
            horizons=None,
            history=history,
        )
    except Exception as exc:                           # pragma: no cover
        logger.warning("contrastive_pretrain failed for candidate: %s", exc)
        return [-1e9] * n_groups

    # ── Noisy mode: best-of-N across training epochs ──────────────────
    if mode == "noisy" and history is not None:
        scores = [h["val_score"] for h in history if h.get("val_score") is not None]
        if scores:
            # val_score is already "higher = better" (for forecasting it is
            # -MSE averaged over the requested horizons).  Max = best epoch.
            best = float(max(scores))
            # Noisy mode produces a single best-of-N estimate — broadcast it
            # to every requested horizon group so callers can stay uniform.
            return [best] * n_groups
        logger.warning(
            "mode=noisy but no per-epoch val_scores recorded; "
            "falling back to _quick_eval on the final encoder.",
        )

    # ── Clean mode (or noisy fallback): eval the final encoder ────────
    encoder.eval()
    perfs: List[float] = []
    for hg in horizon_groups:
        perf = _quick_eval(
            encoder, train_dataset, val_dataset, task_type, device,
            horizons=hg,
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
            # labels use exactly the same pipeline as run_ggs_test / Phase 4
            # evaluation.  val_dataset is passed as ``test_data`` — Ridge's α
            # is then picked from a train tail (default behaviour inside
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
    pretrain_epochs: int = 40,
    pretrain_lr: float = 1e-3,
    batch_size: int = 64,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    dataset_budgets: Optional[Dict[str, Dict[str, int]]] = None,
    fixed_encoders: Optional[List[Dict[str, int]]] = None,
    crop_len: Optional[int] = None,
    mode: str = "clean",
    randomise_init: bool = False,
    n_time_windows: int = 1,
    horizon_groups: Optional[List[Optional[List[int]]]] = None,
    min_window_len: int = 1000,
    n_variable_subsets: int = 1,
    var_size_rates: Optional[List[float]] = None,
    min_var_count: int = 4,
) -> List[SeedRecord]:
    """Generate seed data across source datasets.

    Args:
        source_datasets: Names of datasets to use (e.g. ``['HAR', 'Yahoo', 'ETTh1']``).
        data_dir: Root data directory.
        n_per_dataset: Number of candidates to evaluate per dataset.
        pretrain_epochs: CL pretraining epochs per candidate.
        pretrain_lr: Pretraining learning rate.
        batch_size: Training batch size.
        save_dir: If given, serialise seed records to
            ``{save_dir}/seeds.json``.
        device: Torch device.  ``None`` → auto-detect.
        seed: Global random seed (also used as the base for per-candidate
            deterministic seeds when ``randomise_init=False``).
        fixed_encoders: Optional list of encoder configs to restrict the
            encoder sub-space to (Plan B Stage B). When given, candidates are
            sampled via :func:`batch_sample_strategies` rather than the full
            joint sampler — every seed record's ``encoder_config`` field will
            be drawn from this list.
        crop_len: Optional sliding-window crop length override for
            forecasting training splits.  When given, passed as
            ``window_len_override`` to :func:`load_dataset` so that
            seed generation uses a shorter crop than Phase 4.
        mode: ``"clean"`` (default) or ``"noisy"``.  Passed to
            :func:`_evaluate_candidate`:

            - ``"clean"`` runs full training and evaluates the final encoder.
              Low-variance but expensive — intended for comparator
              fine-tuning stage.
            - ``"noisy"`` runs short training with per-epoch val eval and
              records ``max(val_score)`` over epochs.  Cheap best-of-N
              estimator; intended for comparator pretraining stage.

            This matches AutoCTS++'s two-stage ``noisy_seeds`` /
            ``clean_seeds`` recipe (``reference/AutoCTS_plusplus/exps/
            generate_seeds.py:94-103``).  When ``mode='noisy'`` and the
            per-dataset budget dict contains ``noisy_pretrain_iters`` or
            ``noisy_pretrain_epochs``, those replace the base budget for
            this run (see :func:`_resolve_mode_budget`).  Falls back to
            the base budget when mode-specific keys are absent.
        randomise_init: When True, every candidate gets a fresh seed from
            system entropy (``random.SystemRandom``) rather than the
            deterministic ``seed + i`` derivation.  Use this for a
            complementary "random-init" half of noisy seed generation
            (AutoCTS++ ``use_seed=False`` branch), so that the comparator
            sees pairs from different initialisations and learns to rank
            under seed noise.
        n_time_windows: Number of non-overlapping temporal windows to carve
            out of each forecasting source dataset (AutoCTS++-style
            subset enrichment).  ``1`` (default) preserves the original
            single-task-per-dataset behaviour.  ``>1`` multiplies the seed
            count linearly because each window re-runs CL pretraining.
        horizon_groups: List of horizon sets used at downstream eval time
            (one seed record per group).  ``None`` or ``[None]`` keeps the
            canonical ``[24,48,168,336,720]`` eval.  Horizon groups share
            the CL pretrain (one fit, ``len(horizon_groups)`` evals) so
            this axis is essentially free in compute.  Ignored for noisy
            mode (a single best-of-N value is broadcast to all groups).
        min_window_len: Minimum per-window length required to enable
            time-window slicing — falls back to a single window if the
            source dataset is too short.
        n_variable_subsets: AutoCTS++-style variable subsampling — number of
            random subsets of raw-variable channels per (window).  ``1``
            (default) disables; ``≥2`` engages stratified bucket sampling.
            Datasets with fewer than ``min_var_count`` raw variables (e.g.
            ETT in univariate mode) silently bypass this axis.  Composes
            multiplicatively with ``n_time_windows`` — a source produces
            up to ``n_time_windows × n_variable_subsets`` sub-tasks, each
            requiring its own CL pre-training run.
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
    if mode not in ("clean", "noisy"):
        raise ValueError(f"mode must be 'clean' or 'noisy', got {mode!r}")

    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve task-variant axes once so logging and ID building stay consistent.
    if not horizon_groups:
        horizon_groups = [None]
    has_windows = n_time_windows > 1
    has_var_subsets = n_variable_subsets > 1
    has_hg = not (len(horizon_groups) == 1 and horizon_groups[0] is None)

    # AutoCTS++-style per-candidate seeding:
    #   - randomise_init=False → deterministic ``seed + i`` (default);
    #     runs are reproducible across invocations.
    #   - randomise_init=True  → fresh system entropy per candidate;
    #     teaches the comparator that the CL pipeline is noisy.
    import random as _random_mod
    _sys_random = _random_mod.SystemRandom()

    logger.info(
        "[generate_seeds] mode=%s  randomise_init=%s  n_per_dataset=%d  "
        "n_time_windows=%d  n_variable_subsets=%d  n_horizon_groups=%d",
        mode, randomise_init, n_per_dataset,
        n_time_windows, n_variable_subsets, len(horizon_groups),
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

    all_seeds: List[SeedRecord] = []
    n_datasets = len(source_datasets)
    overall_start = time.time()
    # Global candidate seed offset, incremented per (source × sub-task) so
    # that determinism holds across the full sub-task fan-out without
    # collisions on ``seed + i``.
    global_subtask_idx = 0

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
        # training the source.  ``n_subtasks`` is stored alongside the
        # records so we can advance ``global_subtask_idx`` and keep
        # ``per_cand_seed = seed + global_subtask_idx * n_per_dataset + i``
        # identical to a fresh contiguous run (matters in clean-mode with
        # ``randomise_init=False``).
        ds_ckpt = (
            os.path.join(save_dir, f"_seeds_{ds_name}.json")
            if save_dir is not None else None
        )
        if ds_ckpt and os.path.exists(ds_ckpt) and os.path.getsize(ds_ckpt) > 0:
            try:
                with open(ds_ckpt, encoding="utf-8") as f:
                    payload = json.load(f)
                cached = [SeedRecord.from_dict(d) for d in payload["records"]]
                n_sub_cached = int(payload["n_subtasks"])
                logger.info(
                    "[dataset %d/%d] %s — checkpoint hit (%d records, "
                    "%d sub-tasks); skipping re-training.",
                    ds_idx + 1, n_datasets, ds_name,
                    len(cached), n_sub_cached,
                )
                all_seeds.extend(cached)
                global_subtask_idx += n_sub_cached
                continue
            except Exception as exc:                            # pragma: no cover
                logger.warning(
                    "Checkpoint %s unreadable (%s); re-running %s from scratch.",
                    ds_ckpt, exc, ds_name,
                )

        # Per-dataset budget override (Bug #003a).  Sub-tasks of the same
        # source share the same budget — they are different windows of the
        # same time series, not different datasets.
        # Mode-aware resolution (AutoCTS++ §3.2.4 cheaper-noisy stage):
        # when ``mode='noisy'`` and the YAML provides ``noisy_pretrain_*``
        # keys, those replace the base budget for this run only.  The
        # fall-back to base keys keeps every existing YAML file working
        # without modification.  See :func:`_resolve_mode_budget`.
        ds_budget = (dataset_budgets or {}).get(ds_name, {}) or {}
        # Per-dataset crop_len override.  Wide sources like PEMS07 (883 ch)
        # blow up VRAM at the global crop_len — half the window is enough
        # for relative ranking (the bias is uniform across candidates).
        ds_crop_len = ds_budget.get("crop_len", crop_len)
        if ds_crop_len != crop_len:
            logger.info(
                "  per-dataset crop_len override: %s → %s", crop_len, ds_crop_len,
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

        ds_iters, ds_epochs = _resolve_mode_budget(
            ds_budget, mode, default_epochs=pretrain_epochs,
        )
        if ds_iters > 0:
            logger.info(
                "  budget [mode=%s]: pretrain_iters=%d", mode, ds_iters,
            )
        else:
            logger.info(
                "  budget [mode=%s]: pretrain_epochs=%d", mode, ds_epochs,
            )

        # Sample candidates ONCE per source dataset and reuse them across
        # every sub-task — this mirrors AutoCTS++'s "shared L candidates"
        # trick (random_search.py L26-44), which lets the comparator learn
        # how the same configuration ranks differently across windows /
        # horizons of the same underlying source.
        if fixed_encoders is not None:
            candidates = batch_sample_strategies(n_per_dataset, fixed_encoders)
        else:
            candidates = batch_sample_candidates(n_per_dataset)

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

                if randomise_init:
                    per_cand_seed = _sys_random.randint(0, 2**31 - 1)
                else:
                    per_cand_seed = seed + global_subtask_idx * n_per_dataset + i
                set_seed(per_cand_seed)

                perfs = _evaluate_candidate(
                    enc_cfg, strat_cfg,
                    sub.train, sub.val,
                    task_type,
                    ds_epochs, pretrain_lr, batch_size,
                    device,
                    pretrain_iters=ds_iters,
                    mode=mode,
                    horizon_groups=sub.horizon_groups,
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

            global_subtask_idx += 1
            sub_total = time.time() - sub_start
            logger.info(
                "  [%s] sub-task %d/%d done in %s",
                ds_name, sub_idx + 1, len(sub_tasks), _fmt_hms(sub_total),
            )

        # Promote this source's records into the global list and persist a
        # sidecar checkpoint so a later crash on a different source doesn't
        # cost us this work.  ``n_subtasks`` is recorded so the resume path
        # can advance ``global_subtask_idx`` to match a fresh contiguous run.
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
