"""Dataset slicing for T-CLSC pre-training diversity.

Three complementary mechanisms (mirroring AutoCTS++ §3.2.4 + the
``pretrain_*_subsets_{12,48}.sh`` script family):

1. **Time-window slicing** — concatenate the base dataset's train/val/test
   splits back into the full series, then carve N non-overlapping temporal
   segments.  Each segment is itself split 80/20 into a fresh train/val pair
   and treated as an *independent task* during seed generation.  This is the
   ZeroAutoCL analog of ``reference/AutoCTS_plusplus/exps/dataset_slice.py``,
   adapted for forecasting datasets that come pre-normalised as ``(1, T, C)``.

2. **Variable subsampling** — for datasets wide enough (`n_raw_vars` ≥
   ``min_var_count``), pick K stratified random subsets of the raw variable
   channels.  AutoCTS++'s ``dataset_slice.py`` does this with rate buckets
   ([0.25, 0.5, 0.75, 1.0]); we keep the bucket idea but drop the adjacency
   matrix reconstruction (ZeroAutoCL's dilated-CNN encoder doesn't consume
   adjacency).  Time-feature covariate columns (e.g., ETT's first 7) are
   *never* subsampled — they're per-timestep aids, not "variables".

3. **Horizon-group variation** — for each (base or sliced) sub-task, the
   downstream Ridge-forecast metric can be evaluated under several horizon
   sets.  Each horizon group becomes a separate seed-record label, sharing
   the underlying CL pre-training (the encoder is reused across groups, so
   horizon variation costs nothing extra in CL training time).

The three axes are orthogonal: a source dataset fans into
``n_time_windows × n_var_subsets`` sub-tasks, each carrying the full
``horizon_groups`` list.  Sub-task IDs use the canonical
``"{base}[:tw{i}][:vs{j}][:hg{k}]"`` format produced by :func:`build_task_id`.

Public surface
--------------
``TaskIdParts``               — parsed task_id components (immutable dataclass).
``ForecastingSubTask``        — dataclass bundling a sub-task's splits + horizons.
``make_forecasting_subtasks`` — fan a single source dataset into N sub-tasks.
``build_task_id``             — canonical formatting for sub-task IDs.
``parse_task_id``             — inverse of :func:`build_task_id`.
``slice_dataset``             — legacy helper, kept for callers that may use it.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from data.dataset import TimeSeriesDataset, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-task dataclass + ID helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskIdParts:
    """Decomposed task_id (output of :func:`parse_task_id`).

    All index fields are ``None`` when the corresponding axis is inactive
    in the parent run, e.g. a backward-compat ``"ETTh2"`` parses to
    ``TaskIdParts("ETTh2", None, None, None)``.
    """

    base: str
    tw_idx: Optional[int] = None
    vs_idx: Optional[int] = None
    hg_idx: Optional[int] = None


@dataclass
class ForecastingSubTask:
    """One (window × var-subset × horizon-groups) bundle for the seed generator.

    The (window, variable-subset) pair fully determines the *training data*
    for one CL pre-training run; ``horizon_groups`` then re-uses that single
    encoder across multiple downstream Ridge evals.

    Attributes:
        base_name: Source dataset name (e.g. ``"ETTh2"``).
        tw_idx: Time-window index inside the base dataset, or ``None`` if
            time-window slicing is disabled.
        vs_idx: Variable-subset index, or ``None`` if variable subsampling
            is disabled (or unsupported for this source — see
            :func:`make_forecasting_subtasks`).
        train: Training split for CL pre-training.
        val: Validation split used by ``_quick_eval`` as the eval target.
        horizon_groups: Each entry is either ``None`` (= canonical
            ``[24, 48, 168, 336, 720]``) or an explicit list of horizons.
            One seed record is emitted per group.
        var_indices: Concrete list of raw-variable column indices kept for
            this sub-task (after subtracting :attr:`n_covariate_cols`).
            ``None`` when the full variable set is used.  Mostly diagnostic.
    """

    base_name: str
    tw_idx: Optional[int]
    vs_idx: Optional[int]
    train: TimeSeriesDataset
    val: TimeSeriesDataset
    horizon_groups: List[Optional[List[int]]] = field(default_factory=lambda: [None])
    var_indices: Optional[List[int]] = None

    @property
    def window_id(self) -> str:
        """Stable identifier excluding the horizon-group suffix.

        Used by the comparator pretraining pipeline to map seed records
        back to their underlying training data (one ``window_id`` per CL
        pre-training run).
        """
        out = self.base_name
        if self.tw_idx is not None:
            out += f":tw{self.tw_idx}"
        if self.vs_idx is not None:
            out += f":vs{self.vs_idx}"
        return out


def build_task_id(
    base: str,
    tw_idx: Optional[int] = None,
    vs_idx: Optional[int] = None,
    hg_idx: Optional[int] = None,
    *,
    has_windows: bool = False,
    has_variable_subsets: bool = False,
    has_horizon_groups: bool = False,
) -> str:
    """Build the canonical task_id for a sub-task seed record.

    Suffix order is fixed: ``:tw{i}:vs{j}:hg{k}`` (any subset may be
    omitted).  The ``has_*`` flags suppress the corresponding suffix even
    when an index is given — this lets a parent run that disables an axis
    keep ID strings clean (e.g. ``"ETTh2:tw0"`` rather than
    ``"ETTh2:tw0:vs0:hg0"`` when only the time-window axis is active).

    Args:
        base: Source dataset name (e.g. ``"ETTh2"``).
        tw_idx: Time-window index, or ``None`` if no slicing.
        vs_idx: Variable-subset index, or ``None`` if no subsampling.
        hg_idx: Horizon-group index, or ``None`` if no horizon variation.
        has_windows: Whether the parent run uses time-window slicing.
        has_variable_subsets: Whether the parent run uses variable subsampling.
        has_horizon_groups: Whether the parent run uses horizon variation.

    Returns:
        Canonical task_id string.
    """
    out = base
    if has_windows and tw_idx is not None:
        out += f":tw{tw_idx}"
    if has_variable_subsets and vs_idx is not None:
        out += f":vs{vs_idx}"
    if has_horizon_groups and hg_idx is not None:
        out += f":hg{hg_idx}"
    return out


def parse_task_id(task_id: str) -> TaskIdParts:
    """Inverse of :func:`build_task_id`.

    Tag order is not required — any combination of ``tw{i}`` / ``vs{j}`` /
    ``hg{k}`` suffixes is accepted in any order, and missing tags resolve
    to ``None``.  Raises :class:`ValueError` on unrecognised suffixes.
    """
    parts = task_id.split(":")
    base = parts[0]
    tw_idx: Optional[int] = None
    vs_idx: Optional[int] = None
    hg_idx: Optional[int] = None
    for tag in parts[1:]:
        if tag.startswith("tw"):
            tw_idx = int(tag[2:])
        elif tag.startswith("vs"):
            vs_idx = int(tag[2:])
        elif tag.startswith("hg"):
            hg_idx = int(tag[2:])
        else:
            raise ValueError(f"Unrecognised task_id suffix {tag!r} in {task_id!r}")
    return TaskIdParts(base=base, tw_idx=tw_idx, vs_idx=vs_idx, hg_idx=hg_idx)


# ---------------------------------------------------------------------------
# Variable-subset sampling (AutoCTS++-style stratified buckets)
# ---------------------------------------------------------------------------

def _sample_variable_subsets(
    n_raw_vars: int,
    n_subsets: int,
    rates: List[float],
    seed: int,
    min_subset_size: int = 2,
) -> List[List[int]]:
    """Pick *n_subsets* stratified random subsets of variable indices.

    Mirrors ``reference/AutoCTS_plusplus/exps/dataset_slice.py`` — variables
    are bucketed by fractional size (``rates``), and we draw subsets evenly
    across buckets so the comparator sees both narrow and wide variants of
    the same source.

    Args:
        n_raw_vars: Total raw-variable channels available (excludes time
            covariates for ETT-style datasets).
        n_subsets: Total number of subsets to return.
        rates: Fractional bucket centres (e.g. ``[0.25, 0.5, 0.75]``).  Each
            subset's size is drawn from
            ``round(rate × n_raw_vars)`` for the assigned bucket, clipped
            to ``[min_subset_size, n_raw_vars]``.
        seed: RNG seed for reproducibility.
        min_subset_size: Floor on per-subset variable count (avoid degenerate
            singletons that have no useful contrastive structure).

    Returns:
        List of ``n_subsets`` sorted index lists.  Empty list when
        ``n_raw_vars < min_subset_size`` (no useful subsampling possible).
    """
    if n_raw_vars < min_subset_size or n_subsets <= 0:
        return []
    if not rates:
        rates = [0.5]

    rng = np.random.default_rng(seed)
    subsets: List[List[int]] = []
    for s in range(n_subsets):
        rate = rates[s % len(rates)]
        size = int(round(rate * n_raw_vars))
        size = max(min_subset_size, min(size, n_raw_vars))
        idx = sorted(rng.choice(n_raw_vars, size=size, replace=False).tolist())
        subsets.append(idx)
    return subsets


def _slice_scaler(scaler: Optional[Any], idx_list: List[int]) -> Optional[Any]:
    """Return a deep-copied scaler whose per-feature stats are sliced to ``idx_list``.

    The parent scaler is fit on the full raw-variable column set during data
    loading (e.g. PEMS03 → 358 sensors, ExchangeRate → 8 currencies).  When
    variable subsampling reduces the column count, the subsetted dataset
    must carry a scaler whose ``mean_`` / ``scale_`` / ``var_`` arrays match
    the kept columns — otherwise ``scaler.inverse_transform`` inside
    :func:`utils.metrics.compute_forecasting_metrics` raises a broadcast
    ``ValueError`` (e.g. ``(N, 90)`` vs ``(358,)``).  That exception is
    silently swallowed by ``_quick_eval`` and turns every candidate's
    performance into a ``-1e9`` sentinel, poisoning seed generation.

    Args:
        scaler: A fitted sklearn ``StandardScaler`` (or ``None``).
        idx_list: Raw-variable column indices to keep.  Must be valid
            indices into ``scaler.mean_`` / ``scaler.scale_``.

    Returns:
        A deep copy of ``scaler`` with stats sliced to ``idx_list``, or
        ``None`` when the input is ``None``.
    """
    if scaler is None:
        return None
    sub = copy.deepcopy(scaler)
    idx = np.asarray(idx_list, dtype=np.int64)
    if getattr(sub, "mean_", None) is not None:
        sub.mean_ = sub.mean_[idx]
    if getattr(sub, "scale_", None) is not None:
        sub.scale_ = sub.scale_[idx]
    if getattr(sub, "var_", None) is not None:
        sub.var_ = sub.var_[idx]
    sub.n_features_in_ = len(idx_list)
    return sub


# ---------------------------------------------------------------------------
# Forecasting sub-task expansion (time windows + horizon groups)
# ---------------------------------------------------------------------------

def make_forecasting_subtasks(
    name: str,
    data_dir: str,
    n_time_windows: int = 1,
    horizon_groups: Optional[List[Optional[List[int]]]] = None,
    crop_len: Optional[int] = None,
    min_window_len: int = 1000,
    n_variable_subsets: int = 1,
    var_size_rates: Optional[List[float]] = None,
    min_var_count: int = 4,
    var_subset_seed: int = 0,
) -> List[ForecastingSubTask]:
    """Expand a source dataset into multiple sub-tasks.

    The three orthogonal axes (time-window × variable-subset × horizon-group)
    fan out as a Cartesian product:
        ``n_time_windows × n_variable_subsets`` ForecastingSubTask objects,
        each carrying the full ``horizon_groups`` list for downstream eval.
    Backward-compatible defaults (``n_time_windows=1`` /
    ``n_variable_subsets=1`` / ``horizon_groups in (None, [None])``) yield
    a single sub-task identical to the legacy pipeline.

    Args:
        name: Source dataset name (e.g. ``"ETTh2"``).
        data_dir: Root data directory passed to :func:`load_dataset`.
        n_time_windows: Number of non-overlapping temporal windows.  ``1``
            disables time-window slicing.
        horizon_groups: List of horizon sets used at downstream eval time.
            ``None`` / empty / ``[None]`` keeps the canonical
            ``[24, 48, 168, 336, 720]`` eval.
        crop_len: Sliding-window crop length forwarded to dataset loading
            and to per-sub-task training datasets.
        min_window_len: Skip time-window slicing (fall back to single window)
            when the per-window length would drop below this value.
        n_variable_subsets: AutoCTS++-style variable subsampling — number of
            random subsets of raw-variable channels per (window).  ``1``
            disables subsampling; ``≥2`` engages stratified bucket sampling
            via :func:`_sample_variable_subsets`.  Datasets with fewer than
            ``min_var_count`` raw variables (e.g. ETT in univariate mode)
            silently bypass this axis — their sub-tasks have ``vs_idx=None``.
        var_size_rates: Fractional bucket centres for variable-subset sizing.
            Defaults to ``[0.25, 0.5, 0.75]`` (mirrors AutoCTS++).
        min_var_count: Minimum raw-variable channel count required to enable
            subsampling.  Datasets below this skip the axis entirely.
        var_subset_seed: Seed for the variable-subset RNG so sub-task IDs
            are deterministic across the seed-gen and comparator-pretrain
            scripts.

    Returns:
        ``List[ForecastingSubTask]``.  Every sub-task carries the same
        ``horizon_groups`` list; the caller emits ``len(horizon_groups)``
        seed records per (sub-task, candidate).
    """
    if not horizon_groups:
        horizon_groups = [None]
    if var_size_rates is None:
        var_size_rates = [0.25, 0.5, 0.75]

    splits = load_dataset(name, data_dir, window_len_override=crop_len)
    train_ds = splits["train"]
    val_ds = splits.get("val") or splits["test"]
    scaler = getattr(train_ds, "scaler", None)
    n_cov = int(getattr(train_ds, "n_covariate_cols", 0))

    # ── Backward-compat fast path: no slicing on any axis ──────────────
    if n_time_windows <= 1 and n_variable_subsets <= 1:
        return [
            ForecastingSubTask(
                base_name=name,
                tw_idx=None,
                vs_idx=None,
                train=train_ds,
                val=val_ds,
                horizon_groups=list(horizon_groups),
            )
        ]

    # ── The slicing axes only make sense for forecasting datasets ──────
    if train_ds.task_type != "forecasting":
        logger.warning(
            "make_forecasting_subtasks: %s has task_type=%s; time-window / "
            "variable-subset slicing is forecasting-only — emitting a "
            "single sub-task.",
            name, train_ds.task_type,
        )
        return [
            ForecastingSubTask(
                base_name=name, tw_idx=None, vs_idx=None,
                train=train_ds, val=val_ds,
                horizon_groups=list(horizon_groups),
            )
        ]

    # ── Concatenate train+val+test back into the full continuous series ──
    train_arr = train_ds.data.numpy()                        # (1, T_tr, C)
    val_arr   = val_ds.data.numpy()                          # (1, T_va, C)
    parts = [train_arr, val_arr]
    test_split = splits.get("test")
    if test_split is not None and test_split is not val_ds:
        parts.append(test_split.data.numpy())
    full = np.concatenate(parts, axis=1)                     # (1, T_total, C)
    T_total = full.shape[1]
    n_raw_vars = full.shape[2] - n_cov

    # ── Resolve the time-window axis (start_end pairs + window indices) ─
    use_windows = n_time_windows > 1
    if use_windows:
        seg_len = T_total // n_time_windows
        if seg_len < min_window_len:
            logger.warning(
                "make_forecasting_subtasks: %s segment length (%d) < min "
                "(%d); disabling time-window slicing for this source.",
                name, seg_len, min_window_len,
            )
            use_windows = False

    if use_windows:
        time_windows = [
            (tw, tw * seg_len, tw * seg_len + seg_len)
            for tw in range(n_time_windows)
        ]
    else:
        time_windows = [(None, 0, T_total)]

    # ── Resolve the variable-subset axis ──────────────────────────────
    use_var_subsets = n_variable_subsets > 1
    var_subsets: List[Tuple[Optional[int], Optional[List[int]]]] = []
    if use_var_subsets:
        if n_raw_vars < min_var_count:
            logger.info(
                "make_forecasting_subtasks: %s has only %d raw variables "
                "(< min_var_count=%d); skipping variable subsampling.",
                name, n_raw_vars, min_var_count,
            )
            use_var_subsets = False
        else:
            sampled = _sample_variable_subsets(
                n_raw_vars=n_raw_vars,
                n_subsets=n_variable_subsets,
                rates=var_size_rates,
                seed=var_subset_seed + hash(name) % (2**31),
            )
            var_subsets = [(i, idx) for i, idx in enumerate(sampled)]

    if not use_var_subsets:
        var_subsets = [(None, None)]

    # ── Per-window split ratio mirrors AutoCTS++'s in-window 80/20 ─────
    train_ratio = 0.8
    out: List[ForecastingSubTask] = []

    for tw_idx, t_start, t_end in time_windows:
        seg = full[:, t_start:t_end, :]                      # (1, seg_len, C)
        seg_len_actual = seg.shape[1]
        n_train = int(seg_len_actual * train_ratio) if use_windows else train_arr.shape[1]
        if not use_windows:
            # Preserve original chronological train/val split when the time
            # axis is *not* sliced — re-deriving an 80/20 split from the
            # concatenated series would silently change the val targets
            # for backward-compat callers.
            win_train = train_arr.astype(np.float32)
            win_val = val_arr.astype(np.float32)
        else:
            win_train = seg[:, :n_train, :].astype(np.float32)
            win_val   = seg[:, n_train:, :].astype(np.float32)

        if win_train.shape[1] < 64 or win_val.shape[1] < 32:
            logger.warning(
                "make_forecasting_subtasks: %s window %s too short "
                "(train=%d, val=%d); skipping.",
                name, tw_idx, win_train.shape[1], win_val.shape[1],
            )
            continue

        win_crop = crop_len
        if win_crop is not None and win_crop >= win_train.shape[1]:
            win_crop = max(64, win_train.shape[1] // 2)
            logger.info(
                "make_forecasting_subtasks: %s tw=%s crop_len reduced %s → "
                "%d to fit window length %d.",
                name, tw_idx, crop_len, win_crop, win_train.shape[1],
            )

        for vs_idx, var_idx_list in var_subsets:
            if var_idx_list is None:
                # Full variable set → no per-window column slicing needed.
                tr_arr = win_train
                va_arr = win_val
                sub_scaler = scaler
            else:
                # Keep the n_cov covariate columns (always at the front)
                # plus the chosen raw-variable columns.  Index arithmetic:
                # cov columns occupy [0, n_cov); raw-var indices ``i``
                # become ``n_cov + i`` in the full tensor.
                cols = list(range(n_cov)) + [n_cov + i for i in var_idx_list]
                tr_arr = win_train[:, :, cols].copy()
                va_arr = win_val[:, :, cols].copy()
                # Match the scaler feature count to the sliced raw-variable
                # set so ``compute_forecasting_metrics`` can call
                # ``inverse_transform`` without a broadcast error.  The
                # parent scaler operates on raw variables only (covariates
                # are stripped before the inverse transform), so slicing
                # by ``var_idx_list`` directly is correct for both the
                # n_cov=0 sources (PEMS / ExchangeRate / pems-bay) and the
                # n_cov>0 ETT sources.
                sub_scaler = _slice_scaler(scaler, var_idx_list)

            train_sub = TimeSeriesDataset(
                tr_arr, None, "forecasting", max_len=None,
                window_len=win_crop, window_stride=1,
                scaler=sub_scaler, n_covariate_cols=n_cov,
            )
            val_sub = TimeSeriesDataset(
                va_arr, None, "forecasting", max_len=None,
                scaler=sub_scaler, n_covariate_cols=n_cov,
            )
            sub = ForecastingSubTask(
                base_name=name,
                tw_idx=tw_idx,
                vs_idx=vs_idx,
                train=train_sub,
                val=val_sub,
                horizon_groups=list(horizon_groups),
                var_indices=list(var_idx_list) if var_idx_list is not None else None,
            )
            out.append(sub)
            logger.info(
                "make_forecasting_subtasks: %s tw=%s vs=%s  train=(1,%d,%d) "
                "val=(1,%d,%d)  horizon_groups=%d",
                name, tw_idx, vs_idx,
                tr_arr.shape[1], tr_arr.shape[2],
                va_arr.shape[1], va_arr.shape[2],
                len(horizon_groups),
            )

    if not out:
        logger.warning(
            "make_forecasting_subtasks: %s produced 0 valid sub-tasks; "
            "falling back to the original splits.",
            name,
        )
        return [
            ForecastingSubTask(
                base_name=name, tw_idx=None, vs_idx=None,
                train=train_ds, val=val_ds,
                horizon_groups=list(horizon_groups),
            )
        ]
    return out


# ---------------------------------------------------------------------------
# Legacy slicer (kept to avoid breaking imports — not used by the new pipeline)
# ---------------------------------------------------------------------------

def _time_window_slice(
    data: np.ndarray,
    n_windows: int,
) -> List[np.ndarray]:
    """Split (N, T, C) into *n_windows* non-overlapping temporal segments."""
    N, T, C = data.shape
    seg_len = T // n_windows
    if seg_len < 8:
        return []
    return [data[:, i * seg_len : (i + 1) * seg_len, :] for i in range(n_windows)]


def _channel_subset(data: np.ndarray) -> np.ndarray:
    """Randomly select ⌈C/2⌉ channels from (N, T, C)."""
    N, T, C = data.shape
    if C <= 1:
        return data
    k = math.ceil(C / 2)
    chosen = sorted(random.sample(range(C), k))
    return data[:, :, chosen]


def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Subsample the time axis of (N, T, C) by *factor*."""
    return data[:, ::factor, :]


def slice_dataset(
    dataset: TimeSeriesDataset,
    n_subsets: int = 10,
    random_seed: Optional[int] = None,
) -> List[TimeSeriesDataset]:
    """Legacy round-robin slicer kept for backward compatibility.

    The new pipeline calls :func:`make_forecasting_subtasks` instead.  This
    helper is unused by the seed generator and is preserved only so external
    callers that imported it before the refactor still resolve.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    data_np = dataset.data.numpy()
    labels_np = dataset.labels.numpy() if dataset.labels is not None else None
    task_type = dataset.task_type

    subsets: List[TimeSeriesDataset] = []
    strategy_cycle = ["time_window", "channel_subset", "downsample"]

    for i in range(n_subsets):
        strategy = strategy_cycle[i % len(strategy_cycle)]
        sliced: Optional[np.ndarray] = None

        if strategy == "time_window":
            n_splits = 4 if i % 6 < 3 else 2
            windows = _time_window_slice(data_np, n_splits)
            if windows:
                sliced = windows[i % len(windows)]
        elif strategy == "channel_subset":
            sliced = _channel_subset(data_np)
        elif strategy == "downsample":
            factor = 4 if i % 4 < 2 else 2
            ds = _downsample(data_np, factor)
            if ds.shape[1] >= 8:
                sliced = ds

        if sliced is None or sliced.shape[1] < 8 or sliced.shape[0] < 2:
            sliced = data_np

        sliced_labels: Optional[np.ndarray] = None
        if labels_np is not None:
            sliced_labels = labels_np[: sliced.shape[0]]

        try:
            subsets.append(
                TimeSeriesDataset(
                    sliced.astype(np.float32),
                    sliced_labels,
                    task_type,
                    max_len=None,
                )
            )
        except ValueError as exc:
            logger.warning("slice_dataset[%d] skipped: %s", i, exc)

    return subsets
