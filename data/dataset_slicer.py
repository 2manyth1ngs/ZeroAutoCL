"""Dataset slicing for T-CLSC pre-training diversity.

Generates multiple sub-datasets from one source dataset via three strategies:
  1. **Time windows**   — split long series into non-overlapping segments
  2. **Channel subset** — randomly select ⌈C/2⌉ channels when C > 1
  3. **Down-sampling**  — subsample time axis by factor 2 or 4

Each sub-dataset is treated as an independent task during seed generation,
so even 3 source datasets can produce 30+ diverse tasks (参照 AutoCTS++).
"""

from __future__ import annotations

import logging
import math
import random
from typing import List, Optional

import numpy as np
import torch

from data.dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


def _time_window_slice(
    data: np.ndarray,
    n_windows: int,
) -> List[np.ndarray]:
    """Split (N, T, C) into *n_windows* non-overlapping temporal segments.

    Each segment has shape (N, T // n_windows, C).  Remainder steps are
    discarded.
    """
    N, T, C = data.shape
    seg_len = T // n_windows
    if seg_len < 8:
        return []
    windows = []
    for i in range(n_windows):
        windows.append(data[:, i * seg_len : (i + 1) * seg_len, :])
    return windows


def _channel_subset(data: np.ndarray) -> np.ndarray:
    """Randomly select ⌈C/2⌉ channels from (N, T, C).

    Returns the original array unchanged when C == 1.
    """
    N, T, C = data.shape
    if C <= 1:
        return data
    k = math.ceil(C / 2)
    chosen = sorted(random.sample(range(C), k))
    return data[:, :, chosen]


def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Subsample the time axis of (N, T, C) by *factor* (2 or 4)."""
    return data[:, ::factor, :]


def slice_dataset(
    dataset: TimeSeriesDataset,
    n_subsets: int = 10,
    random_seed: Optional[int] = None,
) -> List[TimeSeriesDataset]:
    """Generate *n_subsets* diverse sub-datasets from *dataset*.

    Subsets are produced by applying one of three strategies (chosen round-
    robin to guarantee variety), with parameter variation to avoid duplicates:

    =====================  ================================================
    Strategy               Description
    =====================  ================================================
    ``time_window``        Non-overlapping temporal segments (2 or 4 splits)
    ``channel_subset``     Random ⌈C/2⌉ channel selection
    ``downsample``         Temporal sub-sampling by factor 2 or 4
    =====================  ================================================

    Args:
        dataset: Source :class:`~data.dataset.TimeSeriesDataset`.
        n_subsets: Number of sub-datasets to generate (≥ 1).
        random_seed: Seed for reproducible channel selection.

    Returns:
        List of :class:`~data.dataset.TimeSeriesDataset` instances, each
        flagged with the same ``task_type`` as the original.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    data_np = dataset.data.numpy()        # (N, T, C)
    labels_np = (
        dataset.labels.numpy() if dataset.labels is not None else None
    )
    task_type = dataset.task_type
    N, T, C = data_np.shape

    subsets: List[TimeSeriesDataset] = []
    strategy_cycle = ["time_window", "channel_subset", "downsample"]

    for i in range(n_subsets):
        strategy = strategy_cycle[i % len(strategy_cycle)]
        sliced: Optional[np.ndarray] = None

        # ── Time window ──────────────────────────────────────────────
        if strategy == "time_window":
            n_splits = 4 if i % 6 < 3 else 2   # alternate between 4-way and 2-way
            windows = _time_window_slice(data_np, n_splits)
            if windows:
                sliced = windows[i % len(windows)]

        # ── Channel subset ───────────────────────────────────────────
        elif strategy == "channel_subset":
            sliced = _channel_subset(data_np)

        # ── Down-sampling ─────────────────────────────────────────────
        elif strategy == "downsample":
            factor = 4 if i % 4 < 2 else 2
            ds = _downsample(data_np, factor)
            if ds.shape[1] >= 8:
                sliced = ds

        # Fallback: use the original data if strategy produced nothing
        if sliced is None or sliced.shape[1] < 8 or sliced.shape[0] < 2:
            logger.debug(
                "slice_dataset[%d]: strategy %r produced unusable subset, "
                "falling back to original.",
                i, strategy,
            )
            sliced = data_np

        # Derive corresponding labels if present
        sliced_labels: Optional[np.ndarray] = None
        if labels_np is not None:
            # Labels are sample-level (N,); they remain unchanged for
            # time-window and downsampling; channel selection keeps all samples
            sliced_labels = labels_np[: sliced.shape[0]]

        try:
            sub_ds = TimeSeriesDataset(
                sliced.astype(np.float32),
                sliced_labels,
                task_type,
                max_len=None,    # already at target length; skip re-truncation
            )
            subsets.append(sub_ds)
            logger.debug(
                "slice_dataset[%d] strategy=%s  shape=%s",
                i, strategy, tuple(sliced.shape),
            )
        except ValueError as exc:
            logger.warning("slice_dataset[%d] skipped: %s", i, exc)

    return subsets
