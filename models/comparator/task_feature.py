"""Cached TS2Vec-aligned task feature loader.

This module is now a **cached loader**: task features are precomputed offline
via :mod:`scripts.precompute_task_features` and saved as ``.npy`` files under
``cache_dir``.  At runtime :meth:`TaskFeatureExtractor.extract` loads the
saved tensor and returns it on the requested device.

Saved shape: ``(N_set, seq_len, repr_dim)`` — mirrors AutoCTS++'s
``generate_task_feature.py`` output, i.e. a *set* of ``N_set`` sequence-summary
representations sampled from the encoded source data.

Why
---
The previous design — train a fresh 4-layer encoder for 5 epochs each call,
then collapse the output to ``mean / std / quantiles`` — was
information-poor (different datasets ended up with near-identical statistics
after z-score normalisation) and non-stationary (each call produced
different features for the same dataset depending on initialisation).
Aligning with AutoCTS++ — full TS2Vec-style pretrain + sliding-window
encode + sample 100 windows — gives the comparator a richer, deterministic
input.

Backward compatibility
----------------------
The public API of :class:`TaskFeatureExtractor` is preserved.  Old keyword
arguments (``encoder_config``, ``pretrain_epochs``, ``lr``, ``batch_size``)
are accepted but ignored with a one-time warning.

Constants
---------
``TASK_FEATURE_REPR_DIM`` — per-element repr dim (default 128).
``TASK_FEATURE_SET_SIZE`` — number of set elements (default 100).
``TASK_FEATURE_SEQ_LEN``  — sequence length per set element (default 12).
``TASK_FEATURE_DIM``      — alias for ``TASK_FEATURE_REPR_DIM``, kept so
                            historic callers ``from .task_feature import
                            TASK_FEATURE_DIM`` keep working (the value's
                            *meaning* changes from "total feature dim" to
                            "per-element dim", which is what TCLSC now needs).
``DEFAULT_CACHE_DIR``     — where the precompute script writes ``.npy``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public constants
# --------------------------------------------------------------------------- #

TASK_FEATURE_REPR_DIM: int = 128       # per-element repr dim (= encoder output_dim)
TASK_FEATURE_SET_SIZE: int = 100       # set size (= AutoCTS++'s sample_num)
TASK_FEATURE_SEQ_LEN:  int = 12        # seq length per set element

# Alias kept for backward-compat (ex: ``from .task_feature import TASK_FEATURE_DIM``).
TASK_FEATURE_DIM:      int = TASK_FEATURE_REPR_DIM

DEFAULT_CACHE_DIR: str = "outputs/task_features"

# Set once we've already warned about deprecated args, to keep stderr quiet.
_DEPRECATION_WARNED = False


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

class TaskFeatureExtractor:
    """Cached loader for precomputed task features.

    The legacy in-place training pipeline (lightweight 4-layer encoder +
    pooled statistics) was removed because it produced non-stationary,
    information-poor features.  Features must now be precomputed via
    ``scripts/precompute_task_features.py``.

    Args:
        cache_dir: Directory holding ``{dataset_name}_task_feature.npy`` files.
            Defaults to :data:`DEFAULT_CACHE_DIR`.
        device: Device for the returned tensor.  ``None`` → auto-detect.
        encoder_config / pretrain_epochs / lr / batch_size: Deprecated
            no-op arguments kept for backward-compatibility with the old
            in-place pretrainer.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[torch.device] = None,
        # Deprecated — accepted for backward-compat, ignored.
        encoder_config=None,
        pretrain_epochs: int = 0,
        lr: float = 0.0,
        batch_size: int = 0,
    ) -> None:
        global _DEPRECATION_WARNED
        if (
            encoder_config is not None
            or pretrain_epochs
            or lr
            or batch_size
        ) and not _DEPRECATION_WARNED:
            logger.warning(
                "TaskFeatureExtractor: in-place pretrain args are deprecated "
                "and ignored — features must be precomputed via "
                "scripts/precompute_task_features.py.",
            )
            _DEPRECATION_WARNED = True

        self.cache_dir = cache_dir
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #

    def extract(
        self,
        dataset,
        task_type: str,
        horizon: int = 0,
        dataset_name: Optional[str] = None,
    ) -> Tensor:
        """Load precomputed task features for *dataset*.

        Args:
            dataset: A :class:`~data.dataset.TimeSeriesDataset`.  Used only
                to look up an attached ``.name`` attribute when
                *dataset_name* is omitted.
            task_type: Kept for backward-compatible signature; unused (the
                cache key is the dataset name only).
            horizon: Kept for backward-compatible signature; unused.
            dataset_name: Logical dataset name for cache lookup (e.g.
                ``"ETTh1"``, ``"PEMS03"``).  When omitted, falls back to
                ``getattr(dataset, "name", None)``.

        Returns:
            Tensor of shape ``(N_set, seq_len, D)`` on ``self.device``.

        Raises:
            ValueError: If neither *dataset_name* nor ``dataset.name`` is set.
            FileNotFoundError: If no cached feature file is found.
        """
        key = dataset_name or getattr(dataset, "name", None)
        if not key:
            raise ValueError(
                "TaskFeatureExtractor.extract: 'dataset_name' is required "
                "when the dataset object has no '.name' attribute. "
                "Either pass dataset_name=… explicitly, or load the dataset "
                "via data.dataset.load_dataset (which attaches '.name' "
                "automatically).",
            )

        path = os.path.join(self.cache_dir, f"{key}_task_feature.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Task feature cache missing for {key!r}: expected {path}.\n"
                f"Run `python scripts/precompute_task_features.py "
                f"--datasets {key} --data_dir <data_root>` first.",
            )

        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(
                f"Task feature {path} has unexpected shape {arr.shape}; "
                f"expected (N_set, seq_len, D).",
            )
        return torch.from_numpy(arr).float().to(self.device)
