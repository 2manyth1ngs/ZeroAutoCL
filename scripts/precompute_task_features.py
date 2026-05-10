"""Offline precompute of TS2Vec-aligned task features.

For each named dataset, this script:

  1. Trains a 10-layer dilated-CNN encoder (output_dim = ``--repr_dim``,
     default 128) under the GGS contrastive strategy for ``--epochs`` epochs.
  2. Performs a *causal sliding* encode of the full training series via
     :func:`train.forecasting_eval.causal_sliding_encode` (left-zero-padded
     window of length ``--padding`` per timestep).
  3. Slices the resulting ``(T, D)`` representation into non-overlapping
     ``(seq_len, D)`` windows.
  4. Randomly samples ``--n_set`` of those windows.
  5. Saves a single tensor of shape ``(N_set, seq_len, D)`` to
     ``{cache_dir}/{slug}_task_feature.npy``.

This mirrors ``reference/AutoCTS_plusplus/exps/generate_task_feature.py`` —
the cached output is what :class:`models.comparator.task_feature
.TaskFeatureExtractor` reads at runtime.

Two modes
---------
* **Default (base mode)**: one task feature per dataset name.  Used for
  *target* datasets (zero-shot inference sees the target as a single task).
* ``--sub_task_mode``: for forecasting sources, expand each dataset into
  ``n_time_windows × n_variable_subsets`` sub-tasks (re-using
  ``make_forecasting_subtasks`` with the same params as seed-gen), and train
  a separate encoder + sample a separate task feature per sub-task.  Cache
  files are named after the sub-task slug (e.g. ``ETTh2__tw0__vs1``).
  Requires ``--config`` to read the slicing params from the YAML.  Mirrors
  AutoCTS++'s ``--loader subset`` flow where each subset is its own dataset.

Usage
-----
::

    # Base mode (targets):
    python scripts/precompute_task_features.py \\
        --datasets ETTh1 Electricity Weather \\
        --data_dir data/datasets \\
        --cache_dir outputs/task_features \\
        --seed 42

    # Sub-task mode (sources):
    python scripts/precompute_task_features.py \\
        --datasets ETTh2 ETTm1 ETTm2 Solar traffic \\
                   AQShunyi AQWanliu AQGuanyuan ExchangeRate \\
        --sub_task_mode \\
        --config configs/default.yaml \\
        --data_dir data/datasets \\
        --cache_dir outputs/task_features \\
        --seed 42

Re-run with ``--force`` to overwrite existing caches.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from data.dataset import load_dataset, TimeSeriesDataset
from data.dataset_slicer import make_forecasting_subtasks
from models.comparator.task_feature import (
    DEFAULT_CACHE_DIR,
    TASK_FEATURE_REPR_DIM,
    TASK_FEATURE_SET_SIZE,
    TASK_FEATURE_SEQ_LEN,
    task_feature_slug,
)
from models.contrastive.cl_pipeline import CLPipeline
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.encoder.encoder_config import EncoderConfig
from train.forecasting_eval import causal_sliding_encode
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


# GGS strategy from CLAUDE.md / AutoCLS Table 5.  Used only to drive the
# task-feature encoder's pretraining — independent of the search-space
# strategies that the comparator will rank later.
_GGS_STRATEGY = {
    "augmentation": {
        "resize": 0.2, "rescale": 0.3, "jitter": 0.0,
        "point_mask": 0.2, "freq_mask": 0.0, "crop": 0.2, "order": 3,
    },
    "embedding_transform": {"jitter_p": 0.7, "mask_p": 0.1, "norm_type": "none"},
    "pair_construction": {
        "instance": True, "temporal": False, "cross_scale": False,
        "kernel_size": 5, "pool_op": "avg", "adj_neighbor": False,
    },
    "loss": {"type": "infonce", "sim_func": "distance", "temperature": 1.0},
}

# Long (N, T, C) classification / anomaly datasets are flattened along the
# instance axis up to this many timesteps before causal_sliding_encode.  Big
# enough to cover seq_len * n_set windows on every dataset we use.
_FLATTEN_MAX_T = 50_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute TS2Vec-aligned task features.",
    )
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Names of datasets to precompute (e.g. PEMS03 ETTh1).")
    p.add_argument("--data_dir", required=True, help="Root data directory.")
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR,
                   help=f"Output dir for .npy files (default: {DEFAULT_CACHE_DIR}).")
    p.add_argument("--iters", type=int, default=600,
                   help="CL pretraining iter budget (matches TS2Vec's "
                        "n_iters=600 default for size>100k datasets, used "
                        "by AutoCTS++ generate_task_feature.py).  Iter "
                        "mode is mandatory here because ZeroAutoCL's "
                        "TimeSeriesDataset uses a stride-1 sliding-window "
                        "view, which inflates ``len(dataset)`` to "
                        "``T - window_len + 1`` (e.g. 12k+ for PEMS) — "
                        "epoch-based budgets would then run ~30k batches "
                        "per dataset and take hours.  Set --epochs to "
                        "override.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Optional epoch override.  When set, takes "
                        "precedence over --iters and runs in epoch mode.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8,
                   help="CL pretraining batch size (small — long crops).")
    p.add_argument("--n_set", type=int, default=TASK_FEATURE_SET_SIZE,
                   help="Number of windows to sample per dataset.")
    p.add_argument("--seq_len", type=int, default=TASK_FEATURE_SEQ_LEN,
                   help="Length of each sampled window.")
    p.add_argument("--repr_dim", type=int, default=TASK_FEATURE_REPR_DIM,
                   help="Encoder output dim (= per-element repr dim of the cache).")
    p.add_argument("--n_layers", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--padding", type=int, default=200,
                   help="Causal sliding-encode left-padding length.")
    p.add_argument("--slide_batch", type=int, default=128,
                   help="Sliding-window batch for causal encode "
                        "(reduce on wide datasets like PEMS07).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Recompute even when {cache_dir}/{ds}_task_feature.npy exists.")
    p.add_argument("--device", default=None,
                   help="Torch device override (e.g. 'cuda:0', 'cpu').")
    # Option B (2026-05-10): sub-task-mode aligns with AutoCTS++ §3.2.4 +
    # generate_task_feature.py's per-subset feature extraction.
    p.add_argument("--sub_task_mode", action="store_true",
                   help="For forecasting sources, expand each dataset into "
                        "n_time_windows × n_variable_subsets sub-tasks and "
                        "precompute one task feature per sub-task.  Requires "
                        "--config to read the slicing params from YAML.")
    p.add_argument("--config", default=None,
                   help="YAML config (typically configs/default.yaml) "
                        "providing forecasting_task_variants and "
                        "dataset_budgets for --sub_task_mode.")
    return p.parse_args()


def _flatten_for_causal_encode(data: torch.Tensor) -> torch.Tensor:
    """Reshape ``data`` so :func:`causal_sliding_encode` can consume it.

    Forecasting datasets are stored as ``(1, T, C)`` and pass through.
    Classification / anomaly datasets are stored as ``(N, T, C)`` — we
    concatenate along the time axis (capped at :data:`_FLATTEN_MAX_T`)
    so the result is a single long ``(1, T_flat, C)`` series.

    Args:
        data: Raw ``self.data`` of a :class:`TimeSeriesDataset`.

    Returns:
        Tensor of shape ``(1, T, C)``.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected (N, T, C), got {tuple(data.shape)}")
    if data.shape[0] == 1:
        return data
    flat = data.reshape(-1, data.shape[-1])           # (N*T, C)
    if flat.shape[0] > _FLATTEN_MAX_T:
        logger.info(
            "  flattening (%d, %d, %d) → (1, %d, %d) (capped at %d)",
            data.shape[0], data.shape[1], data.shape[2],
            _FLATTEN_MAX_T, data.shape[2], _FLATTEN_MAX_T,
        )
        flat = flat[:_FLATTEN_MAX_T]
    else:
        logger.info(
            "  flattening (%d, %d, %d) → (1, %d, %d)",
            data.shape[0], data.shape[1], data.shape[2],
            flat.shape[0], data.shape[2],
        )
    return flat.unsqueeze(0)                          # (1, T_flat, C)


def _train_encode_sample(
    train_ds: TimeSeriesDataset,
    label: str,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """Shared core: train one encoder on *train_ds*, sliding-encode, sample windows.

    Used by both base mode (full dataset) and sub-task mode (each
    (tw, vs)-sliced view).

    Args:
        train_ds: Pre-loaded training dataset (``(1, T, C)`` for forecasting).
        label: Human-readable identifier for log lines (the dataset name in
            base mode, the sub-task window_id in sub-task mode).
        args: Parsed argparse namespace.
        device: Torch device.

    Returns:
        ``(n_set, seq_len, repr_dim)`` ``float32`` numpy array.
    """
    input_dim = train_ds.n_channels
    logger.info(
        "[%s] train data shape=%s  task_type=%s  input_dim=%d",
        label, tuple(train_ds.data.shape), train_ds.task_type, input_dim,
    )

    encoder_cfg = EncoderConfig(
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        output_dim=args.repr_dim,
    )
    encoder = DilatedCNNEncoder(input_dim, encoder_cfg).to(device)
    pipeline = CLPipeline(encoder, _GGS_STRATEGY).to(device)

    pretrain_cfg = {
        "pretrain_lr":     args.lr,
        "batch_size":      args.batch_size,
        "use_ema":         True,
    }
    if args.epochs is not None:
        pretrain_cfg["pretrain_epochs"] = int(args.epochs)
        logger.info(
            "[%s] training task-feature encoder for %d epochs (epoch mode)",
            label, args.epochs,
        )
    else:
        pretrain_cfg["pretrain_iters"] = int(args.iters)
        logger.info(
            "[%s] training task-feature encoder for %d iters (iter mode)",
            label, args.iters,
        )
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config=pretrain_cfg,
        device=device,
        task_type=train_ds.task_type,
    )

    encoder.eval()
    full_series = _flatten_for_causal_encode(train_ds.data)   # (1, T, C)
    logger.info(
        "[%s] causal sliding encode: T=%d  padding=%d  slide_batch=%d",
        label, full_series.shape[1], args.padding, args.slide_batch,
    )
    repr_full = causal_sliding_encode(
        encoder, full_series,
        padding=args.padding,
        batch_size=args.slide_batch,
        device=device,
    )                                                          # (T, D)

    T, D = repr_full.shape
    if D != args.repr_dim:
        # Should not happen with our encoder, but guard anyway.
        raise RuntimeError(
            f"[{label}] encoder output_dim={D} != --repr_dim={args.repr_dim}",
        )

    bn = T // args.seq_len
    if bn == 0:
        raise RuntimeError(
            f"[{label}] series too short ({T}) for seq_len={args.seq_len}",
        )

    repr_windows = repr_full[: bn * args.seq_len].reshape(bn, args.seq_len, D)

    rng = np.random.default_rng(args.seed)
    if bn < args.n_set:
        logger.warning(
            "[%s] only %d windows of length %d available (requested %d) — "
            "sampling with replacement", label, bn, args.seq_len, args.n_set,
        )
        sample_idx = rng.choice(bn, size=args.n_set, replace=True)
    else:
        sample_idx = rng.choice(bn, size=args.n_set, replace=False)
    sample_idx.sort()
    sample_repr = repr_windows[sample_idx]                     # (n_set, seq_len, D)

    return sample_repr.astype(np.float32)


def precompute_one(
    name: str, args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    """Base mode: precompute task feature for one dataset.

    Loads ``name`` via ``load_dataset`` and runs the shared training +
    sampling pipeline on the full training split.

    Returns:
        ``(n_set, seq_len, repr_dim)`` ``float32`` numpy array.
    """
    splits = load_dataset(name, args.data_dir)
    return _train_encode_sample(splits["train"], name, args, device)


def precompute_sub_tasks(
    name: str,
    args: argparse.Namespace,
    device: torch.device,
    variants_cfg: dict,
    dataset_budgets: dict,
    save_dir: str,
    default_crop_len: Optional[int] = None,
) -> tuple:
    """Sub-task mode: precompute one task feature per (tw, vs) sub-task.

    Mirrors AutoCTS++'s ``generate_task_feature.py`` invoked with
    ``--loader subset`` for each subset file produced by
    ``dataset_slice.py``.  We achieve the same effect on-the-fly by replaying
    ``make_forecasting_subtasks`` with the same params seed-gen used, then
    training a fresh encoder per sub-task.

    Args:
        name: Source dataset name.
        args: Parsed argparse namespace.
        device: Torch device.
        variants_cfg: ``forecasting_task_variants`` block from the YAML.
        dataset_budgets: ``dataset_budgets`` block from the YAML (per-source
            ``n_time_windows`` / ``n_variable_subsets`` / ``crop_len`` overrides).
        save_dir: Directory to write the ``{slug}_task_feature.npy`` files.
        default_crop_len: Fallback ``crop_len`` read from the YAML's
            ``seed_generation.crop_len`` block.  Per-source budget entries
            override this.

    Returns:
        ``(n_ok, n_skip, n_fail)`` — count of newly-written, cache-hit, and
        failed sub-task files for THIS source.
    """
    ds_budget = (dataset_budgets or {}).get(name, {}) or {}
    n_tw_global = int(variants_cfg.get("n_time_windows", 1) or 1)
    n_vs_global = int(variants_cfg.get("n_variable_subsets", 1) or 1)
    ds_n_tw = int(ds_budget.get("n_time_windows", n_tw_global))
    ds_n_vs = int(ds_budget.get("n_variable_subsets", n_vs_global))

    crop_len = ds_budget.get("crop_len", default_crop_len)
    if crop_len is not None:
        crop_len = int(crop_len)

    min_window_len = int(variants_cfg.get("min_window_len", 1000) or 1000)
    var_size_rates = variants_cfg.get("var_size_rates")
    min_var_count = int(variants_cfg.get("min_var_count", 4) or 4)

    logger.info(
        "[%s] sub-task mode: n_time_windows=%d  n_variable_subsets=%d  "
        "crop_len=%s  min_window_len=%d  min_var_count=%d",
        name, ds_n_tw, ds_n_vs, crop_len, min_window_len, min_var_count,
    )

    sub_tasks = make_forecasting_subtasks(
        name, args.data_dir,
        n_time_windows=ds_n_tw,
        horizon_groups=[None],     # task feature is horizon-agnostic
        crop_len=crop_len,
        min_window_len=min_window_len,
        n_variable_subsets=ds_n_vs,
        var_size_rates=var_size_rates,
        min_var_count=min_var_count,
        var_subset_seed=args.seed,
    )
    logger.info(
        "[%s] expanded into %d sub-task(s) for task-feature precompute",
        name, len(sub_tasks),
    )

    n_ok = n_skip = n_fail = 0
    for sub in sub_tasks:
        window_id = sub.window_id
        slug = task_feature_slug(window_id)
        out_path = os.path.join(save_dir, f"{slug}_task_feature.npy")

        if os.path.exists(out_path) and not args.force:
            logger.info(
                "[%s] cache hit at %s — skipping (use --force to overwrite)",
                window_id, out_path,
            )
            n_skip += 1
            continue

        logger.info("=" * 60)
        logger.info(
            "[%s] precomputing sub-task feature → %s", window_id, out_path,
        )
        logger.info("=" * 60)

        try:
            arr = _train_encode_sample(sub.train, window_id, args, device)
        except Exception as exc:                              # pragma: no cover
            logger.exception("[%s] precompute failed: %s", window_id, exc)
            n_fail += 1
            continue

        tmp = out_path[:-4] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out_path)
        n_ok += 1
        logger.info(
            "[%s] saved %s shape=%s dtype=%s",
            window_id, out_path, arr.shape, arr.dtype,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return n_ok, n_skip, n_fail


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Sub-task mode plumbing ─────────────────────────────────────────────
    # When --sub_task_mode is set, the YAML supplies the slicing axes
    # (n_time_windows / n_variable_subsets / crop_len) so the sub-task
    # decomposition here matches the one seed-gen uses.  Without --config
    # this mode can't work; fail fast with a clear message.
    variants_cfg: dict = {}
    dataset_budgets: dict = {}
    default_crop_len: Optional[int] = None
    if args.sub_task_mode:
        if not args.config:
            raise SystemExit(
                "--sub_task_mode requires --config to read "
                "forecasting_task_variants and dataset_budgets from the YAML.",
            )
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        variants_cfg = cfg.get("forecasting_task_variants", {}) or {}
        dataset_budgets = cfg.get("dataset_budgets", {}) or {}
        sg_cfg = cfg.get("seed_generation", {}) or {}
        sg_crop_len = sg_cfg.get("crop_len")
        if sg_crop_len is not None:
            default_crop_len = int(sg_crop_len)
        logger.info(
            "[sub_task_mode] YAML loaded: %d budget entries, "
            "n_tw_global=%s n_vs_global=%s seed_gen.crop_len=%s",
            len(dataset_budgets),
            variants_cfg.get("n_time_windows"),
            variants_cfg.get("n_variable_subsets"),
            default_crop_len,
        )

    device = (
        torch.device(args.device) if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    logger.info("device=%s  cache_dir=%s  sub_task_mode=%s",
                device, args.cache_dir, args.sub_task_mode)

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for name in args.datasets:
        # ── Sub-task mode path ────────────────────────────────────────────
        if args.sub_task_mode:
            try:
                ok, skip, fail = precompute_sub_tasks(
                    name, args, device,
                    variants_cfg=variants_cfg,
                    dataset_budgets=dataset_budgets,
                    save_dir=args.cache_dir,
                    default_crop_len=default_crop_len,
                )
            except Exception as exc:                          # pragma: no cover
                logger.exception("[%s] sub-task precompute failed: %s", name, exc)
                n_fail += 1
                continue
            n_ok += ok
            n_skip += skip
            n_fail += fail
            continue

        # ── Base mode path (targets / standalone datasets) ────────────────
        out_path = os.path.join(args.cache_dir, f"{name}_task_feature.npy")
        if os.path.exists(out_path) and not args.force:
            logger.info("[%s] cache hit at %s — skipping (use --force to overwrite)",
                        name, out_path)
            n_skip += 1
            continue

        logger.info("=" * 60)
        logger.info("[%s] precomputing task feature → %s", name, out_path)
        logger.info("=" * 60)
        try:
            arr = precompute_one(name, args, device)
        except Exception as exc:                              # pragma: no cover
            logger.exception("[%s] precompute failed: %s", name, exc)
            n_fail += 1
            continue

        # Atomic write so a crash mid-save can't leave a corrupt cache.
        # tmp must end in .npy so np.save doesn't append a second .npy suffix.
        tmp = out_path[:-4] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out_path)
        n_ok += 1
        logger.info("[%s] saved %s shape=%s dtype=%s",
                    name, out_path, arr.shape, arr.dtype)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info(
        "Done — saved=%d  skipped=%d  failed=%d  (mode=%s, %d source datasets)",
        n_ok, n_skip, n_fail,
        "sub_task" if args.sub_task_mode else "base",
        len(args.datasets),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
