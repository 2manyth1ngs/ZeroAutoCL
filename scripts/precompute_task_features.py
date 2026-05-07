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
     ``{cache_dir}/{dataset}_task_feature.npy``.

This mirrors ``reference/AutoCTS_plusplus/exps/generate_task_feature.py`` —
the cached output is what :class:`models.comparator.task_feature
.TaskFeatureExtractor` reads at runtime.

Usage
-----
::

    python scripts/precompute_task_features.py \\
        --datasets PEMS03 PEMS04 PEMS07 PEMS08 ETTh2 ExchangeRate ETTh1 ETTm1 \\
        --data_dir data/datasets \\
        --cache_dir outputs/task_features \\
        --epochs 20 \\
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

from data.dataset import load_dataset
from models.comparator.task_feature import (
    DEFAULT_CACHE_DIR,
    TASK_FEATURE_REPR_DIM,
    TASK_FEATURE_SET_SIZE,
    TASK_FEATURE_SEQ_LEN,
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


def precompute_one(
    name: str, args: argparse.Namespace, device: torch.device,
) -> np.ndarray:
    """Train + encode + sample a task feature for one dataset.

    Returns:
        ``(n_set, seq_len, repr_dim)`` ``float32`` numpy array.
    """
    splits = load_dataset(name, args.data_dir)
    train_ds = splits["train"]
    input_dim = train_ds.n_channels
    logger.info(
        "[%s] train data shape=%s  task_type=%s  input_dim=%d",
        name, tuple(train_ds.data.shape), train_ds.task_type, input_dim,
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
            name, args.epochs,
        )
    else:
        pretrain_cfg["pretrain_iters"] = int(args.iters)
        logger.info(
            "[%s] training task-feature encoder for %d iters (iter mode)",
            name, args.iters,
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
        name, full_series.shape[1], args.padding, args.slide_batch,
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
            f"[{name}] encoder output_dim={D} != --repr_dim={args.repr_dim}",
        )

    bn = T // args.seq_len
    if bn == 0:
        raise RuntimeError(
            f"[{name}] series too short ({T}) for seq_len={args.seq_len}",
        )

    repr_windows = repr_full[: bn * args.seq_len].reshape(bn, args.seq_len, D)

    rng = np.random.default_rng(args.seed)
    if bn < args.n_set:
        logger.warning(
            "[%s] only %d windows of length %d available (requested %d) — "
            "sampling with replacement", name, bn, args.seq_len, args.n_set,
        )
        sample_idx = rng.choice(bn, size=args.n_set, replace=True)
    else:
        sample_idx = rng.choice(bn, size=args.n_set, replace=False)
    sample_idx.sort()
    sample_repr = repr_windows[sample_idx]                     # (n_set, seq_len, D)

    return sample_repr.astype(np.float32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.cache_dir, exist_ok=True)

    device = (
        torch.device(args.device) if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    logger.info("device=%s  cache_dir=%s", device, args.cache_dir)

    n_ok = 0
    n_skip = 0
    n_fail = 0
    for name in args.datasets:
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
        tmp = out_path + ".tmp"
        np.save(tmp, arr)
        os.replace(tmp, out_path)
        n_ok += 1
        logger.info("[%s] saved %s shape=%s dtype=%s",
                    name, out_path, arr.shape, arr.dtype)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info(
        "Done — saved=%d  skipped=%d  failed=%d  (out of %d datasets)",
        n_ok, n_skip, n_fail, len(args.datasets),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
