"""Single-config GGS smoke test.

Skips the search entirely: takes ``DEFAULT_ENCODER`` + ``GGS_STRATEGY`` from
``models/search_space/cl_strategy_space.py`` (the AutoCLS Table 5 defaults),
trains one encoder on the target dataset, and evaluates on the standard
forecasting horizons in both normalised and raw space.

Example
-------
python scripts/run_ggs_test.py \\
    --target    ETTh1 \\
    --data_dir  data/datasets \\
    --save_dir  outputs/ggs_test_covariates \\
    --seed      42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.dataset import load_dataset
from models.contrastive.cl_pipeline import CLPipeline
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.search_space.cl_strategy_space import DEFAULT_ENCODER, GGS_STRATEGY
from train.evaluate import eval_forecasting
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)

HORIZONS = [24, 48, 168, 336, 720]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GGS (default) strategy once, no search.")
    p.add_argument("--target",         required=True, help="Target dataset name (e.g. ETTh1)")
    p.add_argument("--data_dir",       required=True, help="Root data directory")
    p.add_argument("--save_dir",       required=True, help="Output directory")
    # Defaults tuned on ETTh1 univariate (see outputs/ggs_J_lr5e-5_iter600):
    # batch=8, iters=600, AdamW w/o grad clip, lr=5e-5.  At lr=1e-3 the GGS
    # augmentation regime pushes the encoder to discard the raw-value
    # information Ridge needs; lr=5e-5 is gentle enough that 600 iters of
    # CL training help rather than hurt.  Average mean-MSE across seeds
    # {0, 42} is ~0.115, matching the TS2Vec paper number (0.116).
    p.add_argument("--pretrain_iters", type=int, default=600)
    p.add_argument("--pretrain_lr",    type=float, default=5e-5)
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--optimizer",      default="adamw", choices=["adam", "adamw"])
    p.add_argument("--grad_clip",      type=float, default=0.0,
                   help="Max grad norm; set 0 to disable")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--device",         default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("device=%s  target=%s  seed=%d", device, args.target, args.seed)
    logger.info("encoder=%s", DEFAULT_ENCODER)
    logger.info("strategy=%s", GGS_STRATEGY)

    # ── Load data ───────────────────────────────────────────────────────
    splits = load_dataset(args.target, args.data_dir)
    train_ds, val_ds, test_ds = splits["train"], splits.get("val"), splits["test"]
    logger.info(
        "train.shape=%s  val.shape=%s  test.shape=%s  n_cov=%d  scaler=%s",
        tuple(train_ds.data.shape),
        tuple(val_ds.data.shape) if val_ds is not None else None,
        tuple(test_ds.data.shape),
        getattr(train_ds, "n_covariate_cols", 0),
        type(getattr(train_ds, "scaler", None)).__name__,
    )

    # ── Build encoder & pipeline ────────────────────────────────────────
    input_dim = train_ds.n_channels
    encoder = DilatedCNNEncoder.from_config_dict(input_dim, DEFAULT_ENCODER).to(device)
    pipeline = CLPipeline(encoder, GGS_STRATEGY).to(device)

    # ── Pretrain ────────────────────────────────────────────────────────
    cfg = {
        "pretrain_iters": args.pretrain_iters,
        "pretrain_lr":    args.pretrain_lr,
        "batch_size":     args.batch_size,
        "optimizer":      args.optimizer,
        "grad_clip":      args.grad_clip,
    }
    t0 = time.time()
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config=cfg,
        device=device,
        task_type="forecasting",
    )
    train_secs = time.time() - t0
    logger.info("Pretrain done in %.1fs", train_secs)

    # ── Evaluate ────────────────────────────────────────────────────────
    encoder.eval()
    t1 = time.time()
    metrics = eval_forecasting(
        encoder=encoder,
        train_data=train_ds,
        test_data=test_ds,
        horizons=HORIZONS,
        batch_size=args.batch_size,
        device=device,
        val_data=val_ds,
    )
    eval_secs = time.time() - t1

    if not metrics:
        logger.error("No horizons evaluated (series too short?)")
        return

    # ── Aggregate ───────────────────────────────────────────────────────
    has_raw = any("mse_raw" in m for m in metrics.values())
    mean_mse = float(np.mean([v["mse"] for v in metrics.values()]))
    mean_mae = float(np.mean([v["mae"] for v in metrics.values()]))
    mean_mse_raw = (
        float(np.mean([v["mse_raw"] for v in metrics.values()])) if has_raw else None
    )
    mean_mae_raw = (
        float(np.mean([v["mae_raw"] for v in metrics.values()])) if has_raw else None
    )

    logger.info("=" * 60)
    logger.info("GGS smoke test — target=%s", args.target)
    logger.info("=" * 60)
    for H, m in metrics.items():
        extra = (
            f"  raw_mse={m['mse_raw']:.4f}  raw_mae={m['mae_raw']:.4f}"
            if "mse_raw" in m else ""
        )
        logger.info("H=%3d  mse=%.4f  mae=%.4f%s", H, m["mse"], m["mae"], extra)
    logger.info("-" * 60)
    logger.info("mean   mse=%.4f  mae=%.4f", mean_mse, mean_mae)
    if has_raw:
        logger.info("raw    mse=%.4f  mae=%.4f", mean_mse_raw, mean_mae_raw)
    logger.info("train_secs=%.1f  eval_secs=%.1f", train_secs, eval_secs)

    # ── Persist ─────────────────────────────────────────────────────────
    record = {
        "target":         args.target,
        "encoder_config": DEFAULT_ENCODER,
        "strategy":       GGS_STRATEGY,
        "train_cfg":      cfg,
        "horizons":       HORIZONS,
        "per_horizon":    {int(H): m for H, m in metrics.items()},
        "mean_mse":       mean_mse,
        "mean_mae":       mean_mae,
        "mean_mse_raw":   mean_mse_raw,
        "mean_mae_raw":   mean_mae_raw,
        "train_secs":     train_secs,
        "eval_secs":      eval_secs,
        "seed":           args.seed,
    }
    out_path = os.path.join(args.save_dir, "ggs_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
