"""GGS-on-ETTh1 end-to-end: train DEFAULT_ENCODER under GGS_STRATEGY, then run
the full TS2Vec-aligned ``eval_forecasting`` on the **test** split across all
5 horizons {24, 48, 168, 336, 720} — numbers directly comparable to
``reference/ts2vec/training/.../eval_res.pkl``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data.dataset import load_dataset
from models.contrastive.cl_pipeline import CLPipeline
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.search_space.cl_strategy_space import DEFAULT_ENCODER, GGS_STRATEGY
from train.evaluate import eval_forecasting
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/datasets")
    ap.add_argument("--config",   default="configs/default.yaml")
    ap.add_argument("--dataset",  default="ETTh1")
    ap.add_argument("--iters",    type=int, default=None,
                    help="Override pretrain_iters (default: from YAML).")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sg = cfg.get("seed_generation", {}) or {}
    lr = float(sg.get("pretrain_lr", 1e-3))
    bs = int(sg.get("batch_size", 32))
    crop_len = sg.get("crop_len")
    crop_len = int(crop_len) if crop_len is not None else None

    budget = (cfg.get("dataset_budgets", {}) or {}).get(args.dataset, {}) or {}
    iters  = args.iters if args.iters is not None else int(budget.get("pretrain_iters", 600))
    epochs = int(budget.get("pretrain_epochs", 40))

    logger.info("device=%s  dataset=%s  iters=%d  lr=%.4f  batch=%d  crop_len=%s",
                device, args.dataset, iters, lr, bs, crop_len)
    logger.info("encoder=%s", DEFAULT_ENCODER)
    logger.info("strategy=%s", GGS_STRATEGY)

    splits = load_dataset(args.dataset, args.data_dir, window_len_override=crop_len)
    train_ds = splits["train"]
    val_ds   = splits.get("val") or splits["test"]
    test_ds  = splits["test"]

    input_dim = train_ds.n_channels
    encoder  = DilatedCNNEncoder.from_config_dict(input_dim, DEFAULT_ENCODER).to(device)
    pipeline = CLPipeline(encoder, GGS_STRATEGY).to(device)

    # ── CL pretraining ─────────────────────────────────────────────────────
    t0 = time.time()
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config={
            "pretrain_epochs": epochs,
            "pretrain_iters":  iters,
            "pretrain_lr":     lr,
            "batch_size":      bs,
        },
        device=device,
        task_type=train_ds.task_type,
    )
    train_elapsed = time.time() - t0
    logger.info("pretraining done in %.1fs", train_elapsed)

    # ── Test-set forecasting across all 5 horizons ─────────────────────────
    encoder.eval()
    results = eval_forecasting(
        encoder, train_ds, test_ds,
        horizons=[24, 48, 168, 336, 720],
        batch_size=256,
        device=device,
        val_data=val_ds,
    )

    logger.info("=" * 70)
    logger.info("GGS on %s  (DEFAULT_ENCODER, TS2Vec-aligned eval)", args.dataset)
    logger.info("-" * 70)
    for H in [24, 48, 168, 336, 720]:
        m = results[H]
        logger.info("  H=%3d   test MSE=%.6f   test MAE=%.6f",
                    H, m["mse"], m["mae"])
    logger.info("-" * 70)
    logger.info("TS2Vec official reference (eval_res.pkl):")
    logger.info("  H= 24   MSE=0.042028")
    logger.info("  H= 48   MSE=0.062730")
    logger.info("  H=168   MSE=0.120825")
    logger.info("  H=336   MSE=0.140754")
    logger.info("  H=720   MSE=0.162356")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
