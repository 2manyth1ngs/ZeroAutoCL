"""Full GGS evaluation on ETTh1 with standard forecasting protocol.

Trains with GGS strategy, then evaluates using the prefix-encoding +
RidgeCV protocol at horizons [24, 48, 168, 336, 720].
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
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from models.search_space.cl_strategy_space import GGS_STRATEGY, DEFAULT_ENCODER
from train.pretrain import contrastive_pretrain
from train.evaluate import eval_forecasting
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/datasets")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sg = cfg.get("seed_generation", {}) or {}
    lr = float(sg.get("pretrain_lr", 1e-3))
    bs = int(sg.get("batch_size", 32))

    budget = (cfg.get("dataset_budgets", {}) or {}).get(args.dataset, {}) or {}
    iters = args.iters if args.iters is not None else int(budget.get("pretrain_iters", 600))

    # Use full window length (3000) for final evaluation, not seed-gen crop_len
    splits = load_dataset(args.dataset, args.data_dir, window_len_override=None)
    train_ds = splits["train"]
    val_ds = splits.get("val") or splits["test"]
    test_ds = splits.get("test", val_ds)

    input_dim = train_ds.n_channels
    encoder = DilatedCNNEncoder.from_config_dict(input_dim, DEFAULT_ENCODER).to(device)
    pipeline = CLPipeline(encoder, GGS_STRATEGY).to(device)

    logger.info("=" * 60)
    logger.info("GGS Full Evaluation on %s", args.dataset)
    logger.info("device=%s  iters=%d  lr=%.4f  batch=%d", device, iters, lr, bs)
    logger.info("encoder=%s", DEFAULT_ENCODER)
    logger.info("=" * 60)

    # ── Pretrain ──
    t0 = time.time()
    train_cfg = {
        "pretrain_iters": iters,
        "pretrain_lr": lr,
        "batch_size": bs,
    }
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config=train_cfg,
        device=device,
        task_type="forecasting",
    )
    pretrain_time = time.time() - t0

    # ── Evaluate on test set with standard protocol ──
    encoder.eval()
    horizons = [24, 48, 168, 336, 720]
    results = eval_forecasting(
        encoder, train_ds, test_ds,
        horizons=horizons, device=device,
    )

    logger.info("=" * 60)
    logger.info("Results: GGS on %s (pretrain %.0fs)", args.dataset, pretrain_time)
    logger.info("-" * 60)
    logger.info("%-10s %-12s %-12s", "Horizon", "MSE", "MAE")
    logger.info("-" * 60)

    mse_list, mae_list = [], []
    for h in horizons:
        if h in results:
            mse = results[h]["mse"]
            mae = results[h]["mae"]
            mse_list.append(mse)
            mae_list.append(mae)
            logger.info("%-10d %-12.6f %-12.6f", h, mse, mae)

    if mse_list:
        mean_mse = sum(mse_list) / len(mse_list)
        mean_mae = sum(mae_list) / len(mae_list)
        logger.info("-" * 60)
        logger.info("%-10s %-12.6f %-12.6f", "Mean", mean_mse, mean_mae)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
