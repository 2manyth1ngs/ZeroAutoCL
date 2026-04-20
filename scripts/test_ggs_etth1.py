"""Quick GGS-on-ETTh1 smoke test after the pair_construction.py fix.

Trains DilatedCNN (DEFAULT_ENCODER) with GGS_STRATEGY on ETTh1 for the
configured iter budget, then runs the same ``_quick_eval`` used in seed
generation (Ridge@H=24, negative MSE; higher = better).
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
from models.search_space.cl_strategy_space import GGS_STRATEGY, DEFAULT_ENCODER
from search.seed_generator import _evaluate_candidate, _fmt_hms
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

    t0 = time.time()
    perf = _evaluate_candidate(
        DEFAULT_ENCODER, GGS_STRATEGY,
        train_ds, val_ds, train_ds.task_type,
        epochs, lr, bs, device,
        pretrain_iters=iters,
    )
    elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("GGS on %s : perf=%.6f  (higher = better; -MSE for forecasting)",
                args.dataset, perf)
    logger.info("elapsed  : %s", _fmt_hms(elapsed))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
