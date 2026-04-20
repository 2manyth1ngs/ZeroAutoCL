"""Train DEFAULT_ENCODER (with binomial mask, B) under a **TS2Vec-like**
strategy and evaluate on ETTh1 across 5 horizons.

Strategy choices (vs GGS):
  * augmentation: all zero — TS2Vec's positive pairs come from the encoder's
    binomial mask, not value-level augmentations.
  * embedding_transform: disabled.
  * pair_construction: instance + temporal, hierarchical with kernel_size=2
    and pool_op='max' (TS2Vec's ``hierarchical_contrastive_loss`` recipe).
  * loss: InfoNCE with dot product, temperature=1.0.
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
from models.search_space.cl_strategy_space import DEFAULT_ENCODER
from train.evaluate import eval_forecasting
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


TS2VEC_LIKE_STRATEGY = {
    "augmentation": {
        "resize": 0.0, "rescale": 0.0, "jitter": 0.0,
        "point_mask": 0.0, "freq_mask": 0.0, "crop": 0.0, "order": 0,
    },
    "embedding_transform": {"jitter_p": 0.0, "mask_p": 0.0, "norm_type": "none"},
    "pair_construction": {
        "instance": True, "temporal": True, "cross_scale": False,
        "kernel_size": 2, "pool_op": "max", "adj_neighbor": False,
    },
    "loss": {"type": "infonce", "sim_func": "dot", "temperature": 1.0},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/datasets")
    ap.add_argument("--config",   default="configs/default.yaml")
    ap.add_argument("--dataset",  default="ETTh1")
    ap.add_argument("--iters",    type=int, default=None)
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
    logger.info("encoder=%s  (binomial mask_p=0.5)", DEFAULT_ENCODER)
    logger.info("strategy=TS2Vec-like  %s", TS2VEC_LIKE_STRATEGY)

    splits = load_dataset(args.dataset, args.data_dir, window_len_override=crop_len)
    train_ds = splits["train"]
    val_ds   = splits.get("val") or splits["test"]
    test_ds  = splits["test"]

    input_dim = train_ds.n_channels
    encoder  = DilatedCNNEncoder.from_config_dict(input_dim, DEFAULT_ENCODER).to(device)
    pipeline = CLPipeline(encoder, TS2VEC_LIKE_STRATEGY).to(device)

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
    logger.info("pretraining done in %.1fs", time.time() - t0)

    encoder.eval()
    results = eval_forecasting(
        encoder, train_ds, test_ds,
        horizons=[24, 48, 168, 336, 720],
        batch_size=256, device=device, val_data=val_ds,
    )

    logger.info("=" * 70)
    logger.info("TS2Vec-like strategy + binomial mask on %s", args.dataset)
    logger.info("-" * 70)
    for H in [24, 48, 168, 336, 720]:
        m = results[H]
        logger.info("  H=%3d   test MSE=%.6f   test MAE=%.6f",
                    H, m["mse"], m["mae"])
    logger.info("-" * 70)
    logger.info("TS2Vec official reference (eval_res.pkl):")
    logger.info("  H= 24 MSE=0.042028   H= 48 MSE=0.062730   H=168 MSE=0.120825")
    logger.info("  H=336 MSE=0.140754   H=720 MSE=0.162356")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
