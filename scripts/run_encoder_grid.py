"""CLI entry point: Plan B Stage A — encoder grid search.

For each source dataset, trains every encoder configuration in
``ENCODER_CHOICES`` (36 total) under the GGS default CL strategy and
aggregates the resulting validation performance into a global ranking.

Usage
-----
python scripts/run_encoder_grid.py \\
    --data_dir  ZeroAutoCL/data/datasets \\
    --save_dir  ZeroAutoCL/outputs/full_etth1_two_stage/encoder_grid \\
    --config    ZeroAutoCL/configs/default.yaml \\
    --datasets  ETTm1 PEMS03 PEMS04 PEMS07 PEMS08 ExchangeRate PEMS-BAY \\
    --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from search.encoder_grid_search import encoder_grid_search
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ZeroAutoCL Plan B — Stage A encoder grid search."
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument("--save_dir", required=True,
                   help="Directory to write encoder_grid.json into.")
    p.add_argument("--config",   default=None)
    p.add_argument("--datasets", nargs="+",
                   default=["ETTm1", "PEMS03", "PEMS04", "PEMS07", "PEMS08",
                            "ExchangeRate", "PEMS-BAY"],
                   help="Source dataset names (target dataset MUST NOT be here).")
    p.add_argument("--pretrain_iters",  type=int, default=None,
                   help="Override per-dataset iter budget for ALL datasets.")
    p.add_argument("--pretrain_epochs", type=int, default=None)
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--device",          default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    sg_cfg: Dict = {}
    yaml_dataset_budgets: Dict[str, Dict[str, int]] = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sg_cfg               = cfg.get("seed_generation", {}) or {}
        yaml_dataset_budgets = cfg.get("dataset_budgets",  {}) or {}

    pretrain_epochs = sg_cfg.get("pretrain_epochs", 40)
    pretrain_lr     = sg_cfg.get("pretrain_lr",     1e-3)
    batch_size      = sg_cfg.get("batch_size",      args.batch_size)

    dataset_budgets: Dict[str, Dict[str, int]] = {}
    for ds in args.datasets:
        if args.pretrain_iters is not None and args.pretrain_iters > 0:
            dataset_budgets[ds] = {"pretrain_iters": int(args.pretrain_iters)}
        elif args.pretrain_epochs is not None:
            dataset_budgets[ds] = {"pretrain_epochs": int(args.pretrain_epochs)}
        elif ds in yaml_dataset_budgets:
            dataset_budgets[ds] = dict(yaml_dataset_budgets[ds])
        else:
            dataset_budgets[ds] = {"pretrain_epochs": int(pretrain_epochs)}

    device = torch.device(args.device) if args.device else None

    logger.info(
        "[stage-A CLI] datasets=%s lr=%.4f bs=%d",
        args.datasets, pretrain_lr, batch_size,
    )
    logger.info("[stage-A CLI] per-dataset budgets: %s", dataset_budgets)

    records = encoder_grid_search(
        source_datasets=args.datasets,
        data_dir=args.data_dir,
        pretrain_lr=float(pretrain_lr),
        batch_size=int(batch_size),
        dataset_budgets=dataset_budgets,
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
    )
    logger.info("[stage-A CLI] grid search done — %d records", len(records))


if __name__ == "__main__":
    main()
