"""CLI entry point: generate seed data for T-CLSC pre-training.

Post-Bug-#003a: forecasting datasets default to an iter budget read from
``dataset_budgets`` in ``configs/default.yaml``; classification / anomaly
datasets fall back to epoch-based training.  CLI flags
``--pretrain_iters`` and ``--pretrain_epochs`` override the YAML values
**globally** (applied to every source dataset).

Usage
-----
python scripts/run_generate_seeds.py \\
    --data_dir  ZeroAutoCL/data/datasets \\
    --save_dir  ZeroAutoCL/outputs/seeds \\
    --config    ZeroAutoCL/configs/default.yaml \\
    --datasets  ETTm1 PEMS03 PEMS04 PEMS07 PEMS08 ExchangeRate PEMS-BAY \\
    --n_per_dataset 30 \\
    --batch_size 64 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from search.seed_generator import generate_seeds
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ZeroAutoCL seed data.")
    p.add_argument("--data_dir",        required=True,  help="Root data directory")
    p.add_argument("--save_dir",        required=True,  help="Output directory for seeds.json")
    p.add_argument("--config",          default=None,   help="Path to default.yaml (optional)")
    p.add_argument("--datasets",        nargs="+",
                   default=["ETTm1", "PEMS03", "PEMS04", "PEMS07", "PEMS08",
                            "ExchangeRate", "PEMS-BAY"],
                   help="Source dataset names")
    p.add_argument("--n_per_dataset",   type=int,       default=30)
    p.add_argument("--pretrain_epochs", type=int,       default=None,
                   help="Override per-dataset epoch budget for ALL datasets")
    p.add_argument("--pretrain_iters",  type=int,       default=None,
                   help="Override per-dataset iter budget for ALL datasets "
                        "(takes precedence over --pretrain_epochs)")
    p.add_argument("--batch_size",      type=int,       default=64)
    p.add_argument("--seed",            type=int,       default=42)
    p.add_argument("--device",          default=None,   help="'cpu' or 'cuda'")
    p.add_argument("--encoder_grid",    default=None,
                   help="(Plan B) Path to encoder_grid.json from Stage A. "
                        "When given, seed generation restricts the encoder "
                        "sub-space to the top-K rows of this file.")
    p.add_argument("--top_k_enc",       type=int, default=3,
                   help="(Plan B) How many encoders to keep from --encoder_grid.")
    return p.parse_args()


def _load_top_k_encoders(path: str, k: int) -> List[Dict[str, int]]:
    """Read encoder_grid.json and return its top-K encoder configs.

    The file is a list of records sorted by ``mean_perf`` desc, as written
    by :func:`search.encoder_grid_search.encoder_grid_search`.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"encoder_grid file is empty or malformed: {path}")
    return [r["encoder_config"] for r in records[:k]]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ── Defaults from YAML ────────────────────────────────────────────────
    sg_cfg: Dict = {}
    yaml_dataset_budgets: Dict[str, Dict[str, int]] = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sg_cfg               = cfg.get("seed_generation", {}) or {}
        yaml_dataset_budgets = cfg.get("dataset_budgets",  {}) or {}

    n_per_dataset   = args.n_per_dataset
    pretrain_epochs = sg_cfg.get("pretrain_epochs", 40)
    pretrain_lr     = sg_cfg.get("pretrain_lr",     1e-3)
    batch_size      = sg_cfg.get("batch_size",      args.batch_size)

    # ── Per-dataset budget resolution ─────────────────────────────────────
    # Priority for each source dataset:
    #   1. CLI --pretrain_iters / --pretrain_epochs (global override)
    #   2. configs/default.yaml :: dataset_budgets[name]
    #   3. Fallback: pretrain_epochs from seed_generation block (epoch-based)
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

    fixed_encoders: Optional[List[Dict[str, int]]] = None
    if args.encoder_grid:
        fixed_encoders = _load_top_k_encoders(args.encoder_grid, args.top_k_enc)
        logger.info(
            "[plan-B] using top-%d encoders from %s -> %s",
            args.top_k_enc, args.encoder_grid, fixed_encoders,
        )

    logger.info(
        "Generating seeds: datasets=%s  n_per=%d  lr=%.4f  bs=%d",
        args.datasets, n_per_dataset, pretrain_lr, batch_size,
    )
    logger.info("Per-dataset budgets: %s", dataset_budgets)

    seeds = generate_seeds(
        source_datasets=args.datasets,
        data_dir=args.data_dir,
        n_per_dataset=n_per_dataset,
        pretrain_epochs=int(pretrain_epochs),    # used only as fallback
        pretrain_lr=float(pretrain_lr),
        batch_size=int(batch_size),
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        dataset_budgets=dataset_budgets,
        fixed_encoders=fixed_encoders,
    )

    logger.info("Done — generated %d seed records.", len(seeds))


if __name__ == "__main__":
    main()
