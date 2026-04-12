"""Stage A of Plan B: encoder grid search under fixed GGS strategy.

For every encoder configuration in ``ENCODER_CHOICES`` (36 total) and every
source dataset listed in ``two_stage_search.encoder_grid.sources``, train
under :data:`GGS_STRATEGY` and write the aggregated ranking to
``encoder_grid.json``.

Usage
-----
python scripts/run_encoder_grid.py \\
    --data_dir  data/datasets \\
    --save_dir  outputs/full_etth1_two_stage \\
    --config    configs/default.yaml \\
    --seed      42
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from search.encoder_grid_search import encoder_grid_search
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan B Stage A: encoder grid search.")
    p.add_argument("--data_dir", required=True, help="Root data directory")
    p.add_argument("--save_dir", required=True, help="Where to write encoder_grid.json")
    p.add_argument("--config",   default="configs/default.yaml")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Override source datasets (defaults to YAML)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sg_cfg          = cfg.get("seed_generation", {}) or {}
    yaml_budgets    = cfg.get("dataset_budgets",  {}) or {}
    ts_cfg          = cfg.get("two_stage_search", {}) or {}
    grid_cfg        = ts_cfg.get("encoder_grid", {}) or {}

    sources: List[str] = args.datasets or list(grid_cfg.get("sources") or [])
    if not sources:
        raise ValueError("No source datasets given (CLI or YAML).")

    pretrain_lr = float(sg_cfg.get("pretrain_lr", 1e-3))
    batch_size  = int(sg_cfg.get("batch_size",   32))
    crop_len    = sg_cfg.get("crop_len")
    if crop_len is not None:
        crop_len = int(crop_len)

    # Per-dataset budgets exactly as resolved by run_generate_seeds.py.
    dataset_budgets: Dict[str, Dict[str, int]] = {
        ds: dict(yaml_budgets.get(ds, {"pretrain_iters": 600})) for ds in sources
    }

    device = torch.device(args.device) if args.device else None

    logger.info("[stage-A] sources=%s", sources)
    logger.info("[stage-A] lr=%.4f  batch=%d  crop_len=%s", pretrain_lr, batch_size, crop_len)
    logger.info("[stage-A] budgets=%s", dataset_budgets)

    encoder_grid_search(
        source_datasets=sources,
        data_dir=args.data_dir,
        pretrain_lr=pretrain_lr,
        batch_size=batch_size,
        dataset_budgets=dataset_budgets,
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        crop_len=crop_len,
    )


if __name__ == "__main__":
    main()
