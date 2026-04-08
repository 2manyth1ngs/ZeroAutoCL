"""CLI entry point: pre-train the T-CLSC comparator.

The comparator is task-agnostic w.r.t. Bug #003a — it never runs
contrastive pretraining itself, only learns a pairwise ranker over the
already-collected (config, perf) seed records.  This script therefore
needs no iter-budget plumbing.

Usage
-----
python scripts/run_pretrain_comparator.py \\
    --seeds_path  ZeroAutoCL/outputs/seeds/seeds.json \\
    --data_dir    ZeroAutoCL/data/datasets \\
    --save_path   ZeroAutoCL/outputs/comparator.pt \\
    --config      ZeroAutoCL/configs/default.yaml \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from models.comparator.t_clsc import TCLSC
from models.comparator.task_feature import TaskFeatureExtractor, TASK_FEATURE_DIM
from models.search_space.space_encoder import RAW_DIM
from data.dataset import load_dataset
from search.seed_generator import SeedRecord
from search.pretrain_comparator import pretrain_comparator
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train T-CLSC comparator.")
    p.add_argument("--seeds_path", required=True, help="Path to seeds.json")
    p.add_argument("--data_dir",   required=True, help="Root data directory")
    p.add_argument("--save_path",  required=True, help="Output path for comparator.pt")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    comp_cfg = cfg.get("comparator", {})

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Load seeds ─────────────────────────────────────────────────────
    with open(args.seeds_path) as f:
        seeds = [SeedRecord.from_dict(d) for d in json.load(f)]
    logger.info("Loaded %d seed records from %s", len(seeds), args.seeds_path)

    # ── Extract task features for each unique task_id ─────────────────
    task_ids = sorted({s.task_id for s in seeds})
    tfe = TaskFeatureExtractor(device=device)
    task_features = {}
    for tid in task_ids:
        logger.info("Extracting task features for: %s", tid)
        splits = load_dataset(tid, args.data_dir)
        feat = tfe.extract(splits["train"], splits["train"].task_type)
        task_features[tid] = feat
    logger.info("Task features ready for %d datasets.", len(task_features))

    # ── Build and train comparator ─────────────────────────────────────
    comparator = TCLSC(
        candidate_dim=RAW_DIM,
        task_dim=TASK_FEATURE_DIM,
        hidden_dim=int(comp_cfg.get("hidden_dim", 128)),
    )
    comparator = pretrain_comparator(
        seeds=seeds,
        task_features=task_features,
        config=comp_cfg,
        comparator=comparator,
        save_path=args.save_path,
        device=device,
    )
    logger.info("Comparator saved to %s", args.save_path)


if __name__ == "__main__":
    main()
