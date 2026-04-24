"""CLI entry point: pre-train the T-CLSC comparator.

The comparator is task-agnostic w.r.t. Bug #003a — it never runs
contrastive pretraining itself, only learns a pairwise ranker over the
already-collected (config, perf) seed records.  This script therefore
needs no iter-budget plumbing.

Single-stage usage
------------------
    python scripts/run_pretrain_comparator.py \\
        --seeds_path  ZeroAutoCL/outputs/seeds/seeds.json \\
        --data_dir    ZeroAutoCL/data/datasets \\
        --save_path   ZeroAutoCL/outputs/comparator.pt \\
        --config      ZeroAutoCL/configs/default.yaml \\
        --seed 42

Two-stage usage (AutoCTS++-style noisy → clean curriculum)
----------------------------------------------------------
    python scripts/run_pretrain_comparator.py \\
        --seeds_path        ZeroAutoCL/outputs/seeds_noisy/seeds.json \\
        --clean_seeds_path  ZeroAutoCL/outputs/seeds_clean/seeds.json \\
        --data_dir          ZeroAutoCL/data/datasets \\
        --save_path         ZeroAutoCL/outputs/comparator.pt \\
        --config            ZeroAutoCL/configs/default.yaml

In two-stage mode the comparator is first trained on the noisy seeds
(broad coverage, higher label noise), then fine-tuned on the clean seeds
(fewer seeds, more reliable labels).  Fine-tune defaults to 0.1 × the
noisy lr and half the noisy epochs; override via a ``comparator_clean:``
block in the YAML config.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

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
    p.add_argument("--seeds_path", required=True, nargs="+",
                   help="One or more paths to seeds.json for the (noisy) "
                        "pretraining stage.  Multiple paths are concatenated "
                        "— useful for merging ``--mode noisy`` runs with "
                        "and without ``--randomise_init``.")
    p.add_argument("--clean_seeds_path", default=None, nargs="+",
                   help="Optional one-or-more paths to seeds.json from "
                        "clean-mode runs.  When given, a second fine-tune "
                        "stage is run on top of the noisy pretrained "
                        "comparator (AutoCTS++ two-stage).")
    p.add_argument("--data_dir",   required=True, help="Root data directory")
    p.add_argument("--save_path",  required=True, help="Output path for comparator.pt")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    comp_cfg       = cfg.get("comparator",       {}) or {}
    # Optional clean-stage override block.  If absent the fine-tune stage
    # will derive lr/epochs from comp_cfg (see _derive_clean_config).
    comp_clean_cfg = cfg.get("comparator_clean", None)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Load seeds (+ optional clean seeds) ───────────────────────────
    def _load_seed_paths(paths: List[str], tag: str) -> List[SeedRecord]:
        records: List[SeedRecord] = []
        for path in paths:
            with open(path) as f:
                chunk = [SeedRecord.from_dict(d) for d in json.load(f)]
            logger.info("  loaded %d %s records from %s", len(chunk), tag, path)
            records.extend(chunk)
        return records

    seeds = _load_seed_paths(args.seeds_path, "noisy")
    logger.info("Total: %d noisy seed records across %d file(s).",
                len(seeds), len(args.seeds_path))

    clean_seeds: Optional[List[SeedRecord]] = None
    if args.clean_seeds_path:
        clean_seeds = _load_seed_paths(args.clean_seeds_path, "clean")
        logger.info(
            "Total: %d clean seed records across %d file(s) "
            "(two-stage training enabled).",
            len(clean_seeds), len(args.clean_seeds_path),
        )

    # ── Extract task features for every task_id seen across both sets ─
    all_task_ids = {s.task_id for s in seeds}
    if clean_seeds:
        all_task_ids.update(s.task_id for s in clean_seeds)
    task_ids = sorted(all_task_ids)

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
        clean_seeds=clean_seeds,
        clean_config=comp_clean_cfg,
    )
    logger.info("Comparator saved to %s", args.save_path)


if __name__ == "__main__":
    main()
