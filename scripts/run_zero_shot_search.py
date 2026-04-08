"""CLI entry point: zero-shot search on a target dataset.

Post-Bug-#003a: the top-K full-train budget for the target is read from
``dataset_budgets[target]`` in ``configs/default.yaml`` (iter-based for
forecasting, epoch-based otherwise).  CLI flags ``--pretrain_iters`` and
``--pretrain_epochs`` override the YAML value.

Usage
-----
python scripts/run_zero_shot_search.py \\
    --comparator_path ZeroAutoCL/outputs/comparator.pt \\
    --data_dir        ZeroAutoCL/data/datasets \\
    --target_dataset  ETTh1 \\
    --config          ZeroAutoCL/configs/default.yaml \\
    --output          ZeroAutoCL/outputs/best_ETTh1.json \\
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
from search.zero_shot_search import zero_shot_search
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot CL strategy search.")
    p.add_argument("--comparator_path", required=True, help="Path to comparator.pt")
    p.add_argument("--data_dir",        required=True, help="Root data directory")
    p.add_argument("--target_dataset",  required=True, help="Target dataset name")
    p.add_argument("--config",          default="configs/default.yaml")
    p.add_argument("--pretrain_epochs", type=int, default=None,
                   help="Override top-K full-train epoch budget")
    p.add_argument("--pretrain_iters",  type=int, default=None,
                   help="Override top-K full-train iter budget "
                        "(takes precedence over --pretrain_epochs)")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--device",          default=None)
    p.add_argument("--output",          default=None, help="Optional JSON output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    zs_cfg  = cfg.get("zero_shot", {})
    sg_cfg  = cfg.get("seed_generation", {})
    yaml_dataset_budgets = cfg.get("dataset_budgets", {}) or {}

    # ── Resolve top-K full-train budget for the target ──────────────────
    # Priority: CLI > yaml dataset_budgets[target] > seed_generation default
    target_budget = yaml_dataset_budgets.get(args.target_dataset, {}) or {}
    if args.pretrain_iters is not None and args.pretrain_iters > 0:
        topk_iters  = int(args.pretrain_iters)
        topk_epochs = 0
    elif args.pretrain_epochs is not None:
        topk_iters  = 0
        topk_epochs = int(args.pretrain_epochs)
    elif "pretrain_iters" in target_budget:
        topk_iters  = int(target_budget["pretrain_iters"])
        topk_epochs = 0
    elif "pretrain_epochs" in target_budget:
        topk_iters  = 0
        topk_epochs = int(target_budget["pretrain_epochs"])
    else:
        topk_iters  = 0
        topk_epochs = int(sg_cfg.get("pretrain_epochs", 40))

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Load comparator ─────────────────────────────────────────────
    hidden_dim = int(cfg.get("comparator", {}).get("hidden_dim", 128))
    comparator = TCLSC(
        candidate_dim=RAW_DIM,
        task_dim=TASK_FEATURE_DIM,
        hidden_dim=hidden_dim,
    )
    state = torch.load(args.comparator_path, map_location=device)
    comparator.load_state_dict(state)
    logger.info("Loaded comparator from %s", args.comparator_path)

    tfe = TaskFeatureExtractor(device=device)

    logger.info(
        "Top-K full-train budget for %s: iters=%d epochs=%d",
        args.target_dataset, topk_iters, topk_epochs,
    )

    best_enc, best_strat, best_perf = zero_shot_search(
        target_dataset=args.target_dataset,
        data_dir=args.data_dir,
        comparator=comparator,
        task_feature_extractor=tfe,
        n_candidates=int(zs_cfg.get("n_candidates", 300_000)),
        top_k=int(zs_cfg.get("top_k", 10)),
        tournament_rounds=int(zs_cfg.get("tournament_rounds", 15)),
        pretrain_epochs=topk_epochs,
        pretrain_iters=topk_iters,
        pretrain_lr=float(sg_cfg.get("pretrain_lr", 1e-3)),
        batch_size=int(sg_cfg.get("batch_size", 64)),
        device=device,
    )

    result = {
        "target_dataset": args.target_dataset,
        "best_encoder_config": best_enc,
        "best_strategy": best_strat,
        "performance": best_perf,
    }
    logger.info("Best result: %s", result)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved result to %s", args.output)


if __name__ == "__main__":
    main()
