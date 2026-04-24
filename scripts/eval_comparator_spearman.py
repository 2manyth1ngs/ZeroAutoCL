"""Evaluate a trained T-CLSC comparator via Spearman rank correlation.

Feeds a pool of strategies (with known "ground truth" performance) through
the comparator pairwise, derives a predicted ranking from win counts, and
reports the Spearman correlation with the true performance ranking.

Typical test pool: ``outputs/random_strategies/random_20_strategies.json``
produced by ``scripts/sample_random_strategies.py`` — 20 fully evaluated
strategies on a single task, each with a ``mean_mse`` / ``mean_mae`` label.

Usage
-----
    python scripts/eval_comparator_spearman.py \\
        --comparator_path  outputs/comparator.pt \\
        --strategies_path  outputs/random_strategies/random_20_strategies.json \\
        --task             ETTh1 \\
        --data_dir         data/datasets \\
        --encoder_config   default    # or path to JSON with encoder_config keys
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import combinations
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

from data.dataset import load_dataset
from models.comparator.t_clsc import TCLSC
from models.comparator.task_feature import TASK_FEATURE_DIM, TaskFeatureExtractor
from models.search_space.cl_strategy_space import DEFAULT_ENCODER
from models.search_space.space_encoder import RAW_DIM
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate T-CLSC via Spearman.")
    p.add_argument("--comparator_path", required=True,
                   help="Path to a trained comparator .pt state_dict.")
    p.add_argument("--strategies_path", required=True,
                   help="Path to the strategies JSON (each record needs "
                        "'strategy', 'mean_mse' or 'mean_mae', and "
                        "optionally 'encoder_config').")
    p.add_argument("--task",           required=True,
                   help="Task ID whose task-feature is used for every pair.")
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--label_key",      default="mean_mae",
                   choices=["mean_mae", "mean_mse"],
                   help="Which field to use as ground truth (lower = better).")
    p.add_argument("--encoder_config", default="default",
                   help="'default' → use DEFAULT_ENCODER for every strategy "
                        "(matches the random-strategy sampler).  Otherwise "
                        "treat as a path to JSON with 'encoder_config'.")
    p.add_argument("--hidden_dim",     type=int, default=128)
    p.add_argument("--device",         default=None)
    return p.parse_args()


def _resolve_encoder_config(spec: str, record: Dict) -> Dict:
    """Return the encoder_config dict for *record*.

    Priority:
      1. If *spec* is a path to a JSON file, use its 'encoder_config'.
      2. If *spec* == 'default', use DEFAULT_ENCODER with 'binomial' mask.
      3. Otherwise fall back to record's own 'encoder_config' if present.
    """
    if spec != "default" and os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            return json.load(f)["encoder_config"]
    if "encoder_config" in record:
        enc = dict(record["encoder_config"])
        enc.setdefault("mask_mode", "binomial")
        return enc
    enc = dict(DEFAULT_ENCODER)
    enc.setdefault("mask_mode", "binomial")
    return enc


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Load test pool ─────────────────────────────────────────────────
    with open(args.strategies_path, encoding="utf-8") as f:
        records = json.load(f)
    records = [r for r in records if "error" not in r and args.label_key in r]
    logger.info("Loaded %d test strategies from %s", len(records), args.strategies_path)
    if len(records) < 3:
        logger.error("Need ≥ 3 valid strategies for a meaningful Spearman.")
        return

    strategies   = [r["strategy"] for r in records]
    enc_configs  = [_resolve_encoder_config(args.encoder_config, r) for r in records]
    # Ground truth: lower metric ⇒ better.
    true_metric  = np.array([float(r[args.label_key]) for r in records])
    n = len(strategies)

    # ── Task feature ───────────────────────────────────────────────────
    splits = load_dataset(args.task, args.data_dir)
    tfe = TaskFeatureExtractor(device=device)
    task_feat = tfe.extract(splits["train"], splits["train"].task_type)
    logger.info("Task feature shape: %s  (task=%s)", tuple(task_feat.shape), args.task)

    # ── Load comparator ────────────────────────────────────────────────
    comparator = TCLSC(
        candidate_dim=RAW_DIM, task_dim=TASK_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    state = torch.load(args.comparator_path, map_location=device)
    comparator.load_state_dict(state)
    comparator.eval()
    logger.info("Loaded comparator from %s", args.comparator_path)

    # ── Pairwise inference → win counts ───────────────────────────────
    wins     = np.zeros(n, dtype=np.int64)
    pair_iter = list(combinations(range(n), 2))
    logger.info("Running %d pairs through comparator …", len(pair_iter))

    # Batch per-call for throughput.
    BATCH = 64
    with torch.no_grad():
        for start in range(0, len(pair_iter), BATCH):
            batch = pair_iter[start : start + BATCH]
            enc_a = [enc_configs[i] for i, _ in batch]
            strat_a = [strategies[i]  for i, _ in batch]
            enc_b = [enc_configs[j] for _, j in batch]
            strat_b = [strategies[j]  for _, j in batch]
            probs = comparator.forward_batch(
                enc_a, strat_a, enc_b, strat_b, task_feat,
            ).cpu().numpy()                                     # (B,) P(A > B)
            for (i, j), p in zip(batch, probs):
                if p > 0.5:
                    wins[i] += 1
                else:
                    wins[j] += 1

    # ── Rank correlations ─────────────────────────────────────────────
    # Comparator rank: more wins ⇒ better.  Negate wins so "lower = better"
    # matches true_metric sign convention, then compute Spearman directly.
    rho,  p_rho  = spearmanr(-wins, true_metric)
    tau,  p_tau  = kendalltau(-wins, true_metric)

    print("\n" + "=" * 70)
    print(f"Spearman evaluation  task={args.task}  n_strategies={n}")
    print("=" * 70)
    print(f"  Spearman ρ  = {rho:+.4f}  (p={p_rho:.3g})")
    print(f"  Kendall  τ  = {tau:+.4f}  (p={p_tau:.3g})")
    print(f"  Total pairs = {len(pair_iter)}")
    print()

    # Sort both rankings side-by-side for visual inspection.
    order_true = np.argsort(true_metric)          # best-first in ground truth
    order_pred = np.argsort(-wins)                # most wins first
    print(f"{'rank':>4} {'idx(true)':>10} {'metric':>10} | {'idx(pred)':>10} {'wins':>6}")
    for rank, (it, ip) in enumerate(zip(order_true, order_pred)):
        same = "  *" if it == ip else ""
        print(f"{rank:>4} {it:>10d} {true_metric[it]:>10.4f} | "
              f"{ip:>10d} {wins[ip]:>6d}{same}")
    print("=" * 70)


if __name__ == "__main__":
    main()
