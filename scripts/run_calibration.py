"""CLI entry point: noisy/clean seed calibration test.

Runs Stage A+B of the calibration plan (see ``ZeroAutoCL/search/calibration.py``):

  1. Capture per-epoch val_score trajectories for N CL strategies × K seeds
     on ONE source dataset.  Encoder is held fixed so the only varying
     axis is the CL strategy.

  2. Analyze the trajectories and report Spearman ρ / pairwise
     concordance / top-K recall / mean CV at each candidate noisy budget.

Usage::

    python scripts/run_calibration.py \\
        --data_dir ZeroAutoCL/data/datasets \\
        --save_dir ZeroAutoCL/outputs/calibration_etth2 \\
        --source_dataset ETTh2 \\
        --n_strategies 16 --n_seeds 1 --max_epochs 20 \\
        --crop_len 1024 --batch_size 32

  --analyze_only (re-)runs metric analysis from a previously captured JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from search.calibration import (
    DEFAULT_BUDGETS,
    DEFAULT_ENCODER,
    analyze_trajectories,
    capture_trajectories,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZeroAutoCL noisy/clean seed calibration test.")
    p.add_argument("--data_dir",        required=True)
    p.add_argument("--save_dir",        required=True)
    p.add_argument("--source_dataset",  default="ETTh2",
                   help="Single source dataset to calibrate on (default ETTh2 — small + fast).")
    p.add_argument("--config",          default=None,
                   help="Optional path to default.yaml (for crop_len / lr / batch_size defaults).")
    # Capture knobs
    p.add_argument("--n_strategies",    type=int,   default=16)
    p.add_argument("--n_seeds",         type=int,   default=1,
                   help="K — independent runs per strategy (≥2 enables CV).")
    p.add_argument("--max_epochs",      type=int,   default=20,
                   help="Training epoch cap; also serves as the gold budget.")
    p.add_argument("--pretrain_lr",     type=float, default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--crop_len",        type=int,   default=None)
    p.add_argument("--horizons",        default=None,
                   help="Comma-separated forecasting horizons (default: TS2Vec canonical).")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default=None)
    # Encoder override (defaults to 10/64/320 — the project default).
    p.add_argument("--encoder_n_layers",   type=int, default=DEFAULT_ENCODER["n_layers"])
    p.add_argument("--encoder_hidden_dim", type=int, default=DEFAULT_ENCODER["hidden_dim"])
    p.add_argument("--encoder_output_dim", type=int, default=DEFAULT_ENCODER["output_dim"])
    # Analyze knobs
    p.add_argument("--budgets", default=None,
                   help=f"Comma-separated noisy-budget list to evaluate "
                        f"(default {','.join(map(str, DEFAULT_BUDGETS))}, clamped to ≤ max_epochs).")
    p.add_argument("--gap_threshold", type=float, default=0.02,
                   help="Pairwise-concordance gap filter (matches comparator's "
                        "valid_gap_threshold; default 0.02).")
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip capture; load trajectories.json from --save_dir and run analysis only.")
    return p.parse_args()


def _resolve_defaults_from_yaml(args: argparse.Namespace) -> None:
    """Fill capture-side defaults from configs/default.yaml when CLI omits them."""
    if not args.config:
        return
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sg = cfg.get("seed_generation", {}) or {}
    if args.pretrain_lr is None:
        args.pretrain_lr = float(sg.get("pretrain_lr", 1e-3))
    if args.batch_size is None:
        args.batch_size = int(sg.get("batch_size", 32))
    if args.crop_len is None and sg.get("crop_len") is not None:
        args.crop_len = int(sg["crop_len"])


def main() -> None:
    args = parse_args()
    _resolve_defaults_from_yaml(args)

    # Sensible final defaults if neither CLI nor YAML provided values.
    if args.pretrain_lr is None:
        args.pretrain_lr = 1e-3
    if args.batch_size is None:
        args.batch_size = 32

    horizons = None
    if args.horizons:
        horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    budgets = None
    if args.budgets:
        budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    device = torch.device(args.device) if args.device else None

    encoder = {
        "n_layers":   args.encoder_n_layers,
        "hidden_dim": args.encoder_hidden_dim,
        "output_dim": args.encoder_output_dim,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    traj_path = os.path.join(args.save_dir, "trajectories.json")

    if args.analyze_only:
        if not os.path.exists(traj_path):
            raise FileNotFoundError(
                f"--analyze_only requested but {traj_path} does not exist; "
                "drop the flag to capture first."
            )
        logger.info("[calibration-cli] analyze-only mode — loading %s", traj_path)
        with open(traj_path, encoding="utf-8") as f:
            traj = json.load(f)
    else:
        logger.info(
            "[calibration-cli] capturing trajectories: source=%s "
            "n_strategies=%d  n_seeds=%d  max_epochs=%d  encoder=%s  "
            "crop_len=%s  bs=%d  lr=%g",
            args.source_dataset, args.n_strategies, args.n_seeds, args.max_epochs,
            encoder, args.crop_len, args.batch_size, args.pretrain_lr,
        )
        traj = capture_trajectories(
            source_dataset=args.source_dataset,
            data_dir=args.data_dir,
            n_strategies=args.n_strategies,
            n_seeds=args.n_seeds,
            max_epochs=args.max_epochs,
            fixed_encoder=encoder,
            pretrain_lr=args.pretrain_lr,
            batch_size=args.batch_size,
            crop_len=args.crop_len,
            save_dir=args.save_dir,
            device=device,
            seed=args.seed,
            horizons=horizons,
        )

    summary = analyze_trajectories(
        traj,
        budgets=budgets,
        gap_threshold=args.gap_threshold,
        save_dir=args.save_dir,
    )
    logger.info(
        "[calibration-cli] done — wrote calibration_metrics.json + report + curve to %s",
        args.save_dir,
    )

    # Print a compact end-of-run summary so CI / shell users get a peek
    # without opening the report.
    pb = summary["per_budget"]
    print("\n==== Calibration summary (Mode B / best-of-N) ====")
    print("budget | spearman ρ | pairwise conc | top-5 recall | mean CV")
    for B in summary["budgets"]:
        r = pb.get(f"B{B}_modeB_bestofN", {})
        rho = r.get("spearman")
        conc = r.get("pairwise_concordance")
        t5 = r.get("topk_recall", {}).get("top5")
        cv = r.get("mean_cv_modeB")
        def f(x):
            if x is None: return "  —  "
            if isinstance(x, float) and (x != x): return " NaN "
            return f"{x:6.3f}"
        print(f"  {B:3d}  |  {f(rho)}   |   {f(conc)}    |   {f(t5)}    | {f(cv)}")


if __name__ == "__main__":
    main()
