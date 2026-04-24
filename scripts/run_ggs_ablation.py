"""Ablation of the potential GGS issues identified in the code review.

Runs the full GGS pipeline on ETTh1 once per variant and prints a summary
table.  All variants use the same random seed so differences are protocol,
not noise.

Variants
--------
baseline        — unchanged GGS (crop=0.2, kernel=5 avg, sim=distance, EMA=on)
no_crop         — crop=0.0                       (tests view-alignment issue)
pool_k2_max     — kernel=2, pool=max             (tests hierarchical config)
sim_dot         — sim_func=dot                   (tests similarity function)
no_ema          — use_ema=False                  (tests EMA side-effect)
ts2vec_aligned  — all four fixes + temporal=True (keep lr=5e-5)
ts2vec_recipe   — ts2vec_aligned + lr=1e-3       (TS2Vec-default lr)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.dataset import load_dataset
from models.contrastive.cl_pipeline import CLPipeline
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.search_space.cl_strategy_space import DEFAULT_ENCODER, GGS_STRATEGY
from train.evaluate import eval_forecasting
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)
HORIZONS = [24, 48, 168, 336, 720]


def run_once(
    label: str,
    strategy: Dict,
    train_cfg: Dict,
    target: str,
    data_dir: str,
    seed: int,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    splits = load_dataset(target, data_dir)
    train_ds, val_ds, test_ds = splits["train"], splits.get("val"), splits["test"]
    input_dim = train_ds.n_channels
    encoder = DilatedCNNEncoder.from_config_dict(input_dim, DEFAULT_ENCODER).to(device)
    pipeline = CLPipeline(encoder, strategy).to(device)

    t0 = time.time()
    contrastive_pretrain(
        encoder=encoder, cl_pipeline=pipeline,
        train_data=train_ds, config=train_cfg,
        device=device, task_type="forecasting",
    )
    train_secs = time.time() - t0

    encoder.eval()
    metrics = eval_forecasting(
        encoder=encoder, train_data=train_ds, test_data=test_ds,
        horizons=HORIZONS, batch_size=train_cfg.get("batch_size", 8),
        device=device, val_data=val_ds,
    )
    mean_mse = float(np.mean([v["mse"] for v in metrics.values()]))
    mean_mae = float(np.mean([v["mae"] for v in metrics.values()]))
    logger.info(
        "[%s] mean_mse=%.4f  mean_mae=%.4f  train=%.1fs",
        label, mean_mse, mean_mae, train_secs,
    )
    return {
        "label": label,
        "per_horizon": {int(H): m for H, m in metrics.items()},
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "train_secs": train_secs,
        "strategy": strategy,
        "train_cfg": train_cfg,
    }


def build_variants() -> Dict[str, Dict]:
    base_strategy = copy.deepcopy(GGS_STRATEGY)
    base_cfg = {
        "pretrain_iters": 600,
        "pretrain_lr":    5e-5,
        "batch_size":     8,
        "optimizer":      "adamw",
        "grad_clip":      0.0,
    }

    def strat(**overrides) -> Dict:
        s = copy.deepcopy(base_strategy)
        for path, val in overrides.items():
            section, key = path.split(".")
            s[section][key] = val
        return s

    variants: Dict[str, Dict] = {
        "baseline":       {"strategy": copy.deepcopy(base_strategy), "cfg": dict(base_cfg)},
        "no_crop":        {"strategy": strat(**{"augmentation.crop": 0.0}),
                           "cfg": dict(base_cfg)},
        "pool_k2_max":    {"strategy": strat(**{"pair_construction.kernel_size": 2,
                                                "pair_construction.pool_op": "max"}),
                           "cfg": dict(base_cfg)},
        "sim_dot":        {"strategy": strat(**{"loss.sim_func": "dot"}),
                           "cfg": dict(base_cfg)},
        "no_ema":         {"strategy": copy.deepcopy(base_strategy),
                           "cfg": {**base_cfg, "use_ema": False}},
        "ts2vec_aligned": {"strategy": strat(**{
                                "augmentation.crop": 0.0,
                                "pair_construction.kernel_size": 2,
                                "pair_construction.pool_op": "max",
                                "pair_construction.temporal": True,
                                "loss.sim_func": "dot",
                            }),
                           "cfg": dict(base_cfg)},
        "ts2vec_recipe":  {"strategy": strat(**{
                                "augmentation.crop": 0.0,
                                "pair_construction.kernel_size": 2,
                                "pair_construction.pool_op": "max",
                                "pair_construction.temporal": True,
                                "loss.sim_func": "dot",
                            }),
                           "cfg": {**base_cfg, "pretrain_lr": 1e-3}},
        # Pure GGS strategy, but at the TS2Vec default learning rate.  With
        # the aligned-crop fix, does GGS tolerate the higher lr now?
        "ggs_lr1e3":      {"strategy": copy.deepcopy(base_strategy),
                           "cfg": {**base_cfg, "pretrain_lr": 1e-3}},
        # TS2Vec-style pair/loss but with the aligned crop kept at GGS's
        # 0.2 overlap (instead of crop=0).  This is the closest we get to
        # the original TS2Vec recipe: overlap crop + k=2 max + dot + both
        # instance and temporal contrast.
        "ts2vec_crop_aligned": {"strategy": strat(**{
                                "pair_construction.kernel_size": 2,
                                "pair_construction.pool_op": "max",
                                "pair_construction.temporal": True,
                                "loss.sim_func": "dot",
                            }),
                           "cfg": {**base_cfg, "pretrain_lr": 1e-3}},
    }
    return variants


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target",   default="ETTh1")
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument("--save_dir", default="outputs/ggs_ablation")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--tests",    nargs="+", default=None,
                   help="Subset of variant labels; default = all")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = build_variants()
    labels = args.tests if args.tests else list(variants.keys())

    results: List[Dict] = []
    for label in labels:
        if label not in variants:
            logger.warning("unknown variant %s, skipping", label)
            continue
        v = variants[label]
        try:
            r = run_once(label, v["strategy"], v["cfg"],
                         args.target, args.data_dir, args.seed, device)
            results.append(r)
        except Exception as exc:
            logger.exception("[%s] failed: %s", label, exc)
            results.append({"label": label, "error": str(exc)})

    out = os.path.join(args.save_dir, "ablation_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote %s", out)

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'label':<18}{'mean_MSE':>10}{'mean_MAE':>10}"
          f"{'H24':>8}{'H48':>8}{'H168':>8}{'H336':>8}{'H720':>8}{'train_s':>10}")
    print("=" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['label']:<18} ERROR: {r['error'][:60]}")
            continue
        ph = r["per_horizon"]
        def g(H: int) -> str:
            return f"{ph[H]['mse']:.4f}" if H in ph else "  -   "
        print(f"{r['label']:<18}"
              f"{r['mean_mse']:>10.4f}"
              f"{r['mean_mae']:>10.4f}"
              f"{g(24):>8}{g(48):>8}{g(168):>8}{g(336):>8}{g(720):>8}"
              f"{r['train_secs']:>10.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
