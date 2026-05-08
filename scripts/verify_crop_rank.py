"""Lightweight rank-preservation check: crop_len=1024 vs crop_len=3000.

Goal
----
Seed-gen uses crop_len=1024 to speed up CL pretraining; phase 4 uses
crop_len=3000.  The implicit assumption is that the *ranking* of CL
strategies is preserved between the two crop sizes.  Whether this holds
empirically depends on whether short crops disproportionately penalise
strategies that need long context (e.g. temporal contrast with large
kernel, cross-scale contrast with deep pooling).

What this script does
---------------------
1. Defines 5 candidates spanning the short→long context spectrum:
     - GGS default            (instance only, kernel=5)              — short
     - phase4_best            (instance only, kernel=3, t=0.1)       — short
     - temporal_only          (instance + temporal, k=5, adj=True)   — long
     - cross_scale_only       (instance + cross_scale, k=5)          — long
     - both_long              (instance + temporal + cross_scale)    — longest

2. For each candidate × crop_len ∈ {1024, 3000}:
     - Build encoder (fixed GGS L10/H64/O320) + CLPipeline
     - contrastive_pretrain on the target's train split (iters=600)
     - eval_forecasting (phase 4 protocol, val_data passed)
     - Record mean MSE / MAE across canonical horizons

3. Compute Spearman correlation between the two ranking lists and write
   ``crop_rank_results.json`` next to ``--save_dir``.

Interpretation
--------------
   ρ ≥ 0.8  → short crop preserves rank well, current crop_len=1024 is fine
   0.6–0.8 → mild bias, consider crop_len=2048 to be safe
   ρ < 0.6 → strong bias toward short-context strategies, raise crop_len

Usage
-----
   python scripts/verify_crop_rank.py \
       --target   ETTh1 \
       --data_dir data/datasets \
       --save_dir outputs/crop_rank_check \
       --seed     42
"""

from __future__ import annotations

import argparse
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
from train.evaluate import eval_forecasting
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)

HORIZONS = [24, 48, 168, 336, 720]
CROPS = [1024, 3000]
ENCODER = {"n_layers": 10, "hidden_dim": 64, "output_dim": 320}


def _ggs_strategy() -> Dict:
    return {
        "augmentation": {
            "resize": 0.2, "rescale": 0.3, "jitter": 0.0,
            "point_mask": 0.2, "freq_mask": 0.0, "crop": 0.2, "order": 3,
        },
        "embedding_transform": {"jitter_p": 0.7, "mask_p": 0.1, "norm_type": "none"},
        "pair_construction": {
            "instance": True, "temporal": False, "cross_scale": False,
            "kernel_size": 5, "pool_op": "avg", "adj_neighbor": False,
        },
        "loss": {"type": "infonce", "sim_func": "distance", "temperature": 1.0},
    }


def _phase4_best_strategy() -> Dict:
    """Mirrors phase4_best.json from the latest run_array.sbatch results."""
    return {
        "augmentation": {
            "resize": 0.9, "rescale": 0.7, "jitter": 0.95,
            "point_mask": 0.0, "freq_mask": 0.0, "crop": 0.2, "order": 4,
        },
        "embedding_transform": {"jitter_p": 0.9, "mask_p": 0.95, "norm_type": "layer_norm"},
        "pair_construction": {
            "instance": True, "temporal": False, "cross_scale": False,
            "kernel_size": 3, "pool_op": "max", "adj_neighbor": False,
        },
        "loss": {"type": "infonce", "sim_func": "dot", "temperature": 0.1},
    }


def _temporal_only_strategy() -> Dict:
    s = _ggs_strategy()
    s["pair_construction"] = {
        "instance": True, "temporal": True, "cross_scale": False,
        "kernel_size": 5, "pool_op": "avg", "adj_neighbor": True,
    }
    return s


def _cross_scale_only_strategy() -> Dict:
    s = _ggs_strategy()
    s["pair_construction"] = {
        "instance": True, "temporal": False, "cross_scale": True,
        "kernel_size": 5, "pool_op": "avg", "adj_neighbor": False,
    }
    return s


def _both_long_strategy() -> Dict:
    s = _ggs_strategy()
    s["pair_construction"] = {
        "instance": True, "temporal": True, "cross_scale": True,
        "kernel_size": 5, "pool_op": "avg", "adj_neighbor": True,
    }
    return s


CANDIDATES: List[Dict] = [
    {"name": "ggs_default",      "strategy": _ggs_strategy(),            "context": "short"},
    {"name": "phase4_best",      "strategy": _phase4_best_strategy(),    "context": "short"},
    {"name": "temporal_only",    "strategy": _temporal_only_strategy(),  "context": "long"},
    {"name": "cross_scale_only", "strategy": _cross_scale_only_strategy(),"context": "long"},
    {"name": "both_long",        "strategy": _both_long_strategy(),      "context": "longest"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify rank preservation across crop_len.")
    p.add_argument("--target",   default="ETTh1",
                   help="Dataset to run the check on (default: ETTh1, the actual target).")
    p.add_argument("--data_dir", default="data/datasets")
    p.add_argument("--save_dir", default="outputs/crop_rank_check")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--iters",    type=int, default=600,
                   help="Pretrain iter budget (kept identical across crop sizes).")
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",   default=None)
    return p.parse_args()


def _train_and_eval_one(
    candidate: Dict,
    crop_len: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    """One pretrain + eval at a specific crop_len."""
    set_seed(args.seed)

    # Reload the dataset with the requested crop_len so the train sliding-window
    # length matches the configuration under test.
    splits = load_dataset(args.target, args.data_dir, window_len_override=crop_len)
    train_ds = splits["train"]
    val_ds   = splits.get("val")
    test_ds  = splits["test"]

    encoder  = DilatedCNNEncoder.from_config_dict(train_ds.n_channels, ENCODER).to(device)
    pipeline = CLPipeline(encoder, candidate["strategy"]).to(device)

    train_cfg = {
        "pretrain_iters": args.iters,
        "pretrain_lr":    args.lr,
        "batch_size":     args.batch_size,
    }

    t0 = time.time()
    contrastive_pretrain(
        encoder=encoder, cl_pipeline=pipeline,
        train_data=train_ds, config=train_cfg,
        device=device, task_type="forecasting",
    )
    train_secs = time.time() - t0

    encoder.eval()
    t1 = time.time()
    metrics = eval_forecasting(
        encoder=encoder,
        train_data=train_ds, test_data=test_ds,
        horizons=HORIZONS,
        batch_size=args.batch_size,
        device=device,
        val_data=val_ds,
    )
    eval_secs = time.time() - t1

    mean_mse = float(np.mean([v["mse"] for v in metrics.values()]))
    mean_mae = float(np.mean([v["mae"] for v in metrics.values()]))
    return {
        "mean_mse":   mean_mse,
        "mean_mae":   mean_mae,
        "per_horizon": {int(H): m for H, m in metrics.items()},
        "train_secs": train_secs,
        "eval_secs":  eval_secs,
    }


def _spearman(a: List[float], b: List[float]) -> float:
    """Spearman ρ between two equal-length score lists."""
    from scipy.stats import spearmanr
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("device=%s  target=%s  iters=%d  crops=%s",
                device, args.target, args.iters, CROPS)

    results: List[Dict] = []
    overall_t0 = time.time()
    n_total = len(CANDIDATES) * len(CROPS)
    done = 0

    for cand in CANDIDATES:
        for crop_len in CROPS:
            done += 1
            tag = f"{cand['name']}_crop{crop_len}"
            logger.info("=" * 60)
            logger.info("[%d/%d] %s  context=%s", done, n_total, tag, cand["context"])
            logger.info("=" * 60)

            try:
                r = _train_and_eval_one(cand, crop_len, args, device)
            except Exception as exc:
                logger.exception("[%s] failed: %s", tag, exc)
                results.append({
                    "tag": tag, "name": cand["name"], "context": cand["context"],
                    "crop_len": crop_len, "error": str(exc),
                })
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            record = {
                "tag":      tag,
                "name":     cand["name"],
                "context":  cand["context"],
                "crop_len": crop_len,
                **r,
            }
            results.append(record)
            logger.info(
                "[%s] mean_mse=%.4f mean_mae=%.4f train=%.0fs eval=%.0fs",
                tag, r["mean_mse"], r["mean_mae"], r["train_secs"], r["eval_secs"],
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    overall_secs = time.time() - overall_t0
    logger.info("[verify] all %d runs done in %.0fs (%.1f min)",
                n_total, overall_secs, overall_secs / 60)

    # ── Build rank lists per crop_len ───────────────────────────────────
    # Use higher-is-better keys for spearmanr; mse is lower-is-better, so
    # negate it.  We rank candidates only — drop any failed runs.
    by_crop: Dict[int, Dict[str, float]] = {c: {} for c in CROPS}
    for r in results:
        if "mean_mse" in r:
            by_crop[r["crop_len"]][r["name"]] = -r["mean_mse"]

    common = sorted(set.intersection(*(set(d) for d in by_crop.values())))
    if len(common) >= 2:
        a = [by_crop[CROPS[0]][n] for n in common]
        b = [by_crop[CROPS[1]][n] for n in common]
        rho_mse = _spearman(a, b)

        a_mae = [-next(r["mean_mae"] for r in results
                       if r["name"] == n and r["crop_len"] == CROPS[0]) for n in common]
        b_mae = [-next(r["mean_mae"] for r in results
                       if r["name"] == n and r["crop_len"] == CROPS[1]) for n in common]
        rho_mae = _spearman(a_mae, b_mae)
    else:
        rho_mse = rho_mae = float("nan")
        common = []

    summary = {
        "target":         args.target,
        "iters":          args.iters,
        "crops":          CROPS,
        "encoder":        ENCODER,
        "n_candidates":   len(CANDIDATES),
        "n_runs":         len(results),
        "wall_secs":      overall_secs,
        "spearman_mse":   rho_mse,
        "spearman_mae":   rho_mae,
        "ranked_by_mse":  {
            int(c): sorted(
                [(n, -s) for n, s in by_crop[c].items()],
                key=lambda x: x[1],
            )
            for c in CROPS
        },
        "results":        results,
    }

    out_path = os.path.join(args.save_dir, "crop_rank_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("[verify] wrote %s", out_path)

    # ── Verdict ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("CROP-LEN RANK PRESERVATION CHECK")
    print("=" * 60)
    print(f"target            : {args.target}")
    print(f"crops             : {CROPS}")
    print(f"candidates        : {len(common)} ({', '.join(common)})")
    print(f"Spearman ρ (MSE)  : {rho_mse:.3f}")
    print(f"Spearman ρ (MAE)  : {rho_mae:.3f}")
    print()
    for c in CROPS:
        print(f"Ranking @ crop={c} (best first):")
        for i, (name, mse) in enumerate(summary["ranked_by_mse"][c], 1):
            print(f"  {i}. {name:20s}  mean_mse={mse:.4f}")
    print()
    if not np.isnan(rho_mse):
        if rho_mse >= 0.8:
            verdict = "GOOD: short crop preserves rank — current crop_len=1024 is fine."
        elif rho_mse >= 0.6:
            verdict = "MILD BIAS: consider raising seed_generation.crop_len to 2048."
        else:
            verdict = "STRONG BIAS toward short-context strategies — raise crop_len ≥ 2048."
        print(f"VERDICT: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
