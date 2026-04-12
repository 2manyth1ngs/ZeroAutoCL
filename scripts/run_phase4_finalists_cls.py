"""Phase 4 of Plan B (classification): retrain Cartesian finalists on the target.

Reads ``finalists.json`` (produced by ``run_zero_shot_search.py --encoder_grid``),
and for each ``(encoder_config, strategy)`` pair:

  1. builds a fresh encoder + CL pipeline
  2. runs ``contrastive_pretrain`` on the target's train split using epoch-based
     budget with val-best checkpointing (classification-appropriate)
  3. evaluates with ``eval_classification`` (SVM on pooled embeddings)
  4. records accuracy and macro-F1

Writes ``phase4_results.json`` (full table) and ``phase4_best.json`` (single
best record by accuracy) into ``--save_dir``.

Usage
-----
python scripts/run_phase4_finalists_cls.py \\
    --finalists outputs/full_har_two_stage/finalists.json \\
    --target    HAR \\
    --data_dir  data/datasets \\
    --config    configs/default.yaml \\
    --save_dir  outputs/full_har_two_stage \\
    --seed      42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from data.dataset import load_dataset
from models.contrastive.cl_pipeline import CLPipeline
from models.encoder.dilated_cnn import DilatedCNNEncoder
from train.evaluate import eval_classification
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plan B Phase 4 (classification): Cartesian retrain."
    )
    p.add_argument("--finalists", required=True, help="Path to finalists.json")
    p.add_argument("--target", required=True, help="Target dataset name")
    p.add_argument("--data_dir", required=True, help="Root data directory")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--save_dir", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _retrain_one(
    encoder_config: Dict[str, int],
    strategy: Dict,
    train_ds,
    val_ds,
    test_ds,
    cfg: Dict,
    device: torch.device,
) -> Dict:
    """Train one finalist and evaluate on the target test split."""
    encoder = DilatedCNNEncoder.from_config_dict(
        train_ds.n_channels, encoder_config
    ).to(device)
    pipeline = CLPipeline(encoder, strategy).to(device)

    t0 = time.time()
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config=cfg,
        device=device,
        val_data=val_ds,
        task_type="classification",
    )
    train_secs = time.time() - t0

    encoder.eval()
    t1 = time.time()
    test_metrics = eval_classification(
        encoder=encoder,
        train_data=train_ds,
        test_data=test_ds,
        device=device,
    )
    val_metrics = eval_classification(
        encoder=encoder,
        train_data=train_ds,
        test_data=val_ds,
        device=device,
    )
    eval_secs = time.time() - t1

    return {
        "test_acc": test_metrics["acc"],
        "test_f1": test_metrics["f1"],
        "val_acc": val_metrics["acc"],
        "val_f1": val_metrics["f1"],
        "train_secs": train_secs,
        "eval_secs": eval_secs,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sg_cfg = cfg.get("seed_generation", {}) or {}
    yaml_budgets = cfg.get("dataset_budgets", {}) or {}

    # Resolve training budget for the target dataset.
    target_budget = yaml_budgets.get(args.target, {}) or {}
    pretrain_epochs = int(
        target_budget.get(
            "pretrain_epochs", sg_cfg.get("pretrain_epochs", 40)
        )
    )
    pretrain_lr = float(sg_cfg.get("pretrain_lr", 1e-3))
    batch_size = int(sg_cfg.get("batch_size", 32))

    train_cfg = {
        "pretrain_epochs": pretrain_epochs,
        "pretrain_lr": pretrain_lr,
        "batch_size": batch_size,
        "eval_every": 10,
        "val_best": True,
    }

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("device=%s  train_cfg=%s", device, train_cfg)

    with open(args.finalists, encoding="utf-8") as f:
        finalists: List[Dict] = json.load(f)
    if not finalists:
        raise ValueError(f"finalists list is empty: {args.finalists}")
    logger.info("Loaded %d finalists from %s", len(finalists), args.finalists)

    splits = load_dataset(args.target, args.data_dir)
    train_ds = splits["train"]
    val_ds = splits.get("val")
    test_ds = splits["test"]
    logger.info(
        "target=%s  train.shape=%s  val.shape=%s  test.shape=%s",
        args.target,
        tuple(train_ds.data.shape),
        tuple(val_ds.data.shape) if val_ds is not None else None,
        tuple(test_ds.data.shape),
    )

    results: List[Dict] = []
    overall_t0 = time.time()
    for i, finalist in enumerate(finalists):
        enc_cfg = finalist["encoder_config"]
        strat = finalist["strategy"]
        rank = finalist.get("rank", -1)
        tag = (
            f"L{enc_cfg['n_layers']}_H{enc_cfg['hidden_dim']}"
            f"_O{enc_cfg['output_dim']}_r{rank}"
        )
        logger.info("=" * 60)
        logger.info("[phase4 %d/%d] %s", i + 1, len(finalists), tag)
        logger.info("=" * 60)

        try:
            r = _retrain_one(
                enc_cfg, strat, train_ds, val_ds, test_ds, train_cfg, device
            )
        except Exception as exc:
            logger.exception("[phase4] finalist %s failed: %s", tag, exc)
            results.append({
                "tag": tag, "encoder_config": enc_cfg, "strategy": strat,
                "rank": rank, "error": str(exc),
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        record = {
            "tag": tag,
            "encoder_config": enc_cfg,
            "strategy": strat,
            "rank": rank,
            **r,
        }
        results.append(record)
        logger.info(
            "[phase4 %d/%d] %s  test_acc=%.4f  test_f1=%.4f  "
            "val_acc=%.4f  train=%.0fs  eval=%.0fs",
            i + 1, len(finalists), tag,
            r["test_acc"], r["test_f1"], r["val_acc"],
            r["train_secs"], r["eval_secs"],
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall_secs = time.time() - overall_t0
    logger.info("[phase4] all %d finalists done in %.0fs", len(results), overall_secs)

    out_path = os.path.join(args.save_dir, "phase4_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": args.target,
            "task_type": "classification",
            "train_cfg": train_cfg,
            "wall_secs": overall_secs,
            "results": results,
        }, f, indent=2)
    logger.info("[phase4] wrote %s", out_path)

    # Pick best by test accuracy (higher = better).
    ok = [r for r in results if "test_acc" in r]
    if ok:
        best = max(ok, key=lambda r: r["test_acc"])
        best_path = os.path.join(args.save_dir, "phase4_best.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
        logger.info(
            "[phase4] BEST: %s  test_acc=%.4f  test_f1=%.4f",
            best["tag"], best["test_acc"], best["test_f1"],
        )
        logger.info("[phase4] wrote %s", best_path)
    else:
        logger.error("[phase4] no successful finalists — nothing to pick")

    # Print summary table.
    print("\n" + "=" * 95)
    print(
        f"{'Tag':<28s} {'Test Acc':>9s} {'Test F1':>9s} "
        f"{'Val Acc':>9s} {'Val F1':>9s} {'Train':>7s} {'Eval':>6s}"
    )
    print("-" * 95)
    for r in sorted(ok, key=lambda x: -x["test_acc"]):
        print(
            f"{r['tag']:<28s} {r['test_acc']:>9.4f} {r['test_f1']:>9.4f} "
            f"{r['val_acc']:>9.4f} {r['val_f1']:>9.4f} "
            f"{r['train_secs']:>6.0f}s {r['eval_secs']:>5.0f}s"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
