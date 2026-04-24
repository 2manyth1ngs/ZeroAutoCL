"""Sample N random CL strategies — diagnose search-space signal quality.

For zero-shot search to be meaningful, the CL-strategy space must satisfy:

1. **Spread**: different strategies produce different MSE on a target task.
   If every strategy clusters at ~0.11, the comparator has no signal to
   learn.

2. **No systemic collapse bias**: low-MSE strategies should not all have
   extreme-low Ridge ``rcond`` (a sign of rank-collapsed embeddings that
   happen to fit Ridge well but aren't generalisable).  If they do, the
   search will systematically pick collapse solutions.

3. **Balance across sim / loss variants**: if ``sim=distance`` configs
   all crash at a shared lr, the space is effectively partitioned and the
   comparator can only rank within partitions.

This script samples N strategies with a fixed sample-seed, trains each
with a fixed train-seed (so strategy is the only variation), captures
Ridge ``rcond`` warnings, and summarises the MSE / rcond distributions.

Fixed training recipe for all strategies:
  iter=600, batch_size=8, optimizer=adamw, lr=1e-3, grad_clip=0, EMA=on.
Strategies that fail (OOM, NaN loss, etc.) are recorded with an ``error``
field but do not abort the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.dataset import load_dataset
from models.contrastive.cl_pipeline import CLPipeline
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.search_space.cl_strategy_space import DEFAULT_ENCODER, sample_cl_strategy
from train.evaluate import eval_forecasting
from train.pretrain import contrastive_pretrain
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)
HORIZONS = [24, 48, 168, 336, 720]


def _extract_rconds(warning_records) -> List[float]:
    """Pull rcond values out of sklearn's LinAlgWarning messages."""
    rconds: List[float] = []
    for w in warning_records:
        msg = str(w.message)
        if "rcond=" not in msg:
            continue
        try:
            rc_str = msg.split("rcond=", 1)[1].split(")", 1)[0]
            rconds.append(float(rc_str))
        except (ValueError, IndexError):
            pass
    return rconds


def run_single(
    strategy: Dict,
    target: str,
    data_dir: str,
    train_seed: int,
    device: torch.device,
    train_cfg: Dict,
) -> Dict:
    """Train and eval one strategy.  Returns a flat metrics dict."""
    set_seed(train_seed)
    splits = load_dataset(target, data_dir)
    train_ds, val_ds, test_ds = splits["train"], splits.get("val"), splits["test"]
    input_dim = train_ds.n_channels

    try:
        encoder = DilatedCNNEncoder.from_config_dict(input_dim, DEFAULT_ENCODER).to(device)
        pipeline = CLPipeline(encoder, strategy).to(device)
    except Exception as exc:
        return {"error": f"construct_failed: {exc}"}

    t0 = time.time()
    try:
        contrastive_pretrain(
            encoder=encoder, cl_pipeline=pipeline,
            train_data=train_ds, config=train_cfg,
            device=device, task_type="forecasting",
        )
    except Exception as exc:
        torch.cuda.empty_cache()
        return {"error": f"train_failed: {type(exc).__name__}: {exc}"}
    train_secs = time.time() - t0

    encoder.eval()
    with warnings.catch_warnings(record=True) as captured:
        # sklearn's Ridge emits its own LinAlgWarning class whose location
        # moved across versions (scipy.linalg / sklearn.exceptions).  We
        # dodge the import dance by catching every warning and filtering
        # by the ``rcond=`` substring in ``_extract_rconds``.
        warnings.simplefilter("always")
        try:
            metrics = eval_forecasting(
                encoder=encoder, train_data=train_ds, test_data=test_ds,
                horizons=HORIZONS, batch_size=train_cfg.get("batch_size", 8),
                device=device, val_data=val_ds,
            )
        except Exception as exc:
            return {"error": f"eval_failed: {type(exc).__name__}: {exc}"}

    rconds = _extract_rconds(captured)
    mean_mse = float(np.mean([v["mse"] for v in metrics.values()]))
    mean_mae = float(np.mean([v["mae"] for v in metrics.values()]))

    return {
        "per_horizon":      {int(H): {k: float(v) for k, v in m.items()} for H, m in metrics.items()},
        "mean_mse":         mean_mse,
        "mean_mae":         mean_mae,
        "min_rcond":        float(min(rconds)) if rconds else None,
        "max_rcond":        float(max(rconds)) if rconds else None,
        "n_rcond_warnings": len(rconds),
        "train_secs":       train_secs,
    }


def summarise(results: List[Dict]) -> None:
    """Print human-readable summary of the distribution."""
    valid = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print("\n" + "=" * 90)
    print(f"Random-strategy diagnostic   valid={len(valid)}  failed={len(failed)}")
    print("=" * 90)

    if not valid:
        print("No successful runs.")
        return

    mse   = np.array([r["mean_mse"] for r in valid])
    rc    = np.array([r["min_rcond"] for r in valid if r.get("min_rcond") is not None])

    print("\n  MSE   mean =  %.4f  std =  %.4f" % (mse.mean(), mse.std()))
    print("        min  =  %.4f  max =  %.4f  range = %.4f"
          % (mse.min(), mse.max(), mse.max() - mse.min()))
    if rc.size:
        print("  rcond mean =  %.2e  min =  %.2e  max =  %.2e"
              % (rc.mean(), rc.min(), rc.max()))

    # Correlation of log(rcond) vs MSE — sanity check for collapse bias.
    if rc.size >= 3:
        rc_full = np.array([r.get("min_rcond") for r in valid])
        mask = np.array([rc is not None for rc in rc_full])
        if mask.sum() >= 3:
            corr = np.corrcoef(np.log10(rc_full[mask].astype(float)), mse[mask])[0, 1]
            print("  corr(log rcond, MSE) = %+.3f   (negative ⇒ low rcond co-occurs with low MSE = collapse bias)" % corr)

    print("\nSorted by MSE (best first, showing 3 best + 3 worst):")
    header = (
        f"{'idx':>4} {'MSE':>7} {'MAE':>7} {'rcond':>10} "
        f"{'loss':>8} {'sim':>9} {'temp':>6} {'crop':>5} "
        f"{'k':>2} {'pool':>4} {'tmp':>5} {'xs':>5}"
    )
    print(header)
    print("-" * len(header))

    valid_sorted = sorted(valid, key=lambda r: r["mean_mse"])

    def _fmt(r: Dict) -> str:
        s = r["strategy"]
        lc = s["loss"]; pc = s["pair_construction"]; ac = s["augmentation"]
        rc_txt = ("%.1e" % r["min_rcond"]) if r.get("min_rcond") is not None else "n/a"
        return (
            f"{r['idx']:>4} {r['mean_mse']:>7.4f} {r['mean_mae']:>7.4f} "
            f"{rc_txt:>10} "
            f"{lc['type']:>8} {lc['sim_func']:>9} {lc['temperature']:>6.2f} "
            f"{ac['crop']:>5.2f} {pc['kernel_size']:>2} {pc['pool_op']:>4} "
            f"{str(pc['temporal'])[0]:>5} {str(pc['cross_scale'])[0]:>5}"
        )

    for r in valid_sorted[:3]:
        print(_fmt(r))
    if len(valid_sorted) > 6:
        print("  ...")
    for r in valid_sorted[-3:]:
        print(_fmt(r))

    if failed:
        print("\nFailures:")
        for r in failed:
            s = r.get("strategy", {})
            loss_txt = s.get("loss", {})
            print(f"  idx {r['idx']:>3}  sim={loss_txt.get('sim_func','?')} "
                  f"temp={loss_txt.get('temperature','?')}  → {r['error'][:80]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--target",       default="ETTh1")
    p.add_argument("--data_dir",     default="data/datasets")
    p.add_argument("--save_dir",     default="outputs/random_strategies")
    p.add_argument("--sample_seed",  type=int, default=42,
                   help="RNG for strategy sampling (decoupled from training seed)")
    p.add_argument("--train_seed",   type=int, default=42,
                   help="RNG for training each strategy (fixed across strategies)")
    p.add_argument("--pretrain_iters", type=int, default=600)
    p.add_argument("--pretrain_lr",  type=float, default=1e-3)
    p.add_argument("--batch_size",   type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s target=%s n=%d", device, args.target, args.n)

    # Strategy sampling uses its own RNG, separate from per-run training seed.
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    strategies = [sample_cl_strategy() for _ in range(args.n)]

    # Quiet the sklearn warning spam to avoid drowning the progress log —
    # we still capture them per-run via warnings.catch_warnings.
    logging.getLogger("sklearn").setLevel(logging.ERROR)

    train_cfg = {
        "pretrain_iters": args.pretrain_iters,
        "pretrain_lr":    args.pretrain_lr,
        "batch_size":     args.batch_size,
        "optimizer":      "adamw",
        "grad_clip":      0.0,
    }

    results: List[Dict] = []
    for i, strategy in enumerate(strategies):
        logger.info("── strategy %d/%d ──  sim=%s loss=%s temp=%.2f crop=%.2f k=%d pool=%s temporal=%s",
                    i + 1, args.n,
                    strategy["loss"]["sim_func"], strategy["loss"]["type"],
                    strategy["loss"]["temperature"], strategy["augmentation"]["crop"],
                    strategy["pair_construction"]["kernel_size"],
                    strategy["pair_construction"]["pool_op"],
                    strategy["pair_construction"]["temporal"])
        r = run_single(strategy, args.target, args.data_dir,
                       args.train_seed, device, train_cfg)
        r["idx"] = i
        r["strategy"] = strategy
        if "error" in r:
            logger.warning("  FAILED: %s", r["error"])
        else:
            rc_txt = ("%.2e" % r["min_rcond"]) if r.get("min_rcond") is not None else "n/a"
            logger.info("  mean_mse=%.4f  mean_mae=%.4f  min_rcond=%s  train=%.1fs",
                        r["mean_mse"], r["mean_mae"], rc_txt, r["train_secs"])
        results.append(r)

    out = os.path.join(args.save_dir, f"random_{args.n}_strategies.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("wrote %s", out)

    summarise(results)


if __name__ == "__main__":
    main()
