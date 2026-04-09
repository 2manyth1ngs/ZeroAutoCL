"""A/B test for the §10 training-protocol fixes (CLAUDE_ADV.md §10).

Two runs of the same (encoder, GGS_STRATEGY, seed=42) on ETTh1 univariate
forecasting:

  baseline : crop_len=201,  batch_size=32, legacy per-split eval (no val)
  fixed    : crop_len=3000, batch_size=32, prefix-encoding eval (with val)

Both runs use ``pretrain_iters=3000`` (§10.2 long-budget retrain) so the
differences are the §10 protocol changes (§10.1 crop length + §10.3
prefix-encoding eval) compounded with the longer training schedule.  We
compare per-horizon MSE/MAE plus the wall-clock cost and write everything
to ``outputs/test_protocol_ab/``.

Run from the ZeroAutoCL repo root::

    python scripts/test_protocol_ab.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import data.dataset as dataset_mod
from data.dataset import load_dataset
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from models.search_space.cl_strategy_space import GGS_STRATEGY
from train.pretrain import contrastive_pretrain
from train.evaluate import compute_forecasting_metrics
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)

ENCODER_CONFIG = {"n_layers": 10, "hidden_dim": 64, "output_dim": 320}
HORIZONS = [24, 48, 168, 336, 720]
PRETRAIN_ITERS = 3000
PRETRAIN_LR = 1e-3
SEED = 42
TARGET = "ETTh1"
DATA_DIR = "data/datasets"
OUT_DIR = "outputs/test_protocol_ab"


# ---------------------------------------------------------------------------
# Legacy eval — replicates the pre-§10.3 per-split forward
# ---------------------------------------------------------------------------

_RIDGE_ALPHAS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def legacy_eval_forecasting(encoder, train_data, test_data, horizons, device):
    """Old eval_forecasting: encode train and test independently."""
    from sklearn.linear_model import RidgeCV

    encoder = encoder.to(device).eval()
    with torch.no_grad():
        h_tr = encoder(train_data.data.to(device)).cpu().numpy()[0]
        h_te = encoder(test_data.data.to(device)).cpu().numpy()[0]
    x_tr = train_data.data[0].numpy()
    x_te = test_data.data[0].numpy()

    results: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        if h_tr.shape[0] <= H or h_te.shape[0] <= H:
            continue
        reg = RidgeCV(alphas=_RIDGE_ALPHAS)
        reg.fit(h_tr[:-H], x_tr[H:])
        y_pred = reg.predict(h_te[:-H])
        results[H] = compute_forecasting_metrics(x_te[H:], y_pred)
    return results


# ---------------------------------------------------------------------------
# Prefix eval — replicates the post-§10.3 prefix encoding
# ---------------------------------------------------------------------------

def prefix_eval_forecasting(encoder, train_data, val_data, test_data, horizons, device):
    """New eval_forecasting: concat train+val+test, single forward."""
    from sklearn.linear_model import RidgeCV

    encoder = encoder.to(device).eval()
    x_tr_t = train_data.data
    x_te_t = test_data.data
    T_tr = x_tr_t.shape[1]
    T_te = x_te_t.shape[1]
    if val_data is not None:
        x_va_t = val_data.data
        T_va = x_va_t.shape[1]
        full = torch.cat([x_tr_t, x_va_t, x_te_t], dim=1).to(device)
    else:
        T_va = 0
        full = torch.cat([x_tr_t, x_te_t], dim=1).to(device)

    with torch.no_grad():
        h_full = encoder(full).cpu().numpy()[0]

    h_tr = h_full[:T_tr]
    h_te = h_full[T_tr + T_va : T_tr + T_va + T_te]
    x_tr = x_tr_t[0].numpy()
    x_te = x_te_t[0].numpy()

    results: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        if h_tr.shape[0] <= H or h_te.shape[0] <= H:
            continue
        reg = RidgeCV(alphas=_RIDGE_ALPHAS)
        reg.fit(h_tr[:-H], x_tr[H:])
        y_pred = reg.predict(h_te[:-H])
        results[H] = compute_forecasting_metrics(x_te[H:], y_pred)
    return results


# ---------------------------------------------------------------------------
# A single A/B group runner
# ---------------------------------------------------------------------------

def run_group(
    label: str,
    window_len: int,
    batch_size: int,
    use_prefix_eval: bool,
    device: torch.device,
) -> Dict:
    """Train one (encoder, GGS) on ETTh1 under the given protocol settings."""
    logger.info("=" * 60)
    logger.info("[group=%s] window_len=%d  batch=%d  prefix_eval=%s",
                label, window_len, batch_size, use_prefix_eval)
    logger.info("=" * 60)

    # Monkey-patch the global crop length BEFORE load_dataset.
    dataset_mod._FORECAST_WINDOW_LEN = window_len

    set_seed(SEED)
    splits = load_dataset(TARGET, DATA_DIR)
    train_ds = splits["train"]
    val_ds   = splits.get("val")
    test_ds  = splits["test"]
    logger.info(
        "[group=%s] train.data.shape=%s  val.data.shape=%s  test.data.shape=%s",
        label, tuple(train_ds.data.shape),
        tuple(val_ds.data.shape) if val_ds is not None else None,
        tuple(test_ds.data.shape),
    )

    encoder  = DilatedCNNEncoder.from_config_dict(train_ds.n_channels, ENCODER_CONFIG).to(device)
    pipeline = CLPipeline(encoder, GGS_STRATEGY).to(device)

    cfg = {
        "pretrain_iters": PRETRAIN_ITERS,
        "pretrain_lr":    PRETRAIN_LR,
        "batch_size":     batch_size,
    }

    t0 = time.time()
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config=cfg,
        device=device,
        task_type="forecasting",
    )
    train_secs = time.time() - t0

    encoder.eval()
    t1 = time.time()
    if use_prefix_eval:
        metrics = prefix_eval_forecasting(encoder, train_ds, val_ds, test_ds, HORIZONS, device)
    else:
        metrics = legacy_eval_forecasting(encoder, train_ds, test_ds, HORIZONS, device)
    eval_secs = time.time() - t1

    if not metrics:
        raise RuntimeError(f"[group={label}] no horizons evaluated")

    mean_mse = float(np.mean([v["mse"] for v in metrics.values()]))
    mean_mae = float(np.mean([v["mae"] for v in metrics.values()]))

    logger.info("[group=%s] train_secs=%.1f  eval_secs=%.1f", label, train_secs, eval_secs)
    logger.info("[group=%s] mean_mse=%.4f  mean_mae=%.4f", label, mean_mse, mean_mae)
    for H, m in metrics.items():
        logger.info("  H=%d  mse=%.4f  mae=%.4f", H, m["mse"], m["mae"])

    return {
        "label": label,
        "window_len": window_len,
        "batch_size": batch_size,
        "use_prefix_eval": use_prefix_eval,
        "pretrain_iters": PRETRAIN_ITERS,
        "encoder_config": ENCODER_CONFIG,
        "test_mean_mse": mean_mse,
        "test_mean_mae": mean_mae,
        "test_per_horizon": {int(H): m for H, m in metrics.items()},
        "wall_time_train_secs": train_secs,
        "wall_time_eval_secs": eval_secs,
    }


# ---------------------------------------------------------------------------
# Diff writer
# ---------------------------------------------------------------------------

def write_diff(baseline: Dict, fixed: Dict, path: str) -> None:
    lines: List[str] = []
    push = lines.append

    push("ZeroAutoCL — §10 Protocol Fix A/B Test")
    push("=" * 70)
    push(f"target           : {TARGET}")
    push(f"encoder          : {ENCODER_CONFIG}")
    push(f"strategy         : GGS_STRATEGY (AutoCLS Table 5 default)")
    push(f"pretrain_iters   : {PRETRAIN_ITERS}")
    push(f"seed             : {SEED}")
    push("")
    push(f"{'group':<10}{'crop':<8}{'batch':<8}{'eval':<14}"
         f"{'mean_mse':<12}{'mean_mae':<12}{'train_s':<10}{'eval_s':<10}")
    push("-" * 84)
    for r in (baseline, fixed):
        eval_mode = "prefix" if r["use_prefix_eval"] else "legacy"
        push(f"{r['label']:<10}{r['window_len']:<8}{r['batch_size']:<8}{eval_mode:<14}"
             f"{r['test_mean_mse']:<12.4f}{r['test_mean_mae']:<12.4f}"
             f"{r['wall_time_train_secs']:<10.0f}{r['wall_time_eval_secs']:<10.0f}")
    push("")
    d_mse = fixed["test_mean_mse"] - baseline["test_mean_mse"]
    d_mae = fixed["test_mean_mae"] - baseline["test_mean_mae"]
    push(f"delta mean_mse   : {d_mse:+.4f}  ({100*d_mse/baseline['test_mean_mse']:+.1f}%)")
    push(f"delta mean_mae   : {d_mae:+.4f}  ({100*d_mae/baseline['test_mean_mae']:+.1f}%)")
    push("")

    push("Per-horizon detail:")
    push(f"{'H':<8}{'baseline_mse':<16}{'fixed_mse':<16}{'delta':<14}{'baseline_mae':<16}{'fixed_mae':<16}{'delta':<14}")
    push("-" * 100)
    for H in HORIZONS:
        b = baseline["test_per_horizon"].get(H)
        f = fixed   ["test_per_horizon"].get(H)
        if b is None or f is None:
            continue
        dm = f["mse"] - b["mse"]
        da = f["mae"] - b["mae"]
        push(f"{H:<8}{b['mse']:<16.4f}{f['mse']:<16.4f}{dm:<+14.4f}"
             f"{b['mae']:<16.4f}{f['mae']:<16.4f}{da:<+14.4f}")

    push("")
    push("Anchor check (sanity):")
    push(f"  baseline.mean_mse should land near 0.36–0.42 (prior 4090 runs gave 0.367 / 0.412).")
    push(f"  observed = {baseline['test_mean_mse']:.4f}  →  "
         + ("OK" if 0.30 <= baseline['test_mean_mse'] <= 0.45 else "OUT OF RANGE — investigate"))
    push("")
    push("Reference (prior 4090 runs, batch=32, prefix eval):")
    push("  600  iters : fixed.test_mean_mse = 0.1546")
    push("  3000 iters : (this run) — testing whether §10.2 long-budget compounds.")

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device = %s", device)
    if device.type == "cuda":
        logger.info("gpu = %s", torch.cuda.get_device_name(0))

    baseline = run_group(
        label="baseline",
        window_len=201,
        batch_size=64,
        use_prefix_eval=False,
        device=device,
    )
    with open(os.path.join(OUT_DIR, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)

    # Free GPU memory between runs.
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Long-budget fixed group: crop=3000 (TS2Vec actual default) with
    # batch=32 (validated on 4090, ~98 s/run at 600 iters). The previous
    # batch=64 attempt regressed (negatives crowded the InfoNCE matrix),
    # so we keep bs=32 and instead push pretrain_iters 600 → 3000 (§10.2
    # long-budget retrain) to test whether the extra training compounds
    # with the §10.1/§10.3 fixes.
    fixed = run_group(
        label="fixed",
        window_len=3000,
        batch_size=32,
        use_prefix_eval=True,
        device=device,
    )
    with open(os.path.join(OUT_DIR, "fixed.json"), "w") as f:
        json.dump(fixed, f, indent=2)

    write_diff(baseline, fixed, os.path.join(OUT_DIR, "diff.txt"))
    logger.info("All artefacts written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
