"""Bug #003a — oracle diagnostic curve on ETTh1.

Train ETTh1 (univariate) for 40 epochs with the GGS_STRATEGY, evaluating
the **test split** every epoch (oracle / upper-bound view of early
stopping).  We need this to classify the loss / test-MSE shape:

  - early-bottom (ep ~3–5, then up)  → training budget too large
  - mid-bottom   (ep 10–20, then up)  → budget OK, val signal broken
  - monotone-decrease                 → budget too small

This script does **not** modify production code: it only consumes
``train.pretrain.contrastive_pretrain`` with ``val_data=test_data``,
and writes the per-epoch history to JSON for later inspection.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import torch

# Make package imports work when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import load_dataset                                # noqa: E402
from models.encoder.dilated_cnn import DilatedCNNEncoder             # noqa: E402
from models.contrastive.cl_pipeline import CLPipeline                # noqa: E402
from models.search_space.cl_strategy_space import (                  # noqa: E402
    GGS_STRATEGY, DEFAULT_ENCODER,
)
from train.pretrain import contrastive_pretrain                      # noqa: E402
from train.evaluate import eval_forecasting                          # noqa: E402
from utils.reproducibility import set_seed                           # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("diagnose_etth1_oracle")


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_dir = str(ROOT / "data" / "datasets")
    splits = load_dataset("ETTh1", data_dir)
    train_ds = splits["train"]
    test_ds  = splits["test"]
    logger.info(
        "ETTh1 loaded: train n_windows=%d  test shape=%s  C=%d",
        len(train_ds), tuple(test_ds.data.shape), train_ds.n_channels,
    )

    encoder = DilatedCNNEncoder.from_config_dict(
        train_ds.n_channels, DEFAULT_ENCODER,
    ).to(device)
    pipeline = CLPipeline(encoder, GGS_STRATEGY).to(device)

    config = dict(
        pretrain_epochs=40,
        pretrain_lr=1e-3,
        batch_size=64,
        eval_every=1,        # every epoch
    )

    history: list = []
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config=config,
        device=device,
        val_data=test_ds,           # ⚠ ORACLE: feed test as val (diagnostic only)
        task_type="forecasting",
        horizons=[24, 48, 168, 336, 720],
        history=history,
    )

    out_dir = ROOT / "outputs" / "diagnose_003a"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "etth1_oracle_curve.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved oracle history (%d entries) to %s", len(history), out_path)

    # Quick textual summary so we don't have to open the JSON to classify.
    logger.info("=== Oracle test-MSE curve (negative score, higher = better) ===")
    for h in history:
        v = h["val_score"]
        v_str = f"{v:+.4f}" if v is not None else "  none"
        logger.info("  ep %2d  loss=%.4f  test_score=%s", h["epoch"], h["loss"], v_str)

    # Find the oracle best epoch and the shape signal.
    scored = [h for h in history if h["val_score"] is not None]
    if scored:
        best = max(scored, key=lambda h: h["val_score"])
        logger.info(
            "ORACLE BEST: epoch=%d  test_score=%.4f  (mse=%.4f)",
            best["epoch"], best["val_score"], -best["val_score"],
        )


if __name__ == "__main__":
    main()
