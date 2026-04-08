"""End-to-end ZeroAutoCL experiment.

Target  : ETTh1
Sources : ETTm1, PEMS03, PEMS04, PEMS07, PEMS08, ExchangeRate, PEMS-BAY
          (all forecasting datasets except the ETTh family)

Pipeline:
  Phase 1   — generate seeds on the 7 source datasets
  Phase 1.5 — pretrain T-CLSC comparator on those seeds
  Phase 2   — sample candidate pool, rank with comparator, pick top-K
  Phase 3   — fully retrain top-K and pick the best
  Phase 4   — final eval of the winning candidate on ETTh1 test split

All forecasting pretraining uses the iter-budget mechanism from Bug #003a
(no val-best for forecasting, fixed iter budget per dataset).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import load_dataset                                # noqa: E402
from models.comparator.t_clsc import TCLSC                           # noqa: E402
from models.comparator.task_feature import (                         # noqa: E402
    TaskFeatureExtractor, TASK_FEATURE_DIM,
)
from models.search_space.space_encoder import RAW_DIM                # noqa: E402
from search.seed_generator import generate_seeds, SeedRecord         # noqa: E402
from search.pretrain_comparator import pretrain_comparator           # noqa: E402
from search.zero_shot_search import zero_shot_search                 # noqa: E402
from train.evaluate import eval_forecasting                          # noqa: E402
from models.encoder.dilated_cnn import DilatedCNNEncoder             # noqa: E402
from models.contrastive.cl_pipeline import CLPipeline                # noqa: E402
from train.pretrain import contrastive_pretrain                     # noqa: E402
from utils.reproducibility import set_seed                           # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("full_experiment")


# ── Experiment configuration ────────────────────────────────────────────
TARGET = "ETTh1"
SOURCES = [
    "ETTm1", "PEMS03", "PEMS04", "PEMS07", "PEMS08",
    "ExchangeRate", "PEMS-BAY",
]
DATA_DIR = str(ROOT / "data" / "datasets")
OUT_DIR  = ROOT / "outputs" / "full_etth1"
SEEDS_PATH      = OUT_DIR / "seeds.json"
COMPARATOR_PATH = OUT_DIR / "comparator.pt"
RESULT_PATH     = OUT_DIR / "result.json"

# Budgets — small enough to finish quickly, large enough that the comparator
# sees real performance signal.
N_PER_DATASET     = 6           # candidates per source dataset
SEED_PRETRAIN_ITERS = 300       # iter budget per candidate during seed gen
COMPARATOR_EPOCHS = 60
N_CANDIDATES_POOL = 2000        # sampled candidate pool for zero-shot search
TOP_K             = 5
TOURN_ROUNDS      = 12
TOPK_FULL_ITERS   = 600         # full-train budget for the top-K candidates


def main() -> None:
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Target: %s   Sources: %s", TARGET, SOURCES)

    # Per-dataset iter budget for forecasting (Bug #003a).
    dataset_budgets = {ds: {"pretrain_iters": SEED_PRETRAIN_ITERS} for ds in SOURCES}
    dataset_budgets[TARGET] = {"pretrain_iters": TOPK_FULL_ITERS}
    logger.info("Dataset budgets: %s", dataset_budgets)

    # ────────────────────────────────────────────────────────────────────
    # Phase 1 — generate seeds
    # ────────────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("═══════ Phase 1: seed generation ═══════")
    seeds = generate_seeds(
        source_datasets=SOURCES,
        data_dir=DATA_DIR,
        n_per_dataset=N_PER_DATASET,
        pretrain_epochs=0,                    # ignored when iters > 0
        pretrain_lr=1e-3,
        batch_size=64,
        save_dir=str(OUT_DIR),                # writes seeds.json
        device=device,
        seed=42,
        dataset_budgets=dataset_budgets,
    )
    logger.info(
        "Phase 1 done — %d seed records, took %.1fs",
        len(seeds), time.time() - t0,
    )

    # Quick perf summary per dataset.
    by_ds: dict = {}
    for s in seeds:
        by_ds.setdefault(s.task_id, []).append(s.performance)
    for ds, perfs in by_ds.items():
        logger.info(
            "  %s: n=%d  perf min=%.4f  max=%.4f  mean=%.4f",
            ds, len(perfs), min(perfs), max(perfs), sum(perfs) / len(perfs),
        )

    # ────────────────────────────────────────────────────────────────────
    # Phase 1.5 — pretrain comparator
    # ────────────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("═══════ Phase 1.5: comparator pretraining ═══════")

    # Extract task features for each source.
    tfe = TaskFeatureExtractor(device=device)
    task_features: dict = {}
    for ds in SOURCES:
        logger.info("  extracting task features: %s", ds)
        sp = load_dataset(ds, DATA_DIR)
        task_features[ds] = tfe.extract(sp["train"], "forecasting")

    comp_cfg = {
        "epochs": COMPARATOR_EPOCHS,
        "lr": 1e-4,
        "batch_size": 256,
        "curriculum_levels": 5,
        "hidden_dim": 128,
    }
    comparator = TCLSC(
        candidate_dim=RAW_DIM,
        task_dim=TASK_FEATURE_DIM,
        hidden_dim=int(comp_cfg["hidden_dim"]),
    )
    comparator = pretrain_comparator(
        seeds=seeds,
        task_features=task_features,
        config=comp_cfg,
        comparator=comparator,
        save_path=str(COMPARATOR_PATH),
        device=device,
    )
    logger.info("Phase 1.5 done — saved %s, took %.1fs",
                COMPARATOR_PATH, time.time() - t0)

    # ────────────────────────────────────────────────────────────────────
    # Phase 2 + 3 — zero-shot search on the target
    # ────────────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("═══════ Phase 2+3: zero-shot search on %s ═══════", TARGET)

    best_enc, best_strat, best_perf = zero_shot_search(
        target_dataset=TARGET,
        data_dir=DATA_DIR,
        comparator=comparator,
        task_feature_extractor=tfe,
        n_candidates=N_CANDIDATES_POOL,
        top_k=TOP_K,
        tournament_rounds=TOURN_ROUNDS,
        pretrain_epochs=0,                    # ignored when iters > 0
        pretrain_iters=TOPK_FULL_ITERS,
        pretrain_lr=1e-3,
        batch_size=64,
        device=device,
    )
    logger.info(
        "Phase 2+3 done — best perf=%.4f (val MSE=%.4f), took %.1fs",
        best_perf, -best_perf, time.time() - t0,
    )

    # ────────────────────────────────────────────────────────────────────
    # Phase 4 — final eval of best candidate on the target test split
    # ────────────────────────────────────────────────────────────────────
    t0 = time.time()
    logger.info("═══════ Phase 4: retrain best candidate + ETTh1 test eval ═══════")

    splits = load_dataset(TARGET, DATA_DIR)
    train_ds, test_ds = splits["train"], splits["test"]
    set_seed(42)
    encoder = DilatedCNNEncoder.from_config_dict(
        train_ds.n_channels, best_enc,
    ).to(device)
    pipeline = CLPipeline(encoder, best_strat).to(device)
    contrastive_pretrain(
        encoder=encoder,
        cl_pipeline=pipeline,
        train_data=train_ds,
        config={"pretrain_iters": TOPK_FULL_ITERS,
                "pretrain_lr": 1e-3, "batch_size": 64},
        device=device,
        task_type="forecasting",
    )
    encoder.eval()
    test_metrics = eval_forecasting(
        encoder, train_ds, test_ds,
        horizons=[24, 48, 168, 336, 720], device=device,
    )
    mean_mse = sum(v["mse"] for v in test_metrics.values()) / len(test_metrics)
    mean_mae = sum(v["mae"] for v in test_metrics.values()) / len(test_metrics)
    logger.info(
        "Phase 4 done — ETTh1 TEST mean MSE=%.4f  MAE=%.4f  (took %.1fs)",
        mean_mse, mean_mae, time.time() - t0,
    )

    # Persist final result
    result = {
        "target": TARGET,
        "sources": SOURCES,
        "best_encoder_config": best_enc,
        "best_strategy": best_strat,
        "val_neg_mse": best_perf,
        "test_mean_mse": mean_mse,
        "test_mean_mae": mean_mae,
        "test_per_horizon": test_metrics,
        "n_seeds": len(seeds),
        "config": {
            "n_per_dataset": N_PER_DATASET,
            "seed_pretrain_iters": SEED_PRETRAIN_ITERS,
            "comparator_epochs": COMPARATOR_EPOCHS,
            "n_candidates_pool": N_CANDIDATES_POOL,
            "top_k": TOP_K,
            "topk_full_iters": TOPK_FULL_ITERS,
        },
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved final result to %s", RESULT_PATH)

    logger.info("═════════════════════════════════════════")
    logger.info("FINAL: ETTh1 test mean MSE = %.4f", mean_mse)
    logger.info("       ETTh1 test mean MAE = %.4f", mean_mae)
    logger.info("═════════════════════════════════════════")


if __name__ == "__main__":
    main()
