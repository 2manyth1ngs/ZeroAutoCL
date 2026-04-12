"""Stage A of Plan B: encoder grid search under fixed GGS strategy.

For each source dataset, every encoder configuration in
``ENCODER_CHOICES`` (36 total) is trained with ``GGS_STRATEGY`` and
evaluated on the held-out validation split.  Performance is then
aggregated across source datasets (arithmetic mean, ignoring failed runs)
to produce a global ranking that the rest of Plan B treats as a
task-independent prior over encoder architectures.

Outputs a list of records sorted by ``mean_perf`` descending, and
optionally persists them to ``{save_dir}/encoder_grid.json``.
"""

from __future__ import annotations

import json
import os
import time
from itertools import product
from typing import Dict, List, Optional

import torch

from data.dataset import load_dataset
from models.encoder.encoder_config import ENCODER_CHOICES
from models.search_space.cl_strategy_space import GGS_STRATEGY
from search.seed_generator import _evaluate_candidate, _fmt_hms
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def enumerate_encoder_grid() -> List[Dict[str, int]]:
    """Return all encoder configs from ``ENCODER_CHOICES`` in deterministic order."""
    keys = ["n_layers", "hidden_dim", "output_dim"]
    return [
        dict(zip(keys, vals))
        for vals in product(*(ENCODER_CHOICES[k] for k in keys))
    ]


def encoder_grid_search(
    source_datasets: List[str],
    data_dir: str,
    pretrain_lr: float = 1e-3,
    batch_size: int = 64,
    dataset_budgets: Optional[Dict[str, Dict[str, int]]] = None,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    crop_len: Optional[int] = None,
) -> List[Dict]:
    """Run Stage A: every encoder × every source dataset under ``GGS_STRATEGY``.

    Args:
        source_datasets: Names of source datasets (target dataset must NOT be in here).
        data_dir: Root data directory.
        pretrain_lr: Pretraining learning rate.
        batch_size: Training batch size.
        dataset_budgets: Optional per-dataset budget overrides; same shape
            as ``configs/default.yaml`` ``dataset_budgets``.
        save_dir: If given, persist results to ``{save_dir}/encoder_grid.json``.
        device: Torch device. ``None`` → auto-detect.
        seed: Global random seed.
        crop_len: Optional sliding-window crop length override for
            forecasting training splits (passed to :func:`load_dataset`).

    Returns:
        List of records sorted by ``mean_perf`` desc::

            [
              {"encoder_config": {...}, "mean_perf": float,
               "per_dataset": {ds_name: float, ...}},
              ...
            ]
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = enumerate_encoder_grid()
    n_grid = len(grid)
    n_ds = len(source_datasets)
    logger.info(
        "[stage-A] encoder grid: %d encoders × %d source datasets = %d runs",
        n_grid, n_ds, n_grid * n_ds,
    )

    per_enc: Dict[str, Dict] = {}
    overall_start = time.time()

    for ds_idx, ds_name in enumerate(source_datasets):
        logger.info(
            "[stage-A] dataset %d/%d: %s", ds_idx + 1, n_ds, ds_name,
        )
        splits = load_dataset(ds_name, data_dir, window_len_override=crop_len)
        train_ds = splits["train"]
        val_ds   = splits.get("val") or splits["test"]
        task_type = train_ds.task_type

        budget = (dataset_budgets or {}).get(ds_name, {}) or {}
        ds_iters  = int(budget.get("pretrain_iters", 0))
        ds_epochs = int(budget.get("pretrain_epochs", 40))
        if ds_iters > 0:
            logger.info("  budget: pretrain_iters=%d", ds_iters)
        else:
            logger.info("  budget: pretrain_epochs=%d", ds_epochs)

        ds_start = time.time()
        for i, enc_cfg in enumerate(grid):
            key = (
                f"L{enc_cfg['n_layers']}"
                f"_H{enc_cfg['hidden_dim']}"
                f"_O{enc_cfg['output_dim']}"
            )
            cand_start = time.time()
            perf = _evaluate_candidate(
                enc_cfg, GGS_STRATEGY,
                train_ds, val_ds, task_type,
                ds_epochs, pretrain_lr, batch_size,
                device, pretrain_iters=ds_iters,
            )
            cand_elapsed = time.time() - cand_start

            slot = per_enc.setdefault(
                key, {"encoder_config": enc_cfg, "per_dataset": {}}
            )
            slot["per_dataset"][ds_name] = perf

            done = i + 1
            avg = (time.time() - ds_start) / done
            ds_eta = avg * (n_grid - done)
            logger.info(
                "  [%s] %d/%d %s perf=%.6f  took %s  ds-ETA %s",
                ds_name, done, n_grid, key, perf,
                _fmt_hms(cand_elapsed), _fmt_hms(ds_eta),
            )

        logger.info(
            "[stage-A] %s done in %s",
            ds_name, _fmt_hms(time.time() - ds_start),
        )

    # ── Aggregate ──────────────────────────────────────────────────────
    records: List[Dict] = []
    for slot in per_enc.values():
        vals = [v for v in slot["per_dataset"].values() if v > -1e8]
        slot["mean_perf"] = sum(vals) / len(vals) if vals else -1e9
        records.append(slot)
    records.sort(key=lambda r: r["mean_perf"], reverse=True)

    logger.info(
        "[stage-A] all done in %s — top encoder: %s perf=%.6f",
        _fmt_hms(time.time() - overall_start),
        records[0]["encoder_config"], records[0]["mean_perf"],
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "encoder_grid.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info("[stage-A] saved %d records to %s", len(records), path)

    return records


def select_top_k_encoders(
    records: List[Dict],
    k: int = 3,
) -> List[Dict[str, int]]:
    """Pick the top-K encoder configs from aggregated grid records."""
    return [r["encoder_config"] for r in records[:k]]
