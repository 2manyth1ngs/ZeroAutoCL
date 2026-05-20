"""CLI entry point: generate seed data for T-CLSC pre-training.

Post-Bug-#003a: forecasting datasets default to an iter budget read from
``dataset_budgets`` in ``configs/default.yaml``; classification / anomaly
datasets fall back to epoch-based training.  CLI flags
``--pretrain_iters`` and ``--pretrain_epochs`` override the YAML values
**globally** (applied to every source dataset).

Default source pool (forecasting, rev 2026-05-10 — CL-aligned 9 sources for
zero-shot to ETTh1 / Electricity / Weather targets):

  ETTh2 ETTm1 ETTm2 Solar traffic AQShunyi AQWanliu AQGuanyuan ExchangeRate

Mirrors AutoCTS++ §4.1.1 — every target has ≥2 same-domain neighbours in
the source pool, and the seed-record distribution is balanced across five
task-feature clusters (ETT 37% / Energy 16% / Transport 16% / Atmospheric
16% / Finance 16%) so the comparator's zero-shot ranking doesn't collapse
onto a dominant-domain preference.  PEMS / METR-LA / ILI are excluded
because (a) they are not CL-forecasting benchmarks and (b) the prior
12-source PEMS-heavy run produced mode collapse on ETTh1 (mean MSE 0.180
vs 0.090 for the 6-source pool).

Usage
-----
python scripts/run_generate_seeds.py \\
    --data_dir  ZeroAutoCL/data/datasets \\
    --save_dir  ZeroAutoCL/outputs/seeds \\
    --config    ZeroAutoCL/configs/default.yaml \\
    --datasets  ETTh2 ETTm1 ETTm2 Solar traffic \\
                AQShunyi AQWanliu AQGuanyuan ExchangeRate \\
    --n_per_dataset 30 \\
    --batch_size 64 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from search.seed_generator import generate_seeds
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ZeroAutoCL seed data.")
    p.add_argument("--data_dir",        required=True,  help="Root data directory")
    p.add_argument("--save_dir",        required=True,  help="Output directory for seeds.json")
    p.add_argument("--config",          default=None,   help="Path to default.yaml (optional)")
    p.add_argument("--datasets",        nargs="+",
                   default=["ETTh2", "ETTm1", "ETTm2",
                            "Solar", "traffic",
                            "AQShunyi", "AQWanliu", "AQGuanyuan",
                            "ExchangeRate"],
                   help="Source dataset names. Default = CL-aligned 9-source "
                        "pool (rev 2026-05-10): 3 ETT siblings, 1 Solar "
                        "(energy), 1 traffic.csv (transport), 3 Beijing AQ "
                        "stations (atmospheric), 1 ExchangeRate (finance). "
                        "Excludes PEMS/METR-LA (graph-traffic, not CL "
                        "benchmarks) and ILI (too short for stable CL).")
    p.add_argument("--n_per_dataset",   type=int,       default=30)
    p.add_argument("--n_shared",        type=int,       default=None,
                   help="Cross-source L-share pool size (AutoCTS++ trick). "
                        "When >0, the first n_shared candidates per source are "
                        "identical across ALL sources; the rest are sampled "
                        "fresh per source.  None → read from YAML "
                        "(seed_generation.n_shared, default 0).")
    p.add_argument("--n_noisy_per_dataset", type=int, default=None,
                   help="Noisy candidates per (source × sub-task).  When >0, "
                        "each sub-task runs an additional cheap-budget pass "
                        "(noisy_pretrain_iters per candidate) producing "
                        "stage=\"noisy\" seed records.  Increases the per-task "
                        "pair pool ~10× at ~1.4× the clean-stage cost.  "
                        "None → read from YAML "
                        "(seed_generation.n_noisy_per_dataset, default 0 = "
                        "noisy disabled).")
    p.add_argument("--noisy_pretrain_iters", type=int, default=None,
                   help="Iter budget per noisy candidate.  100 ≈ 1/6 of the "
                        "canonical 600-iter clean budget.  Lower values save "
                        "wall-clock but risk noisier labels; the per-sub-task "
                        "noisy-vs-clean Spearman ρ printed at sub-task end is "
                        "the calibration signal.  None → read from YAML "
                        "(seed_generation.noisy_pretrain_iters, default 100).")
    p.add_argument("--source_global_idx_offset", type=int, default=0,
                   help="Per-source ds-index offset for SLURM job-array "
                        "fan-out.  Each array task processes ONE source "
                        "(--datasets <SRC>) and passes its --array task ID "
                        "here so the per-source random pool seed and the "
                        "per-candidate seed line up across the array exactly "
                        "like a sequential single-job run.  Default 0 = "
                        "sequential mode.")
    p.add_argument("--pretrain_epochs", type=int,       default=None,
                   help="Override per-dataset epoch budget for ALL datasets")
    p.add_argument("--pretrain_iters",  type=int,       default=None,
                   help="Override per-dataset iter budget for ALL datasets "
                        "(takes precedence over --pretrain_epochs)")
    p.add_argument("--batch_size",      type=int,       default=64)
    p.add_argument("--seed",            type=int,       default=42)
    p.add_argument("--device",          default=None,   help="'cpu' or 'cuda'")
    p.add_argument("--encoder_grid",    default=None,
                   help="(Plan B) Path to encoder_grid.json from Stage A. "
                        "When given, seed generation restricts the encoder "
                        "sub-space to the top-K rows of this file.")
    p.add_argument("--top_k_enc",       type=int, default=3,
                   help="(Plan B) How many encoders to keep from --encoder_grid.")
    # ── Forecasting task-variant axes ─────────────────────────────────────
    p.add_argument("--n_time_windows", type=int, default=None,
                   help="Override forecasting_task_variants.n_time_windows "
                        "from the YAML.  >1 enables AutoCTS++-style temporal "
                        "subset enrichment.")
    p.add_argument("--horizon_groups", default=None,
                   help="Override forecasting_task_variants.horizon_groups. "
                        "Format: comma-separated horizons within a group, "
                        "semicolons between groups. "
                        "Example: '24,48,168;336,720' "
                        "→ [[24,48,168],[336,720]].")
    p.add_argument("--n_variable_subsets", type=int, default=None,
                   help="Override forecasting_task_variants.n_variable_subsets. "
                        ">=2 engages AutoCTS++-style stratified variable "
                        "subsampling.  Sources with fewer than min_var_count "
                        "raw variables (e.g. univariate ETT) silently bypass "
                        "this axis.")
    return p.parse_args()


def _load_top_k_encoders(path: str, k: int) -> List[Dict[str, int]]:
    """Read encoder_grid.json and return its top-K encoder configs.

    The file is a list of records sorted by ``mean_perf`` desc, as written
    by :func:`search.encoder_grid_search.encoder_grid_search`.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"encoder_grid file is empty or malformed: {path}")
    return [r["encoder_config"] for r in records[:k]]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # ── Defaults from YAML ────────────────────────────────────────────────
    sg_cfg: Dict = {}
    yaml_dataset_budgets: Dict[str, Dict[str, int]] = {}
    variants_cfg: Dict = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sg_cfg               = cfg.get("seed_generation", {}) or {}
        yaml_dataset_budgets = cfg.get("dataset_budgets",  {}) or {}
        variants_cfg         = cfg.get("forecasting_task_variants", {}) or {}

    n_per_dataset   = args.n_per_dataset
    n_shared        = args.n_shared
    if n_shared is None:
        n_shared = int(sg_cfg.get("n_shared", 0) or 0)
    n_shared = max(0, min(int(n_shared), int(n_per_dataset)))
    pretrain_epochs = sg_cfg.get("pretrain_epochs", 40)
    pretrain_lr     = sg_cfg.get("pretrain_lr",     1e-3)
    batch_size      = sg_cfg.get("batch_size",      args.batch_size)
    crop_len        = sg_cfg.get("crop_len")
    if crop_len is not None:
        crop_len = int(crop_len)

    # Noisy stage resolution: CLI overrides YAML.
    n_noisy_per_dataset = args.n_noisy_per_dataset
    if n_noisy_per_dataset is None:
        n_noisy_per_dataset = int(sg_cfg.get("n_noisy_per_dataset", 0) or 0)
    n_noisy_per_dataset = max(0, int(n_noisy_per_dataset))
    noisy_pretrain_iters = args.noisy_pretrain_iters
    if noisy_pretrain_iters is None:
        noisy_pretrain_iters = int(sg_cfg.get("noisy_pretrain_iters", 100) or 100)
    noisy_pretrain_iters = max(1, int(noisy_pretrain_iters))

    # ── Forecasting task variants (time windows + horizon groups) ─────────
    n_time_windows = int(variants_cfg.get("n_time_windows", 1) or 1)
    if args.n_time_windows is not None:
        n_time_windows = max(1, int(args.n_time_windows))
    horizon_groups = variants_cfg.get("horizon_groups")  # may be null/None
    if args.horizon_groups is not None:
        # CLI override: comma-separated horizons, semicolons split groups,
        # e.g. "24,48,168;336,720" → [[24,48,168],[336,720]].
        horizon_groups = [
            [int(h) for h in grp.split(",") if h.strip()]
            for grp in args.horizon_groups.split(";") if grp.strip()
        ]
    min_window_len = int(variants_cfg.get("min_window_len", 1000) or 1000)
    # ── Variable subsampling axis (AutoCTS++-style stratified buckets) ────
    n_variable_subsets = int(variants_cfg.get("n_variable_subsets", 1) or 1)
    if args.n_variable_subsets is not None:
        n_variable_subsets = max(1, int(args.n_variable_subsets))
    var_size_rates = variants_cfg.get("var_size_rates")  # may be null
    min_var_count  = int(variants_cfg.get("min_var_count", 4) or 4)

    # ── Per-dataset budget resolution ─────────────────────────────────────
    # Resolution order (merge semantics, NOT replace):
    #   1. Start from configs/default.yaml :: dataset_budgets[name] (carries
    #      per-dataset overrides for n_time_windows, n_variable_subsets,
    #      crop_len, eval_horizons, …).
    #   2. If CLI --pretrain_iters / --pretrain_epochs is set, REPLACE only
    #      the compute key (drops the other one to avoid ambiguity, leaves
    #      everything else untouched).
    #   3. If both YAML budget AND CLI overrides are absent, fall back to the
    #      global ``pretrain_epochs`` from the seed_generation block.
    #
    # Bug #1 (2026-05-10): the previous logic REPLACED the whole budget dict
    # when CLI --pretrain_iters/--pretrain_epochs was set, silently dropping
    # YAML-side n_variable_subsets / n_time_windows / etc. overrides.  This
    # caused AQ stations to inflate from 3 to 9 sub-tasks in dry runs.
    dataset_budgets: Dict[str, Dict[str, int]] = {}
    for ds in args.datasets:
        # Step 1: base = YAML budget (may be empty).
        budget: Dict[str, int] = dict(yaml_dataset_budgets.get(ds, {}))

        # Step 2: CLI compute override — replace only the compute key.
        if args.pretrain_iters is not None and args.pretrain_iters > 0:
            budget.pop("pretrain_epochs", None)
            budget["pretrain_iters"] = int(args.pretrain_iters)
        elif args.pretrain_epochs is not None:
            budget.pop("pretrain_iters", None)
            budget["pretrain_epochs"] = int(args.pretrain_epochs)
        elif not budget:
            # Step 3: nothing from YAML and no CLI override → global fallback.
            budget = {"pretrain_epochs": int(pretrain_epochs)}

        dataset_budgets[ds] = budget

    device = torch.device(args.device) if args.device else None

    fixed_encoders: Optional[List[Dict[str, int]]] = None
    if args.encoder_grid:
        fixed_encoders = _load_top_k_encoders(args.encoder_grid, args.top_k_enc)
        logger.info(
            "[plan-B] using top-%d encoders from %s -> %s",
            args.top_k_enc, args.encoder_grid, fixed_encoders,
        )

    logger.info(
        "Generating seeds: datasets=%s  n_per=%d (shared=%d, random=%d)  "
        "lr=%.4f  bs=%d",
        args.datasets, n_per_dataset, n_shared, n_per_dataset - n_shared,
        pretrain_lr, batch_size,
    )
    if n_noisy_per_dataset > 0:
        logger.info(
            "Noisy stage ENABLED: n_noisy_per_dataset=%d  noisy_pretrain_iters=%d",
            n_noisy_per_dataset, noisy_pretrain_iters,
        )
    else:
        logger.info("Noisy stage disabled (n_noisy_per_dataset=0).")
    logger.info("Per-dataset budgets: %s", dataset_budgets)

    logger.info("crop_len=%s", crop_len)
    logger.info(
        "task variants: n_time_windows=%d  n_variable_subsets=%d  "
        "horizon_groups=%s  min_window_len=%d  var_size_rates=%s  "
        "min_var_count=%d",
        n_time_windows, n_variable_subsets, horizon_groups,
        min_window_len, var_size_rates, min_var_count,
    )

    fcv = variants_cfg
    seeds = generate_seeds(
        source_datasets=args.datasets,
        data_dir=args.data_dir,
        n_per_dataset=n_per_dataset,
        n_shared=n_shared,
        source_global_idx_offset=int(args.source_global_idx_offset),
        pretrain_epochs=int(pretrain_epochs),    # used only as fallback
        pretrain_lr=float(pretrain_lr),
        batch_size=int(batch_size),
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        dataset_budgets=dataset_budgets,
        fixed_encoders=fixed_encoders,
        crop_len=crop_len,
        n_time_windows=n_time_windows,
        horizon_groups=horizon_groups,
        min_window_len=min_window_len,
        n_variable_subsets=n_variable_subsets,
        var_size_rates=var_size_rates,
        min_var_count=min_var_count,
        n_noisy_per_dataset=n_noisy_per_dataset,
        noisy_pretrain_iters=noisy_pretrain_iters,
        use_random_time_windows=bool(fcv.get("use_random_time_windows", False)),
        max_overlap_ratio=float(fcv.get("max_overlap_ratio", 0.7)),
        time_window_params=fcv.get("time_window_params"),
    )

    logger.info("Done — generated %d seed records.", len(seeds))


if __name__ == "__main__":
    main()
