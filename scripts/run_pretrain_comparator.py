"""CLI entry point: pre-train the T-CLSC comparator.

The comparator is task-agnostic w.r.t. Bug #003a — it never runs
contrastive pretraining itself, only learns a pairwise ranker over the
already-collected (config, perf) seed records.  This script therefore
needs no iter-budget plumbing.

Single-stage usage
------------------
    python scripts/run_pretrain_comparator.py \\
        --seeds_path  ZeroAutoCL/outputs/seeds/seeds.json \\
        --data_dir    ZeroAutoCL/data/datasets \\
        --save_path   ZeroAutoCL/outputs/comparator.pt \\
        --config      ZeroAutoCL/configs/default.yaml \\
        --seed 42

Two-stage usage (AutoCTS++-style noisy → clean curriculum)
----------------------------------------------------------
    python scripts/run_pretrain_comparator.py \\
        --seeds_path        ZeroAutoCL/outputs/seeds_noisy/seeds.json \\
        --clean_seeds_path  ZeroAutoCL/outputs/seeds_clean/seeds.json \\
        --data_dir          ZeroAutoCL/data/datasets \\
        --save_path         ZeroAutoCL/outputs/comparator.pt \\
        --config            ZeroAutoCL/configs/default.yaml

In two-stage mode the comparator is first trained on the noisy seeds
(broad coverage, higher label noise), then fine-tuned on the clean seeds
(fewer seeds, more reliable labels).  Fine-tune defaults to 0.1 × the
noisy lr and half the noisy epochs; override via a ``comparator_clean:``
block in the YAML config.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from models.comparator.t_clsc import TCLSC
from models.comparator.task_feature import TaskFeatureExtractor, TASK_FEATURE_DIM
from models.search_space.space_encoder import RAW_DIM
from data.dataset import load_dataset
from data.dataset_slicer import (
    make_forecasting_subtasks,
    parse_task_id,
)
from search.seed_generator import SeedRecord
from search.pretrain_comparator import pretrain_comparator
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train T-CLSC comparator.")
    p.add_argument("--seeds_path", required=True, nargs="+",
                   help="One or more paths to seeds.json for the (noisy) "
                        "pretraining stage.  Multiple paths are concatenated "
                        "— useful for merging ``--mode noisy`` runs with "
                        "and without ``--randomise_init``.")
    p.add_argument("--clean_seeds_path", default=None, nargs="+",
                   help="Optional one-or-more paths to seeds.json from "
                        "clean-mode runs.  When given, a second fine-tune "
                        "stage is run on top of the noisy pretrained "
                        "comparator (AutoCTS++ two-stage).")
    p.add_argument("--data_dir",   required=True, help="Root data directory")
    p.add_argument("--save_path",  required=True, help="Output path for comparator.pt")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    comp_cfg       = cfg.get("comparator",       {}) or {}
    # Optional clean-stage override block.  If absent the fine-tune stage
    # will derive lr/epochs from comp_cfg (see _derive_clean_config).
    comp_clean_cfg = cfg.get("comparator_clean", None)
    # Forecasting task variants — must mirror the seed-generation config so
    # sub-task IDs in seeds.json round-trip back to the same sliced data.
    variants_cfg       = cfg.get("forecasting_task_variants", {}) or {}
    n_time_windows     = int(variants_cfg.get("n_time_windows", 1) or 1)
    horizon_groups     = variants_cfg.get("horizon_groups")
    min_window_len     = int(variants_cfg.get("min_window_len", 1000) or 1000)
    n_variable_subsets = int(variants_cfg.get("n_variable_subsets", 1) or 1)
    var_size_rates     = variants_cfg.get("var_size_rates")
    min_var_count      = int(variants_cfg.get("min_var_count", 4) or 4)
    sg_cfg             = cfg.get("seed_generation", {}) or {}
    crop_len           = sg_cfg.get("crop_len")
    if crop_len is not None:
        crop_len = int(crop_len)
    # The variable-subset seed must match what generate_seeds passed in so
    # the (tw_idx, vs_idx) → variable-index mapping is identical at feature
    # extraction time.  We default it to ``args.seed`` for the same reason.

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # ── Load seeds (+ optional clean seeds) ───────────────────────────
    def _load_seed_paths(paths: List[str], tag: str) -> List[SeedRecord]:
        records: List[SeedRecord] = []
        for path in paths:
            with open(path) as f:
                chunk = [SeedRecord.from_dict(d) for d in json.load(f)]
            logger.info("  loaded %d %s records from %s", len(chunk), tag, path)
            records.extend(chunk)
        return records

    seeds = _load_seed_paths(args.seeds_path, "noisy")
    logger.info("Total: %d noisy seed records across %d file(s).",
                len(seeds), len(args.seeds_path))

    clean_seeds: Optional[List[SeedRecord]] = None
    if args.clean_seeds_path:
        clean_seeds = _load_seed_paths(args.clean_seeds_path, "clean")
        logger.info(
            "Total: %d clean seed records across %d file(s) "
            "(two-stage training enabled).",
            len(clean_seeds), len(args.clean_seeds_path),
        )

    # ── Extract task features for every task_id seen across both sets ─
    # Sub-task IDs of the form ``"{base}:tw{i}"`` / ``"{base}:hg{j}"`` /
    # ``"{base}:tw{i}:hg{j}"`` are resolved by re-running the same
    # ``make_forecasting_subtasks`` call used at seed generation time, so
    # the comparator sees task features extracted from the *sliced* data
    # (preserving the per-window distinctions the comparator needs to learn).
    all_task_ids = {s.task_id for s in seeds}
    if clean_seeds:
        all_task_ids.update(s.task_id for s in clean_seeds)
    task_ids = sorted(all_task_ids)

    # Group sub-task IDs by their base dataset so ``make_forecasting_subtasks``
    # is only invoked once per source.
    by_base: dict = {}
    for tid in task_ids:
        parts = parse_task_id(tid)
        by_base.setdefault(parts.base, []).append(tid)

    tfe = TaskFeatureExtractor(device=device)
    task_features = {}
    for base, tids in sorted(by_base.items()):
        # No sub-task suffix → the legacy fast path: extract once on the
        # full dataset.  This keeps backward compatibility with seed files
        # generated before the variant patch.
        if len(tids) == 1 and tids[0] == base:
            logger.info("Extracting task features for: %s", base)
            splits = load_dataset(base, args.data_dir)
            task_features[base] = tfe.extract(
                splits["train"], splits["train"].task_type,
            )
            continue

        # Sub-task path: re-derive the SAME (window × var-subset) sub-tasks
        # used at seed-gen time.  ``var_subset_seed`` must match what
        # ``generate_seeds`` passed in (we use ``args.seed`` on both sides
        # so the per-source variable index lists are reproducible).
        sub_tasks = make_forecasting_subtasks(
            base, args.data_dir,
            n_time_windows=n_time_windows,
            horizon_groups=horizon_groups,
            crop_len=crop_len,
            min_window_len=min_window_len,
            n_variable_subsets=n_variable_subsets,
            var_size_rates=var_size_rates,
            min_var_count=min_var_count,
            var_subset_seed=args.seed,
        )
        # Index by window_id == "{base}[:tw{i}][:vs{j}]" so we can look up
        # any sub-task without horizon-group disambiguation.
        by_window = {sub.window_id: sub for sub in sub_tasks}

        for tid in tids:
            parts = parse_task_id(tid)
            # Reconstruct the window_id by stripping the :hg suffix.
            window_id = base
            if parts.tw_idx is not None:
                window_id += f":tw{parts.tw_idx}"
            if parts.vs_idx is not None:
                window_id += f":vs{parts.vs_idx}"
            if window_id not in by_window:
                logger.warning(
                    "Task feature for %s missing — window_id %s not produced "
                    "by make_forecasting_subtasks; falling back to base "
                    "dataset features.",
                    tid, window_id,
                )
                splits = load_dataset(base, args.data_dir)
                task_features[tid] = tfe.extract(
                    splits["train"], splits["train"].task_type,
                )
                continue
            sub = by_window[window_id]
            # Pick the explicit horizon for this group (when applicable) so
            # the meta-feature horizon scalar differs across hg variants of
            # the same window — otherwise the comparator gets identical
            # task features for differently-labelled (tid, candidate) pairs.
            horizon_meta = 0
            if parts.hg_idx is not None and parts.hg_idx < len(sub.horizon_groups):
                hg = sub.horizon_groups[parts.hg_idx]
                if hg:
                    horizon_meta = int(max(hg))
            logger.info(
                "Extracting task features for: %s  (window=%s  horizon_meta=%d  "
                "n_vars=%d)",
                tid, window_id, horizon_meta, sub.train.n_channels,
            )
            task_features[tid] = tfe.extract(
                sub.train, sub.train.task_type, horizon=horizon_meta,
            )
    logger.info("Task features ready for %d task IDs.", len(task_features))

    # ── Build and train comparator ─────────────────────────────────────
    comparator = TCLSC(
        candidate_dim=RAW_DIM,
        task_dim=TASK_FEATURE_DIM,
        hidden_dim=int(comp_cfg.get("hidden_dim", 128)),
    )
    comparator = pretrain_comparator(
        seeds=seeds,
        task_features=task_features,
        config=comp_cfg,
        comparator=comparator,
        save_path=args.save_path,
        device=device,
        clean_seeds=clean_seeds,
        clean_config=comp_clean_cfg,
    )
    logger.info("Comparator saved to %s", args.save_path)


if __name__ == "__main__":
    main()
