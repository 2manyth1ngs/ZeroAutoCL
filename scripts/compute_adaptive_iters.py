"""Compute per-source pretrain_iters / noisy_pretrain_iters via the
log_clip formula in ``configs/default.yaml::adaptive_iter`` and write the
resolved numbers back into ``dataset_budgets``.

Idempotent: re-running after changing window params updates iter values
in place.  Comment-preserving YAML round-trip via ``ruamel.yaml`` if
available, falling back to ``PyYAML`` (which loses comments — print a
warning in that case).

Usage:
    python ZeroAutoCL/scripts/compute_adaptive_iters.py \
        [--yaml ZeroAutoCL/configs/default.yaml] [--dry-run]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_YAML = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _log2_clip(scale: float, base: int, lo: int, hi: int) -> int:
    raw = base * (1.0 + math.log2(max(scale, 1e-12)))
    return int(round(max(lo, min(hi, raw))))


def _avg_var_size(n_raw_vars: int, n_variable_subsets: int, rates) -> float:
    """Average variable count across stratified bucket subsets.

    Mirrors ``data/dataset_slicer._sample_variable_subsets`` round logic.
    """
    if n_variable_subsets <= 1:
        return float(n_raw_vars)
    sizes = []
    for s in range(n_variable_subsets):
        rate = rates[s % len(rates)]
        size = int(round(rate * n_raw_vars))
        size = max(2, min(size, n_raw_vars))
        sizes.append(size)
    return sum(sizes) / len(sizes)


# Per-source (n_raw_vars, n_variable_subsets) for the CL-aligned pool.
# This mirrors how each source's variable axis behaves at seed-gen time;
# it's data we'd otherwise have to pull from yaml + dataset_slicer to
# recompute from scratch.  Univariate ETT (covariate-masked) → 1.  AQ →
# forced n_variable_subsets=1 in yaml.  Solar/traffic/ExchangeRate → 3
# stratified buckets at rates [0.25, 0.5, 0.75] over the full raw count.
SOURCE_VAR_INFO: Dict[str, Tuple[int, int]] = {
    "ETTh2":        (1,    1),
    "ETTm1":        (1,    1),
    "ETTm2":        (1,    1),
    "Solar":        (137,  3),
    "traffic":      (862,  3),
    "AQShunyi":     (11,   1),
    "AQWanliu":     (11,   1),
    "AQGuanyuan":   (11,   1),
    "ExchangeRate": (8,    3),
}


def compute_iters(cfg: dict) -> Dict[str, Tuple[int, int]]:
    """Compute (pretrain_iters, noisy_pretrain_iters) for every source in
    ``forecasting_task_variants.time_window_params``."""
    adaptive = cfg["adaptive_iter"]
    base = int(adaptive["base_iter"])
    ref = float(adaptive["ref_T_C"])
    lo = int(adaptive["clean_cap_min"])
    hi = int(adaptive["clean_cap_max"])
    noisy_ratio = float(adaptive["noisy_ratio"])
    var_rates = cfg["forecasting_task_variants"].get(
        "var_size_rates", [0.25, 0.5, 0.75]
    )
    tw_params = cfg["forecasting_task_variants"]["time_window_params"]

    out: Dict[str, Tuple[int, int]] = {}
    for src, params in tw_params.items():
        min_len = int(params["min_len"])
        max_len = int(params["max_len"])
        n_raw_vars, n_var_subsets = SOURCE_VAR_INFO.get(src, (1, 1))
        T_eff = (min_len + max_len) / 2.0
        C_eff = _avg_var_size(n_raw_vars, n_var_subsets, var_rates)
        scale = (T_eff * C_eff) / ref
        clean = _log2_clip(scale, base, lo, hi)
        noisy = int(round(clean * noisy_ratio))
        out[src] = (clean, noisy)
    return out


def load_yaml(path: Path):
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        with path.open("r", encoding="utf-8") as fr:
            return yaml, yaml.load(fr)
    except ImportError:
        import yaml
        print("WARN: ruamel.yaml not available — using PyYAML (comments will be lost)",
              file=sys.stderr)
        with path.open("r", encoding="utf-8") as fr:
            return yaml, yaml.safe_load(fr)


def write_yaml(yaml_mod, cfg, path: Path) -> None:
    try:
        # ruamel branch — has .dump method on the YAML instance
        with path.open("w", encoding="utf-8") as fw:
            yaml_mod.dump(cfg, fw)
    except AttributeError:
        # PyYAML branch
        with path.open("w", encoding="utf-8") as fw:
            yaml_mod.safe_dump(cfg, fw, sort_keys=False, allow_unicode=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved iter table without writing yaml.")
    args = parser.parse_args()

    yaml_mod, cfg = load_yaml(args.yaml)
    iters = compute_iters(cfg)

    print(f"{'source':<14} {'clean':>8} {'noisy':>8}")
    print("-" * 32)
    for src in sorted(iters):
        clean, noisy = iters[src]
        print(f"{src:<14} {clean:>8} {noisy:>8}")

    if args.dry_run:
        print("\n[dry-run] yaml NOT modified.")
        return 0

    db = cfg.setdefault("dataset_budgets", {})
    for src, (clean, noisy) in iters.items():
        bucket = db.setdefault(src, {})
        bucket["pretrain_iters"] = clean
        bucket["noisy_pretrain_iters"] = noisy

    write_yaml(yaml_mod, cfg, args.yaml)
    print(f"\nUpdated {args.yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
