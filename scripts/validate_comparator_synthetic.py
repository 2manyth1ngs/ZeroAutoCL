"""Synthetic architecture validation for the T-CLSC comparator.

Builds seed records whose ``performance`` is a **known function** of
strategy features + additive noise, trains the comparator two-stage on
them, and evaluates Spearman correlation on a held-out strategy pool.

Purpose
-------
Proves (or disproves) that the comparator code + training loop CAN learn
a pairwise ranker, independently of the real CL pipeline and real seed
noise.  If Spearman ≈ 0 here, there is a bug.  If Spearman is high, any
shortcoming on real data is about **data volume**, not architecture.

Ground-truth formula
--------------------
Higher = better:

    perf = -0.30·|crop_p - 0.30|                    # peaked at 0.30
         + -0.20·|log10(temperature)|               # peaked at 1.0
         + {dot: 0.10, euclidean: 0.05, cosine: 0}  # sim preference
         + {k=2: 0.08, k=5: 0.03, else: 0}          # kernel preference
         + (0.05 if temporal else 0)                # mild +
         + N(0, noise_std)

Run
---
    python scripts/validate_comparator_synthetic.py
"""

from __future__ import annotations

import math
import os
import random
import sys
from itertools import combinations
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

from models.comparator.t_clsc import TCLSC
from models.comparator.task_feature import TASK_FEATURE_DIM
from models.search_space.cl_strategy_space import DEFAULT_ENCODER, sample_cl_strategy
from search.pretrain_comparator import pretrain_comparator
from search.seed_generator import SeedRecord


# ---------------------------------------------------------------------------
# Synthetic ground truth
# ---------------------------------------------------------------------------

def true_performance(strategy: Dict) -> float:
    """Deterministic performance as a function of strategy features."""
    aug  = strategy["augmentation"]
    pair = strategy["pair_construction"]
    loss = strategy["loss"]

    c1 = -0.30 * abs(aug["crop"] - 0.30)                        # crop sweet spot 0.3
    c2 = -0.20 * abs(math.log10(loss["temperature"] + 1e-9))    # temp sweet spot 1.0
    c3 = {"dot": 0.10, "euclidean": 0.05, "cosine": 0.0,
          "distance": 0.05}[loss["sim_func"]]                   # sim preference
    c4 = {2: 0.08, 5: 0.03}.get(pair["kernel_size"], 0.0)       # kernel preference
    c5 = 0.05 if pair["temporal"] else 0.0                       # mild temporal bonus
    return float(c1 + c2 + c3 + c4 + c5)


def make_seed(strategy: Dict, task_id: str, noise_std: float,
              encoder_config: Dict, rng: np.random.Generator) -> SeedRecord:
    perf = true_performance(strategy) + float(rng.normal(0.0, noise_std))
    return SeedRecord(
        encoder_config=dict(encoder_config),
        strategy=strategy,
        task_id=task_id,
        performance=perf,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Scale — fast (< 1 min).  With 3 tasks × 45 noisy + 3 × 15 clean = 180
    # seeds, the comparator has plenty of signal; any failure is a bug.
    n_tasks        = 3
    noisy_per_task = 45
    clean_per_task = 15
    n_test         = 20
    noise_noisy    = 0.05
    noise_clean    = 0.02

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random but consistent task features.
    torch.manual_seed(42)
    task_ids = [f"fake_task_{i}" for i in range(n_tasks)]
    task_features = {tid: torch.randn(TASK_FEATURE_DIM) for tid in task_ids}

    encoder_config = {**DEFAULT_ENCODER, "mask_mode": "binomial"}

    # Strategy sampling — separate streams so splits don't overlap.
    random.seed(0)
    noisy_strategies = [sample_cl_strategy() for _ in range(noisy_per_task * n_tasks)]
    clean_strategies = [sample_cl_strategy() for _ in range(clean_per_task * n_tasks)]
    test_strategies  = [sample_cl_strategy() for _ in range(n_test)]

    rng = np.random.default_rng(42)
    noisy_seeds = [
        make_seed(noisy_strategies[i], task_ids[i % n_tasks], noise_noisy, encoder_config, rng)
        for i in range(len(noisy_strategies))
    ]
    clean_seeds = [
        make_seed(clean_strategies[i], task_ids[i % n_tasks], noise_clean, encoder_config, rng)
        for i in range(len(clean_strategies))
    ]

    # Sanity snapshot of the ground truth distribution.
    tests_perf = np.array([true_performance(s) for s in test_strategies])
    print(f"Ground-truth perf range on test pool: "
          f"[{tests_perf.min():+.4f}, {tests_perf.max():+.4f}]  "
          f"std={tests_perf.std():.4f}")
    noisy_perfs = np.array([s.performance for s in noisy_seeds])
    print(f"Noisy seed perf range: [{noisy_perfs.min():+.4f}, "
          f"{noisy_perfs.max():+.4f}]  std={noisy_perfs.std():.4f}")

    # ── Train comparator two-stage ────────────────────────────────────
    print("\n=== training comparator ===")
    comparator = pretrain_comparator(
        seeds=noisy_seeds,
        task_features=task_features,
        config={
            "epochs": 100, "lr": 1e-3, "batch_size": 256, "hidden_dim": 128,
            "valid_gap_threshold": 0.02, "patience": 10, "eval_every": 1,
        },
        clean_seeds=clean_seeds,
        device=device,
    )

    # ── Spearman evaluation on held-out pool ──────────────────────────
    comparator.eval()
    task_feat = task_features[task_ids[0]]
    n = len(test_strategies)
    wins = np.zeros(n, dtype=np.int64)

    pair_iter = list(combinations(range(n), 2))
    BATCH = 64
    with torch.no_grad():
        for start in range(0, len(pair_iter), BATCH):
            batch = pair_iter[start : start + BATCH]
            enc_a   = [encoder_config     for _ in batch]
            strat_a = [test_strategies[i] for i, _ in batch]
            enc_b   = [encoder_config     for _ in batch]
            strat_b = [test_strategies[j] for _, j in batch]
            probs = comparator.forward_batch(
                enc_a, strat_a, enc_b, strat_b, task_feat,
            ).cpu().numpy()
            for (i, j), p in zip(batch, probs):
                if p > 0.5:
                    wins[i] += 1
                else:
                    wins[j] += 1

    rho, p_rho = spearmanr(wins, tests_perf)
    tau, p_tau = kendalltau(wins, tests_perf)

    print("\n" + "=" * 70)
    print(f"Synthetic comparator validation  n_test={n}  pairs={len(pair_iter)}")
    print("=" * 70)
    print(f"  Spearman rho = {rho:+.4f}  (p={p_rho:.3g})")
    print(f"  Kendall  tau = {tau:+.4f}  (p={p_tau:.3g})")
    print()

    # Rank-by-rank comparison.
    order_true = np.argsort(-tests_perf)
    order_pred = np.argsort(-wins)
    print(f"{'rank':>4} {'idx(true)':>9} {'true_perf':>9} | {'idx(pred)':>9} {'wins':>6}")
    for r, (it, ip) in enumerate(zip(order_true, order_pred)):
        mark = "  *" if it == ip else ""
        print(f"{r:>4} {it:>9d} {tests_perf[it]:>+9.4f} | {ip:>9d} {wins[ip]:>6d}{mark}")
    print("=" * 70)

    # Decision gate.
    if rho >= 0.5:
        print("\nPASS: comparator architecture is sound — Spearman >= 0.5 under "
              "controlled conditions.  Real-data failure must therefore be a "
              "data-volume / seed-noise issue, not a code bug.")
    elif rho >= 0.2:
        print("\nMARGINAL: some signal learned but weak.  Consider tuning lr, "
              "epochs, or hidden_dim before scaling to real seeds.")
    else:
        print("\nFAIL: the architecture could not learn a known function.  "
              "Debug the training loop / pair encoder / loss before touching "
              "real data again.")


if __name__ == "__main__":
    main()
