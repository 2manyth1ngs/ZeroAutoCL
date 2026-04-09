"""Zero-shot search: rank candidates with T-CLSC and validate top-K.

Pipeline
--------
1. Extract task features for the target dataset.
2. Sample a large pool of candidates (default 300 000).
3. Swiss-system tournament ranking using the pretrained T-CLSC.
4. Fully train & evaluate the top-K candidates.
5. Return the best (encoder_config, cl_strategy) and its performance.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from models.comparator.t_clsc import TCLSC
from models.comparator.task_feature import TaskFeatureExtractor
from data.dataset import TimeSeriesDataset, load_dataset
from .sampler import batch_sample_candidates, batch_sample_strategies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Swiss-system tournament
# ---------------------------------------------------------------------------

def tournament_rank(
    comparator: TCLSC,
    candidates: List[Tuple[Dict[str, int], Dict]],
    task_features: Tensor,
    rounds: int = 15,
    batch_size: int = 4096,
) -> List[int]:
    """Rank candidates via a Swiss-system tournament.

    Each round:
      1. Sort candidates by current score (descending).
      2. Pair adjacent entries (1v2, 3v4, …).
      3. Batch-evaluate all pairs through the comparator.
      4. Winner of each match gets +1 score.

    Complexity: O(n_candidates × rounds) comparisons, much less than
    the O(n²) of an all-pairs comparison.

    Args:
        comparator: Pretrained :class:`TCLSC`.
        candidates: List of ``(encoder_config, strategy_config)`` tuples.
        task_features: Task feature vector, shape ``(D_task,)``.
        rounds: Number of tournament rounds (default 15).
        batch_size: Maximum number of pairs to evaluate in one forward
            pass through the comparator.

    Returns:
        List of candidate indices sorted by descending score.
    """
    n = len(candidates)
    if n <= 1:
        return list(range(n))

    scores = [0] * n
    device = task_features.device

    comparator.eval()

    for r in range(rounds):
        # Sort by score (descending); ties broken by index for stability.
        sorted_idx = sorted(range(n), key=lambda i: (-scores[i], i))

        # Pair adjacent indices.
        pairs_a_idx: List[int] = []
        pairs_b_idx: List[int] = []
        for i in range(0, n - 1, 2):
            pairs_a_idx.append(sorted_idx[i])
            pairs_b_idx.append(sorted_idx[i + 1])

        n_matches = len(pairs_a_idx)
        if n_matches == 0:
            break

        # Evaluate in chunks for memory safety.
        with torch.no_grad():
            for bs_start in range(0, n_matches, batch_size):
                bs_end = min(bs_start + batch_size, n_matches)
                a_indices = pairs_a_idx[bs_start:bs_end]
                b_indices = pairs_b_idx[bs_start:bs_end]

                enc_a  = [candidates[i][0] for i in a_indices]
                str_a  = [candidates[i][1] for i in a_indices]
                enc_b  = [candidates[i][0] for i in b_indices]
                str_b  = [candidates[i][1] for i in b_indices]

                probs = comparator.forward_batch(
                    enc_a, str_a, enc_b, str_b, task_features,
                )  # (chunk,)

                for j, (ia, ib) in enumerate(zip(a_indices, b_indices)):
                    if probs[j].item() > 0.5:
                        scores[ia] += 1
                    else:
                        scores[ib] += 1

        logger.debug("Tournament round %d/%d complete", r + 1, rounds)

    # Final ranking by score.
    return sorted(range(n), key=lambda i: (-scores[i], i))


# ---------------------------------------------------------------------------
# Full zero-shot search pipeline
# ---------------------------------------------------------------------------

def zero_shot_search(
    target_dataset: str,
    data_dir: str,
    comparator: TCLSC,
    task_feature_extractor: Optional[TaskFeatureExtractor] = None,
    n_candidates: int = 300_000,
    top_k: int = 10,
    tournament_rounds: int = 15,
    pretrain_epochs: int = 40,
    pretrain_iters: int = 0,
    pretrain_lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, int], Dict, float]:
    """Run the full zero-shot search on a target dataset.

    Args:
        target_dataset: Name of the target dataset (e.g. ``'Epilepsy'``).
        data_dir: Root data directory.
        comparator: Pretrained :class:`TCLSC`.
        task_feature_extractor: If ``None``, a default one is created.
        n_candidates: Number of candidates to sample.
        top_k: Number of top candidates to fully evaluate.
        tournament_rounds: Rounds for Swiss-system ranking.
        pretrain_epochs: Epochs for full CL pretraining of top-K.
        pretrain_lr: Learning rate for full CL pretraining.
        batch_size: Training batch size.
        device: Torch device.

    Returns:
        Tuple of ``(best_encoder_config, best_strategy, best_performance)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparator = comparator.to(device)

    # ── Step 1: Task features ─────────────────────────────────────────
    logger.info("Loading target dataset: %s", target_dataset)
    splits = load_dataset(target_dataset, data_dir)
    train_ds = splits["train"]
    val_ds   = splits["val"]
    task_type = train_ds.task_type

    if task_feature_extractor is None:
        task_feature_extractor = TaskFeatureExtractor(device=device)

    task_feat = task_feature_extractor.extract(train_ds, task_type).to(device)
    logger.info("Task features extracted: dim=%d", task_feat.shape[0])

    # ── Step 2: Sample candidates ─────────────────────────────────────
    logger.info("Sampling %d candidates", n_candidates)
    candidates = batch_sample_candidates(n_candidates)

    # ── Step 3: Tournament ranking ────────────────────────────────────
    logger.info("Running Swiss-system tournament (%d rounds)", tournament_rounds)
    ranking = tournament_rank(
        comparator, candidates, task_feat,
        rounds=tournament_rounds,
    )
    top_indices = ranking[:top_k]
    logger.info("Top-%d candidates selected", top_k)

    # ── Step 4-5: Full CL pretrain + validation for top-K ─────────────
    from models.encoder.dilated_cnn import DilatedCNNEncoder
    from models.contrastive.cl_pipeline import CLPipeline
    from train.pretrain import contrastive_pretrain
    from train.evaluate import evaluate as full_evaluate

    pretrain_cfg = {
        "pretrain_epochs": pretrain_epochs,
        "pretrain_iters":  pretrain_iters,
        "pretrain_lr":     pretrain_lr,
        "batch_size":      batch_size,
    }
    if pretrain_iters > 0:
        logger.info("Top-K full train budget: %d iters", pretrain_iters)
    else:
        logger.info("Top-K full train budget: %d epochs", pretrain_epochs)

    best_enc: Dict[str, int] = {}
    best_strat: Dict = {}
    best_perf = -float("inf")

    for rank, idx in enumerate(top_indices):
        enc_cfg, strat_cfg = candidates[idx]
        logger.info(
            "Evaluating top-%d candidate (rank %d/%d)",
            rank + 1, rank + 1, top_k,
        )
        try:
            input_dim = train_ds.n_channels
            encoder = DilatedCNNEncoder.from_config_dict(input_dim, enc_cfg).to(device)
            pipeline = CLPipeline(encoder, strat_cfg).to(device)
            encoder = contrastive_pretrain(
                encoder, pipeline, train_ds, pretrain_cfg, device,
                task_type=task_type,
            )
            metrics = full_evaluate(encoder, train_ds, val_ds, task_type, device=device)

            # Extract primary metric (higher = better).
            if task_type == "classification":
                perf = float(metrics["acc"])
            elif task_type == "forecasting":
                # Average negative MSE across all horizons.
                mses = [v["mse"] for v in metrics.values()]
                perf = -float(sum(mses) / len(mses)) if mses else -1e9
            elif task_type == "anomaly_detection":
                perf = float(metrics["f1"])
            else:
                perf = -1e9
        except Exception as exc:
            logger.warning("Candidate %d failed: %s", rank + 1, exc)
            perf = -1e9

        logger.info("  performance=%.6f", perf)

        if perf > best_perf:
            best_perf = perf
            best_enc = enc_cfg
            best_strat = strat_cfg

    logger.info(
        "Best candidate: perf=%.6f  encoder=%s",
        best_perf, best_enc,
    )
    return best_enc, best_strat, best_perf


# ---------------------------------------------------------------------------
# Plan B: ranking-only entry point that returns Cartesian finalists
# ---------------------------------------------------------------------------

def rank_finalists(
    target_dataset: str,
    data_dir: str,
    comparator: TCLSC,
    fixed_encoders: List[Dict[str, int]],
    task_feature_extractor: Optional[TaskFeatureExtractor] = None,
    n_candidates: int = 200_000,
    top_k_strategies: int = 5,
    tournament_rounds: int = 25,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Plan B: rank CL strategies for each fixed encoder, return finalists.

    Unlike :func:`zero_shot_search`, this function does **not** retrain any
    candidate. It only runs the comparator-based Swiss tournament and
    bucket-orders the result so each ``fixed_encoders`` entry contributes
    exactly ``top_k_strategies`` strategies. The caller (Phase 4) is
    responsible for the actual retrain + test evaluation on the cartesian
    product (``len(fixed_encoders) * top_k_strategies`` configs).

    Args:
        target_dataset: Name of the target dataset.
        data_dir: Root data directory.
        comparator: Pretrained :class:`TCLSC`.
        fixed_encoders: Top-K_enc encoder configs from Stage A.
        task_feature_extractor: Defaults to a fresh :class:`TaskFeatureExtractor`.
        n_candidates: Size of the (encoder, strategy) candidate pool sampled
            via :func:`batch_sample_strategies`.
        top_k_strategies: How many strategies to keep per encoder.
        tournament_rounds: Swiss-system tournament rounds.
        device: Torch device.

    Returns:
        List of finalist records, length ``len(fixed_encoders) * top_k_strategies``::

            [
              {"encoder_config": {...}, "strategy": {...}, "rank": int},
              ...
            ]
        Order is grouped by encoder; ``rank`` is the position within that
        encoder's bucket (0 = best per the comparator).
    """
    if not fixed_encoders:
        raise ValueError("rank_finalists: fixed_encoders must be non-empty")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparator = comparator.to(device)

    logger.info("[plan-B] rank_finalists target=%s", target_dataset)
    logger.info(
        "[plan-B]  K_enc=%d  top_k_strategies=%d  pool=%d  rounds=%d",
        len(fixed_encoders), top_k_strategies, n_candidates, tournament_rounds,
    )

    splits = load_dataset(target_dataset, data_dir)
    train_ds = splits["train"]
    task_type = train_ds.task_type

    if task_feature_extractor is None:
        task_feature_extractor = TaskFeatureExtractor(device=device)
    task_feat = task_feature_extractor.extract(train_ds, task_type).to(device)

    candidates = batch_sample_strategies(n_candidates, fixed_encoders)
    ranking = tournament_rank(
        comparator, candidates, task_feat, rounds=tournament_rounds,
    )

    # Bucket by encoder; keep at most top_k_strategies per bucket.
    import json as _json
    by_enc: Dict[str, List[int]] = {}
    n_enc = len(fixed_encoders)
    for idx in ranking:
        enc_cfg, _ = candidates[idx]
        key = _json.dumps(enc_cfg, sort_keys=True)
        bucket = by_enc.setdefault(key, [])
        if len(bucket) < top_k_strategies:
            bucket.append(idx)
        if len(by_enc) == n_enc and all(
            len(b) >= top_k_strategies for b in by_enc.values()
        ):
            break

    finalists: List[Dict] = []
    for key, bucket in by_enc.items():
        for local_rank, cand_idx in enumerate(bucket):
            enc_cfg, strat_cfg = candidates[cand_idx]
            finalists.append({
                "encoder_config": enc_cfg,
                "strategy": strat_cfg,
                "rank": local_rank,
            })

    logger.info("[plan-B] selected %d finalists", len(finalists))
    return finalists
