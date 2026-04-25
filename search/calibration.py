"""Noisy/clean seed calibration test for ZeroAutoCL.

Implements Stage A+B of the calibration plan:

  Stage A — train N CL strategies × K seeds × E max_epochs on ONE source
            dataset, with eval_every=1 so we capture the full per-epoch
            val_score trajectory.  ``gold[s] = max(val_score)`` over the
            full trajectory; ``σ_gold[s] = std`` across seeds.

  Stage B — derive noisy estimates at multiple budgets B from the
            captured trajectory (no extra training):
              • mode-A: val_score at epoch B (naive truncation)
              • mode-B: max(val_score over [1..B]) — the best-of-N
                        estimator that ``seed_generator.mode='noisy'``
                        currently uses.

The captured trajectory is persisted to JSON; analysis is a separate
step that consumes the JSON and produces:
  - Spearman ρ between (mean-seed noisy) and gold, per (mode, budget)
  - Kendall τ
  - Top-K recall
  - Pairwise concordance @ ``gap >= valid_gap_threshold`` — the metric
    that directly maps onto comparator BCE accuracy.
  - Per-strategy CV (std/|mean| across seeds) at each budget.

The point of this calibration is to answer: "with the best-of-N + gap
filtering already in the ZeroAutoCL pipeline, does a small noisy budget
preserve enough pairwise sign information to train T-CLSC?"  See the
plan in chat ``2026-04-25`` for context.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data.dataset import TimeSeriesDataset, load_dataset
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

from .sampler import batch_sample_strategies

logger = get_logger(__name__)


DEFAULT_ENCODER: Dict[str, int] = {"n_layers": 10, "hidden_dim": 64, "output_dim": 320}
DEFAULT_BUDGETS: Tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20, 30, 50, 80)


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Stage A: Trajectory capture
# ---------------------------------------------------------------------------

def capture_trajectories(
    source_dataset: str,
    data_dir: str,
    n_strategies: int = 16,
    n_seeds: int = 1,
    max_epochs: int = 20,
    fixed_encoder: Optional[Dict[str, int]] = None,
    pretrain_lr: float = 1e-3,
    batch_size: int = 32,
    crop_len: Optional[int] = 1024,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Train N strategies × K seeds and persist their full val trajectories.

    Args:
        source_dataset: Single source dataset name (e.g. ``"ETTh2"``).
        data_dir: Path to dataset root.
        n_strategies: Number of CL strategies to sample.
        n_seeds: Number of independent seed runs per strategy.
        max_epochs: Training epoch cap (also serves as the gold-budget).
        fixed_encoder: Encoder config dict. ``None`` → default 10/64/320.
        pretrain_lr: AdamW/Adam learning rate.
        batch_size: Training batch size.
        crop_len: Sliding-window crop for forecasting; ``None`` keeps the
            dataset default (3000 — too slow at v1 scale).
        save_dir: If given, writes ``trajectories.json`` here.
        device: Torch device.  ``None`` → auto.
        seed: Base seed; per-(strategy, seed_idx) seeds are derived as
            ``seed + s_idx*1000 + k_idx``.
        horizons: Forecasting horizons for val eval.  ``None`` →
            ``[24, 48, 168, 336, 720]`` (canonical TS2Vec set).

    Returns:
        A dict with the same shape as the persisted JSON.
    """
    from train.pretrain import contrastive_pretrain

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fixed_encoder is None:
        fixed_encoder = dict(DEFAULT_ENCODER)

    set_seed(seed)
    splits = load_dataset(source_dataset, data_dir, window_len_override=crop_len)
    train_data = splits["train"]
    val_data = splits.get("val") or splits.get("test")
    if val_data is None:
        raise ValueError(
            f"{source_dataset!r} has no val/test split — calibration needs val data."
        )
    task_type = train_data.task_type
    input_dim = train_data.n_channels

    logger.info(
        "[calibration] source=%s  task=%s  input_dim=%d  "
        "train=%d  val=%d  encoder=%s  budget cap=%d epochs",
        source_dataset, task_type, input_dim,
        len(train_data), len(val_data), fixed_encoder, max_epochs,
    )

    # Sample strategies once — every strategy is paired with the SAME fixed
    # encoder, so the only varying axis is the CL strategy.
    candidates = batch_sample_strategies(n_strategies, [fixed_encoder])
    logger.info(
        "[calibration] sampled %d strategies × %d seed(s) = %d total runs",
        n_strategies, n_seeds, n_strategies * n_seeds,
    )

    overall_start = time.time()
    strategies_out: List[Dict[str, Any]] = []
    total_runs = n_strategies * n_seeds
    run_idx = 0

    for s_idx, (enc_cfg, strat_cfg) in enumerate(candidates):
        traj_per_seed: List[Dict[str, Any]] = []

        for k_idx in range(n_seeds):
            run_idx += 1
            run_seed = seed + s_idx * 1000 + k_idx
            set_seed(run_seed)

            run_start = time.time()
            encoder = DilatedCNNEncoder.from_config_dict(input_dim, enc_cfg).to(device)
            pipeline = CLPipeline(encoder, strat_cfg).to(device)

            cfg: Dict[str, Any] = {
                "pretrain_epochs": max_epochs,
                "pretrain_iters":  0,           # epoch-mode is required for trajectory comparability
                "pretrain_lr":     pretrain_lr,
                "batch_size":      batch_size,
                "eval_every":      1,           # capture per-epoch val_score
                "val_best":        False,       # do NOT restore best — we want the raw trajectory
            }

            history: List[Dict[str, Any]] = []
            try:
                contrastive_pretrain(
                    encoder=encoder,
                    cl_pipeline=pipeline,
                    train_data=train_data,
                    config=cfg,
                    device=device,
                    task_type=task_type,
                    val_data=val_data,
                    horizons=horizons,
                    history=history,
                )
                ok = True
            except Exception as exc:                         # pragma: no cover
                logger.warning(
                    "[calibration] run %d/%d crashed: %s",
                    run_idx, total_runs, exc,
                )
                ok = False

            run_elapsed = time.time() - run_start
            losses = [h.get("loss", float("nan")) for h in history]
            val_scores = [h.get("val_score") for h in history]
            # Replace None with NaN for clean numeric handling downstream.
            val_scores = [
                float("nan") if v is None else float(v) for v in val_scores
            ]

            traj_per_seed.append({
                "seed":            run_seed,
                "n_epochs":        len(history),
                "loss":            losses,
                "val_score":       val_scores,
                "ok":              ok,
                "wall_clock_sec":  run_elapsed,
            })

            avg_per_run = (time.time() - overall_start) / run_idx
            eta = avg_per_run * (total_runs - run_idx)
            best_val = max(
                (v for v in val_scores if not math.isnan(v)),
                default=float("nan"),
            )
            logger.info(
                "[calibration] run %d/%d  strat=%d  seed=%d  "
                "epochs=%d  best_val=%.4f  took %s  ETA %s",
                run_idx, total_runs, s_idx, k_idx,
                len(history), best_val,
                _fmt_hms(run_elapsed), _fmt_hms(eta),
            )

            del encoder, pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        strategies_out.append({
            "strategy_id":    s_idx,
            "encoder_config": enc_cfg,
            "strategy":       strat_cfg,
            "trajectories":   traj_per_seed,
        })

    elapsed = time.time() - overall_start
    logger.info(
        "[calibration] capture done in %s  (%d runs)",
        _fmt_hms(elapsed), total_runs,
    )

    out: Dict[str, Any] = {
        "meta": {
            "source_dataset":   source_dataset,
            "task_type":        task_type,
            "n_strategies":     n_strategies,
            "n_seeds":          n_seeds,
            "max_epochs":       max_epochs,
            "fixed_encoder":    fixed_encoder,
            "pretrain_lr":      pretrain_lr,
            "batch_size":       batch_size,
            "crop_len":         crop_len,
            "horizons":         horizons,
            "base_seed":        seed,
            "wall_clock_sec":   elapsed,
        },
        "strategies": strategies_out,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "trajectories.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info("[calibration] wrote %s", path)

    return out


# ---------------------------------------------------------------------------
# Stage B: Metric analysis
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; returns NaN if either vector is degenerate."""
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, b)
        return float(rho)
    except ImportError:                                # pragma: no cover
        # Manual fallback: rank-and-pearson.
        ar = np.argsort(np.argsort(a))
        br = np.argsort(np.argsort(b))
        ar = ar - ar.mean(); br = br - br.mean()
        denom = (np.linalg.norm(ar) * np.linalg.norm(br))
        return float(ar @ br / denom) if denom > 0 else float("nan")


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    try:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(a, b)
        return float(tau) if not np.isnan(tau) else float("nan")
    except ImportError:                                # pragma: no cover
        return float("nan")


def _topk_recall(noisy: np.ndarray, gold: np.ndarray, k: int) -> float:
    """Fraction of top-K (by gold) strategies that the noisy ranking also ranks top-K."""
    if k >= noisy.size:
        return 1.0
    top_gold = set(np.argsort(-gold)[:k].tolist())
    top_noisy = set(np.argsort(-noisy)[:k].tolist())
    return len(top_gold & top_noisy) / k


def _pairwise_concordance(
    noisy: np.ndarray, gold: np.ndarray, gap_threshold: float,
) -> Tuple[float, int]:
    """For each pair (i, j) with |gold[i] - gold[j]| ≥ gap_threshold,
    compute the fraction where sign(noisy[i] - noisy[j]) matches sign(gold[i] - gold[j]).

    Returns ``(concordance, n_pairs)``.  This is the metric that directly
    maps onto T-CLSC's pairwise BCE accuracy under the same gap filter.
    """
    n = noisy.size
    matches = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            gd = gold[i] - gold[j]
            if abs(gd) < gap_threshold:
                continue
            nd = noisy[i] - noisy[j]
            if nd == 0:
                # Treat ties as half-credit (rare but possible at coarse budgets).
                matches += 0.5
            elif (nd > 0) == (gd > 0):
                matches += 1
            total += 1
    if total == 0:
        return float("nan"), 0
    return matches / total, total


def _seed_aggregate(
    trajectories: List[Dict[str, Any]],
    max_epochs: int,
) -> np.ndarray:
    """Stack per-seed val_score into a (K, max_epochs) array, NaN-padded.

    NaN epochs (failed runs or eval skipped) propagate so downstream
    nanmean / nanmax handles them correctly.
    """
    K = len(trajectories)
    out = np.full((K, max_epochs), np.nan, dtype=np.float64)
    for k, t in enumerate(trajectories):
        vs = t.get("val_score", [])
        for e, v in enumerate(vs[:max_epochs]):
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[k, e] = v
    return out


def analyze_trajectories(
    traj: Dict[str, Any],
    budgets: Optional[List[int]] = None,
    gap_threshold: float = 0.02,
    topk_values: Tuple[int, ...] = (3, 5, 10),
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute calibration metrics from a captured trajectory bundle.

    Args:
        traj: Output of :func:`capture_trajectories` (or its persisted JSON).
        budgets: Budgets B to evaluate noisy estimators at.  ``None`` →
            :data:`DEFAULT_BUDGETS` (clamped to ≤ ``meta.max_epochs``).
        gap_threshold: Minimum |Δgold| for a pair to enter the pairwise
            concordance count.  Mirrors the comparator's
            ``valid_gap_threshold`` (default 0.02 in the YAML).
        topk_values: Top-K cut-offs to report recall at.
        save_dir: If given, writes ``calibration_metrics.json``,
            ``calibration_report.md``, and ``calibration_curve.png``
            (matplotlib optional).

    Returns:
        Dict with ``per_budget`` metrics keyed by ``f"B{B}_{mode}"`` and
        per-strategy summary.
    """
    meta = traj["meta"]
    max_epochs = int(meta["max_epochs"])
    if budgets is None:
        budgets = [b for b in DEFAULT_BUDGETS if b <= max_epochs]
        if max_epochs not in budgets:
            budgets.append(max_epochs)
    else:
        budgets = sorted(set(b for b in budgets if 1 <= b <= max_epochs))

    strategies = traj["strategies"]
    N = len(strategies)
    K = int(meta["n_seeds"])
    logger.info(
        "[calibration] analyzing N=%d strategies, K=%d seeds, max_epochs=%d, "
        "budgets=%s, gap_threshold=%.4f",
        N, K, max_epochs, budgets, gap_threshold,
    )

    # Build (N, K, max_epochs) value tensor.
    val = np.full((N, K, max_epochs), np.nan, dtype=np.float64)
    for s_idx, s in enumerate(strategies):
        per_seed = _seed_aggregate(s["trajectories"], max_epochs)
        val[s_idx, : per_seed.shape[0], :] = per_seed

    # Gold = nan-mean over seeds of (nan-max over the FULL trajectory).
    with np.errstate(all="ignore"):
        gold_per_seed = np.nanmax(val, axis=2)               # (N, K)
        gold = np.nanmean(gold_per_seed, axis=1)             # (N,)
        gold_std = np.nanstd(gold_per_seed, axis=1)          # (N,)

    # Drop strategies whose gold is NaN (all seeds failed).
    valid_mask = ~np.isnan(gold)
    if not valid_mask.all():
        n_drop = int((~valid_mask).sum())
        logger.warning("[calibration] dropping %d strategies with NaN gold", n_drop)

    n_valid = int(valid_mask.sum())
    if n_valid < 2:
        raise RuntimeError(
            f"[calibration] only {n_valid} usable strategies — need ≥ 2 to "
            "compute any rank metric.  Re-run with more strategies or check "
            "the trajectory log for crashes.",
        )

    gold_v = gold[valid_mask]
    per_budget: Dict[str, Dict[str, Any]] = {}

    for B in budgets:
        # Mode A: val_score at exactly epoch B (1-indexed → idx B-1).
        with np.errstate(all="ignore"):
            mode_a = np.nanmean(val[:, :, B - 1], axis=1)            # (N,)
            # Mode B: max(val_score over [1..B]).
            mode_b = np.nanmean(np.nanmax(val[:, :, :B], axis=2), axis=1)

        # Per-strategy CV across seeds at budget B (mode B is the one that
        # the seed_generator actually uses, so we report CV for it).
        with np.errstate(all="ignore"):
            mode_b_per_seed = np.nanmax(val[:, :, :B], axis=2)       # (N, K)
            mu = np.nanmean(mode_b_per_seed, axis=1)
            sd = np.nanstd(mode_b_per_seed, axis=1)
            cv = np.where(np.abs(mu) > 1e-12, sd / np.abs(mu), np.nan)

        for label, est in (("modeA_naive", mode_a), ("modeB_bestofN", mode_b)):
            est_v = est[valid_mask]
            # Drop pairs where the noisy estimator itself is NaN.
            both = ~np.isnan(est_v)
            if both.sum() < 2:
                rho = tau = float("nan")
                conc = float("nan"); n_pairs = 0
                tk = {f"top{k}": float("nan") for k in topk_values}
            else:
                est_vv = est_v[both]
                gold_vv = gold_v[both]
                rho = _spearman(est_vv, gold_vv)
                tau = _kendall(est_vv, gold_vv)
                conc, n_pairs = _pairwise_concordance(est_vv, gold_vv, gap_threshold)
                tk = {
                    f"top{k}": _topk_recall(est_vv, gold_vv, k) for k in topk_values
                }

            per_budget[f"B{B}_{label}"] = {
                "budget":               B,
                "mode":                 label,
                "spearman":             rho,
                "kendall":              tau,
                "pairwise_concordance": conc,
                "n_pairs_used":         n_pairs,
                "topk_recall":          tk,
                "mean_cv_modeB":        float(np.nanmean(cv[valid_mask])),
                "n_strategies_used":    int(both.sum()),
            }

    summary = {
        "meta":         meta,
        "budgets":      budgets,
        "gap_threshold": gap_threshold,
        "n_strategies": N,
        "n_strategies_valid": n_valid,
        "gold":         {
            "mean":         gold.tolist(),
            "std_per_strategy": gold_std.tolist(),
        },
        "per_budget":   per_budget,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "calibration_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        _write_report(summary, save_dir)
        try:
            _plot_curve(summary, save_dir)
        except Exception as exc:                              # pragma: no cover
            logger.warning("[calibration] plot failed: %s", exc)

    return summary


def _write_report(summary: Dict[str, Any], save_dir: str) -> None:
    meta = summary["meta"]
    budgets = summary["budgets"]
    pb = summary["per_budget"]
    lines: List[str] = []
    lines.append("# ZeroAutoCL — Noisy/Clean Seed Calibration Report\n")
    lines.append(
        f"- Source dataset: **{meta['source_dataset']}** "
        f"(task: {meta['task_type']})",
    )
    lines.append(
        f"- N strategies: {summary['n_strategies']} (valid: {summary['n_strategies_valid']})  "
        f"| K seeds: {meta['n_seeds']}  | max_epochs: {meta['max_epochs']}",
    )
    lines.append(
        f"- Encoder: {meta['fixed_encoder']}  | crop_len: {meta['crop_len']}  "
        f"| batch_size: {meta['batch_size']}  | lr: {meta['pretrain_lr']}",
    )
    lines.append(
        f"- gap_threshold for pairwise concordance: **{summary['gap_threshold']}**",
    )
    lines.append(
        f"- Wall clock: {_fmt_hms(meta['wall_clock_sec'])}\n",
    )
    lines.append(
        "Gold = ``mean_seeds( max_epoch( val_score ) )`` over the full "
        "trajectory.  Each row below uses a noisy estimator at budget B "
        "and reports its agreement with gold across the strategy pool.\n",
    )

    # Mode B (best-of-N) is what seed_generator.mode='noisy' actually uses.
    lines.append("## Mode B (best-of-N, current `mode='noisy'`)\n")
    lines.append(
        "| Budget | Spearman ρ | Kendall τ | Pairwise concordance @ gap "
        "| #pairs | Top-3 recall | Top-5 recall | Top-10 recall | Mean CV |",
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for B in budgets:
        r = pb.get(f"B{B}_modeB_bestofN", {})
        lines.append(
            "| {B} | {rho} | {tau} | {conc} | {n} | {t3} | {t5} | {t10} | {cv} |".format(
                B=B,
                rho=_fmt(r.get("spearman")),
                tau=_fmt(r.get("kendall")),
                conc=_fmt(r.get("pairwise_concordance")),
                n=r.get("n_pairs_used", 0),
                t3=_fmt(r.get("topk_recall", {}).get("top3")),
                t5=_fmt(r.get("topk_recall", {}).get("top5")),
                t10=_fmt(r.get("topk_recall", {}).get("top10")),
                cv=_fmt(r.get("mean_cv_modeB")),
            ),
        )

    lines.append("\n## Mode A (naive truncation)\n")
    lines.append(
        "| Budget | Spearman ρ | Kendall τ | Pairwise concordance @ gap "
        "| #pairs | Top-3 recall | Top-5 recall | Top-10 recall |",
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for B in budgets:
        r = pb.get(f"B{B}_modeA_naive", {})
        lines.append(
            "| {B} | {rho} | {tau} | {conc} | {n} | {t3} | {t5} | {t10} |".format(
                B=B,
                rho=_fmt(r.get("spearman")),
                tau=_fmt(r.get("kendall")),
                conc=_fmt(r.get("pairwise_concordance")),
                n=r.get("n_pairs_used", 0),
                t3=_fmt(r.get("topk_recall", {}).get("top3")),
                t5=_fmt(r.get("topk_recall", {}).get("top5")),
                t10=_fmt(r.get("topk_recall", {}).get("top10")),
            ),
        )

    lines.append("\n## Decision thresholds (from plan)\n")
    lines.append(
        "- Spearman ρ ≥ 0.6 — minimum for noisy to be useful overall\n"
        "- Pairwise concordance ≥ 0.75 @ gap=valid_gap_threshold — directly "
        "bounds T-CLSC BCE accuracy from above\n"
        "- Top-10 recall ≥ 0.5 — filters usefully for zero-shot top-K\n"
        "- Mean CV ≤ 0.15 — single run is reliable enough\n",
    )

    path = os.path.join(save_dir, "calibration_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("[calibration] wrote %s", path)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return f"{x:.3f}"


def _plot_curve(summary: Dict[str, Any], save_dir: str) -> None:
    """Plot Spearman ρ + pairwise concordance vs budget for both modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = summary["budgets"]
    pb = summary["per_budget"]

    def collect(metric: str, mode: str) -> List[float]:
        return [
            pb.get(f"B{B}_{mode}", {}).get(metric, float("nan")) for B in budgets
        ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(budgets, collect("spearman", "modeB_bestofN"),
            "o-", label="Mode B (best-of-N)")
    ax.plot(budgets, collect("spearman", "modeA_naive"),
            "s--", label="Mode A (naive)")
    ax.axhline(0.6, color="gray", linestyle=":", linewidth=1, label="ρ=0.6 threshold")
    ax.set_xlabel("Noisy budget (epochs)")
    ax.set_ylabel("Spearman ρ vs gold")
    ax.set_title("Rank correlation vs noisy budget")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    ax.plot(budgets, collect("pairwise_concordance", "modeB_bestofN"),
            "o-", label="Mode B (best-of-N)")
    ax.plot(budgets, collect("pairwise_concordance", "modeA_naive"),
            "s--", label="Mode A (naive)")
    ax.axhline(0.75, color="gray", linestyle=":", linewidth=1,
               label="conc=0.75 threshold")
    gap = summary["gap_threshold"]
    ax.set_xlabel("Noisy budget (epochs)")
    ax.set_ylabel(f"Pairwise concordance @ gap={gap}")
    ax.set_title("Comparator-relevant pairwise sign agreement")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(save_dir, "calibration_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("[calibration] wrote %s", path)
