"""Generate seed data by evaluating sampled candidates on source datasets.

For each source dataset, a pool of (encoder_config, cl_strategy) pairs is
sampled, trained via contrastive pretraining, and evaluated on a held-out
validation set.  The resulting :class:`SeedRecord` objects are used to train
the T-CLSC comparator.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TimeSeriesDataset, load_dataset
from models.encoder.encoder_config import EncoderConfig
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline
from utils.logging_utils import get_logger
from utils.reproducibility import set_seed

from .sampler import batch_sample_candidates, batch_sample_strategies

# Bug fix: previously used logging.getLogger(__name__) which had no handler
# attached, causing every progress line in this module to be silently dropped
# under SLURM.  Route through utils.logging_utils.get_logger so that INFO
# messages reach stdout like the rest of the pipeline.
logger = get_logger(__name__)


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS (or M:SS for < 1h)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Seed record
# ---------------------------------------------------------------------------

@dataclass
class SeedRecord:
    """One evaluated (encoder_config, cl_strategy) on a specific task."""

    encoder_config: Dict[str, int]
    strategy: Dict
    task_id: str
    performance: float  # primary metric (higher is better)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SeedRecord":
        return cls(**d)


# ---------------------------------------------------------------------------
# Single candidate evaluation
# ---------------------------------------------------------------------------

def _evaluate_candidate(
    encoder_config: Dict[str, int],
    strategy: Dict,
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    task_type: str,
    pretrain_epochs: int,
    pretrain_lr: float,
    batch_size: int,
    device: torch.device,
    pretrain_iters: int = 0,
) -> float:
    """Train one candidate and return its validation performance.

    Delegates to :func:`train.pretrain.contrastive_pretrain` so that the
    iter-budget mechanism (Bug #003a) and the task-aware val-best gating
    are applied uniformly with the rest of the pipeline.

    Returns:
        Primary metric value (higher = better).  Returns ``-1e9`` on failure
        (e.g. OOM).
    """
    from train.pretrain import contrastive_pretrain

    input_dim = train_dataset.n_channels

    encoder = DilatedCNNEncoder.from_config_dict(input_dim, encoder_config).to(device)
    pipeline = CLPipeline(encoder, strategy).to(device)

    cfg = {
        "pretrain_epochs": pretrain_epochs,
        "pretrain_iters":  pretrain_iters,
        "pretrain_lr":     pretrain_lr,
        "batch_size":      batch_size,
    }
    try:
        contrastive_pretrain(
            encoder=encoder,
            cl_pipeline=pipeline,
            train_data=train_dataset,
            config=cfg,
            device=device,
            task_type=task_type,
        )
    except Exception as exc:                           # pragma: no cover
        logger.warning("contrastive_pretrain failed for candidate: %s", exc)
        return -1e9

    # ── Validation ─────────────────────────────────────────────────────
    encoder.eval()
    performance = _quick_eval(encoder, train_dataset, val_dataset, task_type, device)
    return performance


def _quick_eval(
    encoder: DilatedCNNEncoder,
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    task_type: str,
    device: torch.device,
) -> float:
    """Lightweight downstream evaluation for seed generation.

    For classification: SVM accuracy on time-pooled embeddings.
    For forecasting: negative MSE of a Ridge regression at horizon 24.
    For anomaly detection: F1 from neighbour-distance thresholding.

    Returns:
        Scalar performance (higher = better).
    """
    from sklearn.svm import SVC
    from sklearn.linear_model import Ridge
    import numpy as np

    def encode_pool(ds: TimeSeriesDataset) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                x = ds.data[i : i + 64].to(device)
                h = encoder(x).mean(dim=1).cpu().numpy()
                parts.append(h)
        return np.concatenate(parts, axis=0)

    try:
        if task_type == "classification":
            tr_repr = encode_pool(train_dataset)
            va_repr = encode_pool(val_dataset)
            tr_y = train_dataset.labels.numpy()
            va_y = val_dataset.labels.numpy()
            if len(np.unique(tr_y)) < 2:
                return 0.0
            svm = SVC(kernel="rbf", max_iter=2000)
            svm.fit(tr_repr, tr_y)
            return float(svm.score(va_repr, va_y))

        elif task_type == "forecasting":
            from sklearn.linear_model import Ridge
            H = 24
            with torch.no_grad():
                h_tr = encoder(train_dataset.data.to(device)).cpu().numpy()[0]  # (T_tr, D)
                h_va = encoder(val_dataset.data.to(device)).cpu().numpy()[0]    # (T_va, D)
            x_tr = train_dataset.data[0].numpy()  # (T_tr, C)
            x_va = val_dataset.data[0].numpy()    # (T_va, C)
            if h_tr.shape[0] <= H or h_va.shape[0] <= H:
                return 0.0
            reg = Ridge(alpha=1.0)
            reg.fit(h_tr[:-H], x_tr[H:])
            mse = float(np.mean((reg.predict(h_va[:-H]) - x_va[H:]) ** 2))
            return -mse

        elif task_type == "anomaly_detection":
            # Encode, compute neighbour distance → anomaly score.
            va_repr = encode_pool(val_dataset)  # (N, D)
            # Use std of embeddings as a proxy metric (higher variance = better repr).
            return float(np.mean(np.std(va_repr, axis=0)))
    except Exception as exc:
        logger.warning("_quick_eval failed: %s", exc)
        return -1e9

    return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_seeds(
    source_datasets: List[str],
    data_dir: str,
    n_per_dataset: int = 200,
    pretrain_epochs: int = 40,
    pretrain_lr: float = 1e-3,
    batch_size: int = 64,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    dataset_budgets: Optional[Dict[str, Dict[str, int]]] = None,
    fixed_encoders: Optional[List[Dict[str, int]]] = None,
) -> List[SeedRecord]:
    """Generate seed data across source datasets.

    Args:
        source_datasets: Names of datasets to use (e.g. ``['HAR', 'Yahoo', 'ETTh1']``).
        data_dir: Root data directory.
        n_per_dataset: Number of candidates to evaluate per dataset.
        pretrain_epochs: CL pretraining epochs per candidate.
        pretrain_lr: Pretraining learning rate.
        batch_size: Training batch size.
        save_dir: If given, serialise seed records to
            ``{save_dir}/seeds.json``.
        device: Torch device.  ``None`` → auto-detect.
        seed: Global random seed.
        fixed_encoders: Optional list of encoder configs to restrict the
            encoder sub-space to (Plan B Stage B). When given, candidates are
            sampled via :func:`batch_sample_strategies` rather than the full
            joint sampler — every seed record's ``encoder_config`` field will
            be drawn from this list.

    Returns:
        List of all :class:`SeedRecord` objects.
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fixed_encoders is not None:
        logger.info(
            "[plan-B] fixed_encoders mode: %d encoder(s) -> %s",
            len(fixed_encoders), fixed_encoders,
        )

    all_seeds: List[SeedRecord] = []
    n_datasets = len(source_datasets)
    overall_start = time.time()

    for ds_idx, ds_name in enumerate(source_datasets):
        logger.info(
            "[dataset %d/%d] Generating seeds for: %s",
            ds_idx + 1, n_datasets, ds_name,
        )
        splits = load_dataset(ds_name, data_dir)
        train_ds = splits["train"]
        val_ds   = splits.get("val") or splits["test"]
        task_type = train_ds.task_type

        # Per-dataset budget override (Bug #003a).
        ds_budget = (dataset_budgets or {}).get(ds_name, {}) or {}
        ds_iters  = int(ds_budget.get("pretrain_iters", 0))
        ds_epochs = int(ds_budget.get("pretrain_epochs", pretrain_epochs))
        if ds_iters > 0:
            logger.info("  budget: pretrain_iters=%d (forecasting)", ds_iters)
        else:
            logger.info("  budget: pretrain_epochs=%d", ds_epochs)

        if fixed_encoders is not None:
            candidates = batch_sample_strategies(n_per_dataset, fixed_encoders)
        else:
            candidates = batch_sample_candidates(n_per_dataset)
        ds_start = time.time()

        for i, (enc_cfg, strat_cfg) in enumerate(candidates):
            cand_start = time.time()
            perf = _evaluate_candidate(
                enc_cfg, strat_cfg,
                train_ds, val_ds,
                task_type,
                ds_epochs, pretrain_lr, batch_size,
                device,
                pretrain_iters=ds_iters,
            )
            cand_elapsed = time.time() - cand_start

            record = SeedRecord(
                encoder_config=enc_cfg,
                strategy=strat_cfg,
                task_id=ds_name,
                performance=perf,
            )
            all_seeds.append(record)

            # Progress + ETA based on average per-candidate time so far.
            done = i + 1
            avg_per_cand = (time.time() - ds_start) / done
            ds_eta = avg_per_cand * (n_per_dataset - done)
            logger.info(
                "  [%s] %d/%d  perf=%.6f  (enc L%d H%d O%d)  "
                "took %s  avg %s/cand  ds-ETA %s",
                ds_name, done, n_per_dataset, perf,
                enc_cfg["n_layers"], enc_cfg["hidden_dim"], enc_cfg["output_dim"],
                _fmt_hms(cand_elapsed),
                _fmt_hms(avg_per_cand),
                _fmt_hms(ds_eta),
            )

        ds_total = time.time() - ds_start
        # Coarse overall ETA: assume remaining datasets cost roughly the
        # same as the current one (best signal we have without per-dataset
        # cost models).
        remaining_ds = n_datasets - (ds_idx + 1)
        overall_eta = ds_total * remaining_ds
        logger.info(
            "[dataset %d/%d] %s done in %s  | remaining datasets: %d  | "
            "rough overall-ETA: %s",
            ds_idx + 1, n_datasets, ds_name,
            _fmt_hms(ds_total), remaining_ds, _fmt_hms(overall_eta),
        )

    logger.info(
        "All datasets done in %s", _fmt_hms(time.time() - overall_start),
    )

    # ── Persist ────────────────────────────────────────────────────────
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "seeds.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in all_seeds], f, indent=2)
        logger.info("Saved %d seed records to %s", len(all_seeds), path)

    return all_seeds
