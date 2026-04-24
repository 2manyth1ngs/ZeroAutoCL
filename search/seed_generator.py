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
    mode: str = "clean",
) -> float:
    """Train one candidate and return its validation performance.

    Delegates to :func:`train.pretrain.contrastive_pretrain` so that the
    iter-budget mechanism (Bug #003a) and the task-aware val-best gating
    are applied uniformly with the rest of the pipeline.

    Modes (``mode`` parameter) — AutoCTS++-style seed generation:

    - ``"clean"``  (default): the original protocol.  Run full training,
      evaluate the final encoder via :func:`_quick_eval`.  Low noise but
      expensive; used for the comparator's fine-tuning stage.

    - ``"noisy"``: short training (budget set by caller) with a val eval
      after every epoch.  Performance = ``max(val_score_per_epoch)``
      (equivalent to AutoCTS++'s ``min(MAE)`` over the 5 noisy epochs at
      ``reference/AutoCTS_plusplus/exps/random_search.py:285``).  Cheap
      best-of-N estimator that naturally attenuates per-epoch seed noise;
      used for the comparator's pretraining stage.

    Returns:
        Primary metric value (higher = better).  Returns ``-1e9`` on failure
        (e.g. OOM).
    """
    from train.pretrain import contrastive_pretrain

    input_dim = train_dataset.n_channels

    encoder = DilatedCNNEncoder.from_config_dict(input_dim, encoder_config).to(device)
    pipeline = CLPipeline(encoder, strategy).to(device)

    cfg: Dict = {
        "pretrain_epochs": pretrain_epochs,
        "pretrain_iters":  pretrain_iters,
        "pretrain_lr":     pretrain_lr,
        "batch_size":      batch_size,
    }

    # In noisy mode we need per-epoch val scores so we can return the best
    # one as the "best-of-N" performance estimate.  val_best=False so we
    # don't restore checkpoints (the encoder is discarded after this call).
    history: Optional[List[Dict]] = None
    if mode == "noisy":
        cfg["eval_every"] = 1
        cfg["val_best"]   = False
        history = []

    try:
        contrastive_pretrain(
            encoder=encoder,
            cl_pipeline=pipeline,
            train_data=train_dataset,
            config=cfg,
            device=device,
            task_type=task_type,
            val_data=(val_dataset if mode == "noisy" else None),
            # P1-A: noisy mode uses full horizons (None → default
            # [24, 48, 168, 336, 720]) so the best-of-N val_score reflects
            # end-to-end forecasting quality, not short-horizon bias.
            horizons=None,
            history=history,
        )
    except Exception as exc:                           # pragma: no cover
        logger.warning("contrastive_pretrain failed for candidate: %s", exc)
        return -1e9

    # ── Noisy mode: best-of-N across training epochs ──────────────────
    if mode == "noisy" and history is not None:
        scores = [h["val_score"] for h in history if h.get("val_score") is not None]
        if scores:
            # val_score is already "higher = better" (for forecasting it is
            # -MSE averaged over the requested horizons).  Max = best epoch.
            return float(max(scores))
        logger.warning(
            "mode=noisy but no per-epoch val_scores recorded; "
            "falling back to _quick_eval on the final encoder.",
        )

    # ── Clean mode (or noisy fallback): eval the final encoder ────────
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

    Task-wise metrics (all return "higher = better"):

    - ``classification``: SVM accuracy on time-pooled embeddings.
    - ``forecasting``: negative mean MAE across the full horizon set
      ``[24, 48, 168, 336, 720]`` under the TS2Vec-aligned protocol
      (causal sliding encode + multi-step Ridge with α picked on a val
      tail).  P1-A: switched from H=24 MSE to all-horizons MAE so the
      comparator supervision reflects both short and long-range forecasting
      quality and has less heavy-tail noise.  Horizons that do not fit the
      series length are skipped automatically by :func:`eval_forecasting`.
    - ``anomaly_detection``: mean embedding-std as a proxy (higher = more
      structured representation).
    """
    from sklearn.svm import SVC
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
            # Delegate to the shared TS2Vec-protocol evaluator so the seed
            # labels use exactly the same pipeline as run_ggs_test / Phase 4
            # evaluation.  val_dataset is passed as ``test_data`` — Ridge's α
            # is then picked from a train tail (default behaviour inside
            # :func:`eval_forecasting`).
            from train.evaluate import eval_forecasting
            m = eval_forecasting(
                encoder=encoder,
                train_data=train_dataset,
                test_data=val_dataset,
                horizons=None,            # full [24, 48, 168, 336, 720]
                device=device,
            )
            if not m:
                return 0.0
            maes = [v["mae"] for v in m.values()]
            return -float(sum(maes) / len(maes))

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
    crop_len: Optional[int] = None,
    mode: str = "clean",
    randomise_init: bool = False,
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
        seed: Global random seed (also used as the base for per-candidate
            deterministic seeds when ``randomise_init=False``).
        fixed_encoders: Optional list of encoder configs to restrict the
            encoder sub-space to (Plan B Stage B). When given, candidates are
            sampled via :func:`batch_sample_strategies` rather than the full
            joint sampler — every seed record's ``encoder_config`` field will
            be drawn from this list.
        crop_len: Optional sliding-window crop length override for
            forecasting training splits.  When given, passed as
            ``window_len_override`` to :func:`load_dataset` so that
            seed generation uses a shorter crop than Phase 4.
        mode: ``"clean"`` (default) or ``"noisy"``.  Passed to
            :func:`_evaluate_candidate`:

            - ``"clean"`` runs full training and evaluates the final encoder.
              Low-variance but expensive — intended for comparator
              fine-tuning stage.
            - ``"noisy"`` runs short training with per-epoch val eval and
              records ``max(val_score)`` over epochs.  Cheap best-of-N
              estimator; intended for comparator pretraining stage.

            This matches AutoCTS++'s two-stage ``noisy_seeds`` /
            ``clean_seeds`` recipe (``reference/AutoCTS_plusplus/exps/
            generate_seeds.py:94-103``).
        randomise_init: When True, every candidate gets a fresh seed from
            system entropy (``random.SystemRandom``) rather than the
            deterministic ``seed + i`` derivation.  Use this for a
            complementary "random-init" half of noisy seed generation
            (AutoCTS++ ``use_seed=False`` branch), so that the comparator
            sees pairs from different initialisations and learns to rank
            under seed noise.

    Returns:
        List of all :class:`SeedRecord` objects.
    """
    if mode not in ("clean", "noisy"):
        raise ValueError(f"mode must be 'clean' or 'noisy', got {mode!r}")

    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AutoCTS++-style per-candidate seeding:
    #   - randomise_init=False → deterministic ``seed + i`` (default);
    #     runs are reproducible across invocations.
    #   - randomise_init=True  → fresh system entropy per candidate;
    #     teaches the comparator that the CL pipeline is noisy.
    import random as _random_mod
    _sys_random = _random_mod.SystemRandom()

    logger.info(
        "[generate_seeds] mode=%s  randomise_init=%s  n_per_dataset=%d",
        mode, randomise_init, n_per_dataset,
    )

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
        splits = load_dataset(ds_name, data_dir, window_len_override=crop_len)
        train_ds = splits["train"]
        val_ds   = splits.get("val") or splits["test"]
        task_type = train_ds.task_type

        # Per-dataset budget override (Bug #003a).
        ds_budget = (dataset_budgets or {}).get(ds_name, {}) or {}
        ds_iters  = int(ds_budget.get("pretrain_iters", 0))
        ds_epochs = int(ds_budget.get("pretrain_epochs", pretrain_epochs))
        if ds_iters > 0:
            logger.info("  budget: pretrain_iters=%d", ds_iters)
        else:
            logger.info("  budget: pretrain_epochs=%d", ds_epochs)

        if fixed_encoders is not None:
            candidates = batch_sample_strategies(n_per_dataset, fixed_encoders)
        else:
            candidates = batch_sample_candidates(n_per_dataset)
        ds_start = time.time()

        for i, (enc_cfg, strat_cfg) in enumerate(candidates):
            cand_start = time.time()

            # Per-candidate seeding (AutoCTS++-style).  This controls BOTH
            # encoder init and augmentation/loader randomness inside
            # ``contrastive_pretrain``.  With randomise_init=True we reseed
            # from system entropy, deliberately injecting per-candidate
            # variance so that pairs across candidates reflect real-world
            # run-to-run noise.
            if randomise_init:
                per_cand_seed = _sys_random.randint(0, 2**31 - 1)
            else:
                # Global offset (``ds_idx``) ensures per-dataset determinism
                # without clashing on the same ``seed + i`` across datasets.
                per_cand_seed = seed + ds_idx * n_per_dataset + i
            set_seed(per_cand_seed)

            perf = _evaluate_candidate(
                enc_cfg, strat_cfg,
                train_ds, val_ds,
                task_type,
                ds_epochs, pretrain_lr, batch_size,
                device,
                pretrain_iters=ds_iters,
                mode=mode,
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
