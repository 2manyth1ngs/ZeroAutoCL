"""Contrastive pre-training loop.

``contrastive_pretrain`` is the standard training loop used both during
seed generation (short runs) and final full training (longer runs).
It optimises the encoder parameters only; the remaining pipeline components
(augmentation, embedding transform) carry no learned weights that need
updating here.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TimeSeriesDataset
from models.encoder.dilated_cnn import DilatedCNNEncoder
from models.contrastive.cl_pipeline import CLPipeline

logger = logging.getLogger(__name__)


def _val_score(
    encoder: DilatedCNNEncoder,
    train_data: TimeSeriesDataset,
    val_data: TimeSeriesDataset,
    task_type: str,
    horizons: Optional[List[int]],
    device: torch.device,
) -> Optional[float]:
    """Run a quick downstream eval and return a *higher-is-better* scalar.

    Returns ``None`` if eval is not applicable (e.g. val_data missing or
    val_data too short for any forecasting horizon).  See Bug #003a.
    """
    from train.evaluate import (
        eval_classification, eval_forecasting, eval_anomaly_detection,
    )

    encoder.eval()
    try:
        if task_type == "classification":
            m = eval_classification(encoder, train_data, val_data, device=device)
            return float(m["acc"])
        elif task_type == "forecasting":
            m = eval_forecasting(
                encoder, train_data, val_data, horizons=horizons, device=device,
            )
            if not m:
                return None
            mses = [v["mse"] for v in m.values()]
            return -float(sum(mses) / len(mses))   # negate: higher = better
        elif task_type == "anomaly_detection":
            m = eval_anomaly_detection(encoder, train_data, val_data, device=device)
            return float(m["f1"])
    except Exception as exc:                          # pragma: no cover
        logger.warning("val eval failed: %s", exc)
        return None
    finally:
        encoder.train()
    return None


def contrastive_pretrain(
    encoder: DilatedCNNEncoder,
    cl_pipeline: CLPipeline,
    train_data: TimeSeriesDataset,
    config: dict,
    device: Optional[torch.device] = None,
    val_data: Optional[TimeSeriesDataset] = None,
    task_type: Optional[str] = None,
    horizons: Optional[List[int]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> DilatedCNNEncoder:
    """Train *encoder* via contrastive learning.

    Args:
        encoder: The dilated-CNN encoder to train (must be the same object
            stored inside *cl_pipeline*).
        cl_pipeline: Fully configured :class:`~models.contrastive.CLPipeline`.
        train_data: Training split of the dataset.
        config: Training config dict.  Recognised keys:

            - ``pretrain_iters`` (int, default 0) — if > 0, train for exactly
              this many optimizer steps and **ignore** ``pretrain_epochs``.
              This is the TS2Vec-style budget and is the recommended unit
              for forecasting datasets where val-based early stopping is
              unreliable (see Bug #003a).
            - ``pretrain_epochs`` (int, default 40) — used only when
              ``pretrain_iters`` is 0 / unset.
            - ``pretrain_lr`` / ``lr`` (float, default 1e-3)
            - ``batch_size`` (int, default 64)
            - ``eval_every`` (int, default 0) — if > 0 and *val_data* given,
              run a downstream eval every N epochs and keep the best
              checkpoint by val score.  See Bug #003a.
            - ``val_best`` (bool, default True for classification /
              anomaly_detection, **False for forecasting**) — toggle the
              best-by-val checkpoint mechanism.  ETT-style forecasting
              datasets exhibit train→val→test distribution drift, so the
              val signal points at the wrong epoch; the default is to
              return the last checkpoint and rely on a fixed iter budget
              instead.

        device: Torch device.  ``None`` → auto-detect.
        val_data: Optional validation split.  When provided together with
            ``eval_every > 0``, ``task_type``, and ``val_best=True``, the
            encoder is evaluated every N epochs and the best-by-val
            state_dict is restored before return.
        task_type: ``'classification' | 'forecasting' | 'anomaly_detection'``.
            Required when *val_data* is given.
        horizons: Forecasting horizons (only when ``task_type='forecasting'``).
        history: Optional caller-supplied list; if given, training appends one
            ``{epoch, loss, val_score}`` dict per epoch.  Useful for plotting
            loss / val curves without re-running training.

    Returns:
        The trained encoder (same object as input, updated in-place).  When
        val-best is enabled, the encoder's parameters are restored to the
        best-by-val checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs     = int(config.get("pretrain_epochs", config.get("epochs", 40)))
    iters      = int(config.get("pretrain_iters", 0))
    lr         = float(config.get("pretrain_lr", config.get("lr", 1e-3)))
    batch_size = int(config.get("batch_size", 64))
    eval_every = int(config.get("eval_every", 0))

    # Forecasting datasets (especially ETT) suffer from train→val→test
    # distribution drift, so val-based early stopping picks the wrong
    # epoch.  Default val-best off for forecasting; on for everything else.
    # Callers may explicitly override via ``config['val_best']``.
    # See Bug #003a "真正的根因 (修订)" in CLAUDE_DEBUG.md.
    if "val_best" in config:
        val_best_flag = bool(config["val_best"])
    else:
        val_best_flag = task_type != "forecasting"

    val_enabled = (
        val_best_flag
        and eval_every > 0
        and val_data is not None
        and task_type is not None
    )

    encoder.to(device)
    cl_pipeline.to(device)
    cl_pipeline.train()

    optimizer = torch.optim.Adam(cl_pipeline.parameters(), lr=lr)
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    best_score: float = float("-inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: int = -1

    # Iter-budget mode (TS2Vec-style): keep cycling through the loader
    # until ``iters`` optimizer steps have been taken.  We still expose
    # an "epoch" notion to keep logging / val-eval cadence consistent;
    # in iter mode each epoch becomes whatever fraction of the loader
    # we happen to consume before hitting the iter budget.
    iter_mode = iters > 0
    if iter_mode:
        # Stop iterating in epochs once iter budget is exhausted, but
        # set ``epochs`` high enough that the iter budget is the binding
        # constraint.  We also recompute a "logical" epoch count for
        # logging based on the loader length.
        epochs_per_loader = max(1, len(train_data) // batch_size)
        epochs = max(1, (iters + epochs_per_loader - 1) // epochs_per_loader)
        logger.info(
            "Iter-budget mode: %d iters → %d logical epochs "
            "(loader=%d batches/epoch)",
            iters, epochs, epochs_per_loader,
        )

    global_step = 0
    stop = False

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, _ in loader:
            x_batch = x_batch.to(device)

            # OOM guard: covers both forward and backward passes.
            loss = None
            try:
                loss, _ = cl_pipeline(x_batch)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(cl_pipeline.parameters(), max_norm=1.0)
                optimizer.step()
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                # Free tensors from the failed forward/backward before retrying.
                del loss
                torch.cuda.empty_cache()
                batch_size = max(2, batch_size // 2)
                loader = DataLoader(
                    train_data, batch_size=batch_size,
                    shuffle=True, drop_last=True,
                )
                logger.warning("OOM — reducing batch_size to %d", batch_size)
                break

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if iter_mode and global_step >= iters:
                stop = True
                break

        mean_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")

        # ── Optional val eval + best-checkpoint tracking (Bug #003a) ──
        val_score: Optional[float] = None
        if val_enabled and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            val_score = _val_score(
                encoder, train_data, val_data, task_type, horizons, device,
            )
            if val_score is not None and val_score > best_score:
                best_score = val_score
                best_state = copy.deepcopy(encoder.state_dict())
                best_epoch = epoch + 1

        # ── Per-epoch logging (was: every 10 % of training) ──
        if val_score is not None:
            logger.info(
                "Pretrain epoch %d/%d  loss=%.4f  val_score=%.4f%s",
                epoch + 1, epochs, mean_loss, val_score,
                "  *best*" if best_epoch == epoch + 1 else "",
            )
        else:
            logger.info(
                "Pretrain epoch %d/%d  loss=%.4f",
                epoch + 1, epochs, mean_loss,
            )

        if history is not None:
            history.append({
                "epoch": epoch + 1,
                "step": global_step,
                "loss": mean_loss,
                "val_score": val_score,
            })

        if stop:
            logger.info(
                "Reached iter budget (%d steps) at epoch %d — stopping.",
                global_step, epoch + 1,
            )
            break

    # ── Restore best-by-val checkpoint if available ──
    if val_enabled and best_state is not None:
        encoder.load_state_dict(best_state)
        logger.info(
            "Restored best-by-val checkpoint from epoch %d (score=%.4f)",
            best_epoch, best_score,
        )

    return encoder
