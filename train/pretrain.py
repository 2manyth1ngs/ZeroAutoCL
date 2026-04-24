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
from torch.optim.swa_utils import AveragedModel
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
            # Primary metric: mean MAE across all evaluated horizons (P1-A).
            # MAE is linear in errors so catastrophic outliers do not dominate
            # the way they do with MSE — this matches AutoCTS++'s seed labels
            # and reduces the heavy-tail load on the pairwise comparator.
            maes = [v["mae"] for v in m.values()]
            return -float(sum(maes) / len(maes))   # negate: higher = better
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
            - ``use_ema`` (bool, default True) — maintain an averaged copy
              of the encoder parameters (``torch.optim.swa_utils
              .AveragedModel`` with default simple-average ``avg_fn``) that
              is updated every optimizer step.  When enabled, all val-eval
              and the final returned encoder use the averaged weights, which
              follows the TS2Vec recipe and reduces variance between
              otherwise-equivalent seeds.

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
    use_ema    = bool(config.get("use_ema", True))
    # Optimizer / gradient-clipping toggles.  Defaults preserve the original
    # ZeroAutoCL behaviour (Adam + clip_grad_norm 1.0) so existing callers
    # stay on rails; set ``optimizer='adamw'`` and ``grad_clip=0`` to match
    # the TS2Vec recipe.
    optimizer_type = str(config.get("optimizer", "adam")).lower()
    grad_clip = float(config.get("grad_clip", 1.0))

    # Forecasting datasets (especially ETT) suffer from train→val→test
    # distribution drift, so val-based early stopping picks the wrong
    # epoch.  Default val-best off for forecasting; on for everything else.
    # Callers may explicitly override via ``config['val_best']``.
    # See Bug #003a "真正的根因 (修订)" in CLAUDE_DEBUG.md.
    if "val_best" in config:
        val_best_flag = bool(config["val_best"])
    else:
        val_best_flag = task_type != "forecasting"

    # Split into two gates:
    #   - ``eval_enabled``     controls per-epoch val_score computation and
    #                          the history-record population.  Required by
    #                          AutoCTS++-style noisy seed generation, which
    #                          needs max(val_score) across epochs even when
    #                          val-best restoration is disabled.
    #   - ``val_best_enabled`` controls whether to remember the best-by-val
    #                          state_dict and restore it on exit.
    eval_enabled = (
        eval_every > 0
        and val_data is not None
        and task_type is not None
    )
    val_best_enabled = val_best_flag and eval_enabled

    encoder.to(device)
    cl_pipeline.to(device)
    cl_pipeline.train()

    # EMA / SWA averaged copy of the encoder.  Mirrors TS2Vec's
    # ``torch.optim.swa_utils.AveragedModel`` usage — parameters are updated
    # every optimizer step and swapped in for evaluation and for the final
    # returned encoder.  The AveragedModel deepcopies the encoder internally,
    # so cost is one extra encoder-sized tensor set (typically <5 MB for
    # our 10-layer / 64-hidden / 320-output config).
    if use_ema:
        ema_encoder: Optional[AveragedModel] = AveragedModel(encoder).to(device)
    else:
        ema_encoder = None

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(cl_pipeline.parameters(), lr=lr)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(cl_pipeline.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type!r}")
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
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(cl_pipeline.parameters(), max_norm=grad_clip)
                optimizer.step()
                if ema_encoder is not None:
                    ema_encoder.update_parameters(encoder)
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
        # NOTE: ``eval_enabled`` only governs whether we COMPUTE val_score
        # each epoch; ``val_best_enabled`` governs whether we remember the
        # best state.  Noisy seed generation turns val_best off but still
        # needs the per-epoch scores through ``history``.
        val_score: Optional[float] = None
        if eval_enabled and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
            # When EMA is on, evaluate the AVERAGED weights (that's what
            # downstream code ultimately uses).  Swap them into encoder
            # temporarily, then restore the training weights so optimisation
            # continues from where it left off.
            if ema_encoder is not None:
                train_state = copy.deepcopy(encoder.state_dict())
                encoder.load_state_dict(ema_encoder.module.state_dict())
                try:
                    val_score = _val_score(
                        encoder, train_data, val_data, task_type, horizons, device,
                    )
                finally:
                    encoder.load_state_dict(train_state)
            else:
                val_score = _val_score(
                    encoder, train_data, val_data, task_type, horizons, device,
                )
            if val_best_enabled and val_score is not None and val_score > best_score:
                best_score = val_score
                # Save the EMA weights as the best checkpoint — matching
                # what was actually evaluated.  Without EMA, fall back to
                # the live training weights.
                if ema_encoder is not None:
                    best_state = copy.deepcopy(ema_encoder.module.state_dict())
                else:
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

    # ── Copy EMA weights into the encoder for downstream use ──
    # When val-best is also on, this happens BEFORE best_state restore so
    # that best_state (which itself holds EMA weights) overrides correctly.
    if ema_encoder is not None:
        encoder.load_state_dict(ema_encoder.module.state_dict())
        logger.info("EMA weights copied into encoder for downstream eval")

    # ── Restore best-by-val checkpoint if available ──
    if val_best_enabled and best_state is not None:
        encoder.load_state_dict(best_state)
        logger.info(
            "Restored best-by-val checkpoint from epoch %d (score=%.4f)",
            best_epoch, best_score,
        )

    return encoder
