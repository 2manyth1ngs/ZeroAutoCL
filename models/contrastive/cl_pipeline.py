"""Complete contrastive learning pipeline.

``CLPipeline`` wires together:
  1. Data augmentation (two independent views)
  2. Encoder (shared weights for both views)
  3. Embedding transform (jitter / mask / norm)
  4. Pair construction + loss computation

The ``forward`` method returns ``(total_loss, loss_dict)`` where
``loss_dict`` contains per-contrast-type scalar losses for logging.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from data.augmentations import AugmentationPipeline
from .embedding_transform import EmbeddingTransform
from .pair_construction import ContrastivePairConstructor
from .losses import InfoNCELoss, TripletLoss


# ---------------------------------------------------------------------------
# Strategy config keys and defaults
# ---------------------------------------------------------------------------

_DEFAULT_AUG_CFG = {
    "resize": 0.2, "rescale": 0.3, "jitter": 0.0,
    "point_mask": 0.2, "freq_mask": 0.0, "crop": 0.2, "order": 3,
}
_DEFAULT_EMB_CFG = {"jitter_p": 0.7, "mask_p": 0.1, "norm_type": "none"}
_DEFAULT_PAIR_CFG = {
    "instance": True, "temporal": False, "cross_scale": False,
    "kernel_size": 5, "pool_op": "avg", "adj_neighbor": False,
}
_DEFAULT_LOSS_CFG = {"type": "infonce", "sim_func": "euclidean", "temperature": 1.0}


# ---------------------------------------------------------------------------
# CLPipeline
# ---------------------------------------------------------------------------

class CLPipeline(nn.Module):
    """Full contrastive learning pipeline.

    Constructed from an encoder and a strategy config dict that covers all
    four strategy dimensions (augmentation, embedding transform, pair
    construction, loss).

    Example strategy config (GGS default from AutoCLS Table 5)::

        strategy_config = {
            'augmentation': {
                'resize': 0.2, 'rescale': 0.3, 'jitter': 0.0,
                'point_mask': 0.2, 'freq_mask': 0.0, 'crop': 0.2, 'order': 3,
            },
            'embedding_transform': {
                'jitter_p': 0.7, 'mask_p': 0.1, 'norm_type': 'none',
            },
            'pair_construction': {
                'instance': True, 'temporal': False, 'cross_scale': False,
                'kernel_size': 5, 'pool_op': 'avg', 'adj_neighbor': False,
            },
            'loss': {
                'type': 'infonce', 'sim_func': 'euclidean', 'temperature': 1.0,
            },
        }

    Args:
        encoder: A :class:`~models.encoder.DilatedCNNEncoder` (or any
            ``nn.Module`` that maps (B, T, C) → (B, T, D)).
        strategy_config: Strategy config dict as shown above.  Missing
            sub-dicts fall back to GGS defaults.
    """

    def __init__(
        self,
        encoder: nn.Module,
        strategy_config: Dict,
        max_temporal_len: int = 200,
    ) -> None:
        super().__init__()

        aug_cfg  = strategy_config.get("augmentation",       _DEFAULT_AUG_CFG)
        emb_cfg  = strategy_config.get("embedding_transform", _DEFAULT_EMB_CFG)
        pair_cfg = strategy_config.get("pair_construction",   _DEFAULT_PAIR_CFG)
        loss_cfg = strategy_config.get("loss",                _DEFAULT_LOSS_CFG)

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder = encoder

        # Infer output dimensionality from the encoder's config if available,
        # otherwise fall back to a dummy forward pass.
        if hasattr(encoder, "config"):
            embed_dim: int = encoder.config.output_dim
        else:
            embed_dim = self._infer_embed_dim(encoder)

        # ── Augmentation pipeline ─────────────────────────────────────────
        self.aug_pipeline = AugmentationPipeline(aug_cfg)

        # ── Embedding transform ───────────────────────────────────────────
        jitter_p  = float(emb_cfg.get("jitter_p", 0.0))
        mask_p    = float(emb_cfg.get("mask_p",   0.0))
        norm_type = str(emb_cfg.get("norm_type",  "none"))
        self.emb_transform = EmbeddingTransform(jitter_p, mask_p, norm_type, embed_dim)

        # ── Pair constructor ──────────────────────────────────────────────
        # Ensure instance=True (not searched).
        pair_cfg = dict(pair_cfg)
        pair_cfg["instance"] = True
        self.pair_constructor = ContrastivePairConstructor(pair_cfg, max_temporal_len)

        # ── Loss function ─────────────────────────────────────────────────
        loss_type = str(loss_cfg.get("type", "infonce")).lower()
        sim_func  = str(loss_cfg.get("sim_func", "dot"))
        temperature = float(loss_cfg.get("temperature", 0.1))

        if loss_type == "infonce":
            self.loss_fn: Union[InfoNCELoss, TripletLoss] = InfoNCELoss(sim_func, temperature)
        elif loss_type == "triplet":
            self.loss_fn = TripletLoss(sim_func)
        else:
            raise ValueError(f"Unknown loss type: {loss_type!r}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run the full CL pipeline on a batch.

        Args:
            x: Input time series of shape (B, T, C).

        Returns:
            Tuple of:
              - ``total_loss``: Mean of all active contrast losses (scalar).
              - ``loss_dict``:  Dict mapping contrast-type name → scalar loss.
        """
        # ── Step 1: Augmentation (no grad — no learnable params) ─────────
        # The pipeline returns two views of potentially DIFFERENT lengths
        # together with slice objects locating the shared overlap region
        # inside each view (TS2Vec / AutoCLS protocol).  Encoding each view
        # independently gives the overlap positions different left / right
        # context — the core contrastive signal.
        with torch.no_grad():
            x1, x2, slice1, slice2 = self.aug_pipeline(x)
        x1 = x1.detach()
        x2 = x2.detach()

        # ── Step 2: Encoding ─────────────────────────────────────────────
        h1 = self.encoder(x1)   # (B, L1, D)
        h2 = self.encoder(x2)   # (B, L2, D)

        # ── Step 2b: Slice to the overlap — now h1 and h2 are time-aligned ─
        h1 = h1[:, slice1, :]   # (B, overlap_len, D)
        h2 = h2[:, slice2, :]

        # ── Step 3: Embedding transform ──────────────────────────────────
        h1 = self.emb_transform(h1)
        h2 = self.emb_transform(h2)

        # ── Steps 4–8: Pair construction + loss ──────────────────────────
        loss_dict = self.pair_constructor.compute_all_losses(h1, h2, self.loss_fn)

        if not loss_dict:
            total_loss = x.new_zeros(1).squeeze()
        else:
            total_loss = torch.stack(list(loss_dict.values())).mean()

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_embed_dim(encoder: nn.Module) -> int:
        """Infer output embedding dimension via a small dummy forward pass.

        Args:
            encoder: The encoder module.

        Returns:
            Output channel count D.
        """
        device = next(encoder.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 16, 1, device=device)
            out = encoder(dummy)
        return out.shape[-1]
