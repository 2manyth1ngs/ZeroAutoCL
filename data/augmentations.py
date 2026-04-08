"""Data augmentations for contrastive learning on time series.

All augmentations follow the convention:
  - Input / output shape: (B, T, C)
  - ``p`` is the primary augmentation parameter (typically noise std or mask
    ratio).
  - ``p == 0.0`` means the augmentation is disabled and the input is returned
    unchanged.

Augmentation ordering is based on AutoCLS Table 8 (5 pre-defined orders).
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Pre-defined augmentation orders (AutoCLS Table 8)
# ---------------------------------------------------------------------------
AUGMENTATION_ORDERS: Dict[int, List[str]] = {
    0: ["resize", "rescale", "freq_mask", "jitter", "point_mask", "crop"],
    1: ["resize", "rescale", "freq_mask", "jitter", "crop", "point_mask"],
    2: ["resize", "rescale", "freq_mask", "crop", "jitter", "point_mask"],
    3: ["resize", "rescale", "crop", "freq_mask", "jitter", "point_mask"],
    4: ["resize", "crop", "rescale", "freq_mask", "jitter", "point_mask"],
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseAugmentation:
    """Base class for all augmentations.

    Args:
        p: Augmentation strength parameter in [0, 1].  p=0.0 disables the
           augmentation entirely.
    """

    def __init__(self, p: float) -> None:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the augmentation.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Augmented tensor of shape (B, T, C).
        """
        if self.p == 0.0:
            return x
        return self._apply(x)

    def _apply(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

class Resizing(BaseAugmentation):
    """Temporal resizing by a random scale factor drawn from N(1, p²).

    The factor is clamped to [0.5, 2.0] to prevent degenerate cases.  After
    rescaling the time axis, the result is interpolated back to the original
    length T.

    Args:
        p: Standard deviation of the scale factor noise.
    """

    def _apply(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        factor = torch.empty(1, device=x.device).normal_(mean=1.0, std=self.p)
        factor = factor.clamp(0.5, 2.0).item()

        new_T = max(1, int(round(T * factor)))
        # F.interpolate expects (B, C, T)
        x_t = x.permute(0, 2, 1)  # (B, C, T)
        x_resized = F.interpolate(x_t, size=new_T, mode="linear", align_corners=False)
        x_back = F.interpolate(x_resized, size=T, mode="linear", align_corners=False)
        return x_back.permute(0, 2, 1)  # (B, T, C)


class Rescaling(BaseAugmentation):
    """Per-channel amplitude rescaling by a random factor drawn from N(1, p²).

    Args:
        p: Standard deviation of the per-channel scale factor noise.
    """

    def _apply(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        # One scale factor per (batch, channel); shape (B, 1, C)
        factor = torch.empty(B, 1, C, device=x.device).normal_(mean=1.0, std=self.p)
        return x * factor


class Jittering(BaseAugmentation):
    """Additive Gaussian noise with std=p.

    Args:
        p: Noise standard deviation.
    """

    def _apply(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x) * self.p
        return x + noise


class PointMasking(BaseAugmentation):
    """Randomly zero-out individual time steps with probability p.

    Args:
        p: Fraction of time steps to mask (masking probability).
    """

    def _apply(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        # Mask shape (B, T, 1); broadcast over channels.
        mask = torch.bernoulli(
            torch.full((B, T, 1), 1.0 - self.p, device=x.device)
        )
        return x * mask


class FrequencyMasking(BaseAugmentation):
    """Randomly mask p fraction of frequency components via DFT.

    Uses ``torch.fft.rfft`` / ``torch.fft.irfft`` along the time axis
    (dim=-2 in (B, T, C) layout).

    Args:
        p: Fraction of frequency components to zero out.
    """

    def _apply(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        # DFT along time axis; rfft gives n_freq = T // 2 + 1 complex components.
        x_freq = torch.fft.rfft(x, dim=-2)  # (B, n_freq, C)
        n_freq = x_freq.shape[-2]

        n_masked = max(0, int(math.ceil(n_freq * self.p)))
        if n_masked == 0:
            return x

        # For each sample in the batch, independently choose which frequencies
        # to mask so that randomness is per-sample.
        for b in range(B):
            indices = torch.randperm(n_freq, device=x.device)[:n_masked]
            x_freq[b, indices, :] = 0.0

        x_rec = torch.fft.irfft(x_freq, n=T, dim=-2)  # (B, T, C)
        return x_rec


class RandomCropping(BaseAugmentation):
    """Generate two overlapping crops of the time series.

    The shared (overlapping) segment spans a fraction p of the total length T.
    Two crops are drawn such that they both contain this shared segment.

    ``__call__`` returns a tuple ``(crop1, crop2, overlap_start, overlap_end)``
    instead of a single tensor so that temporal contrast can align the two
    views correctly.

    When used inside :class:`AugmentationPipeline`, only the first crop is
    returned as the augmented view; the second crop is tracked separately.

    Args:
        p: Fraction of T that forms the shared (overlapping) segment.
    """

    def __call__(  # type: ignore[override]
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, int, int]:
        if self.p == 0.0:
            return x, x, 0, x.shape[1]
        return self._apply_crop(x)

    def _apply(self, x: Tensor) -> Tensor:  # used by AugmentationPipeline
        crop1, _, _, _ = self._apply_crop(x)
        return crop1

    def _apply_crop(self, x: Tensor) -> Tuple[Tensor, Tensor, int, int]:
        B, T, C = x.shape
        shared_len = max(1, int(round(T * self.p)))
        # Shared segment starts somewhere that leaves room for both crops.
        max_start = T - shared_len
        if max_start <= 0:
            return x, x, 0, T

        overlap_start = torch.randint(0, max_start + 1, (1,)).item()
        overlap_end = int(overlap_start) + shared_len

        crop1 = x[:, :overlap_end, :]     # prefix crop — ends at overlap_end
        crop2 = x[:, int(overlap_start):, :]  # suffix crop — starts at overlap_start

        # Pad both crops to the original length T for shape consistency.
        def pad_to(t: Tensor, target_len: int) -> Tensor:
            pad_needed = target_len - t.shape[1]
            if pad_needed > 0:
                t = F.pad(t.permute(0, 2, 1), (0, pad_needed)).permute(0, 2, 1)
            return t

        crop1 = pad_to(crop1, T)
        crop2 = pad_to(crop2, T)
        return crop1, crop2, int(overlap_start), overlap_end


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Apply a chain of augmentations in a pre-defined order and generate two views.

    Args:
        config: Dict with keys matching the augmentation names and an 'order'
            key specifying which of the 5 pre-defined orders to use.  Example::

                {
                    'resize': 0.2, 'rescale': 0.3, 'jitter': 0.0,
                    'point_mask': 0.2, 'freq_mask': 0.0,
                    'crop': 0.2, 'order': 3
                }
    """

    _AUG_CLASSES: Dict[str, type] = {
        "resize": Resizing,
        "rescale": Rescaling,
        "jitter": Jittering,
        "point_mask": PointMasking,
        "freq_mask": FrequencyMasking,
        "crop": RandomCropping,
    }

    def __init__(self, config: Dict[str, float]) -> None:
        order_idx = int(config.get("order", 0))
        if order_idx not in AUGMENTATION_ORDERS:
            raise ValueError(
                f"'order' must be in {list(AUGMENTATION_ORDERS)}, got {order_idx}"
            )
        self.order: List[str] = AUGMENTATION_ORDERS[order_idx]
        self.augmentations: Dict[str, BaseAugmentation] = {
            name: self._AUG_CLASSES[name](float(config.get(name, 0.0)))
            for name in self.order
        }

    def _apply_chain(self, x: Tensor) -> Tensor:
        """Apply all augmentations in sequence to produce one view."""
        for name in self.order:
            aug = self.augmentations[name]
            if isinstance(aug, RandomCropping):
                x = aug._apply(x)
            else:
                x = aug(x)
        return x

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate two independently augmented views of *x*.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Tuple (x1, x2) of independently augmented views, each (B, T, C).
        """
        x1 = self._apply_chain(x)
        x2 = self._apply_chain(x)
        return x1, x2
