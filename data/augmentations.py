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
import random
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
    """Placeholder for crop — real logic lives in :class:`AugmentationPipeline`.

    Only ``p`` (the overlap fraction) is stored.  The pipeline reads this
    directly so it can sample a **shared** overlap and two asymmetric view
    windows (TS2Vec / AutoCLS protocol).  Calling :meth:`_apply` is a
    programming error — the pipeline special-cases the ``'crop'`` step.

    Args:
        p: Overlap fraction.  ``0.0`` disables cropping (both views span the
            full input).
    """

    def _apply(self, x: Tensor) -> Tensor:  # pragma: no cover
        raise RuntimeError(
            "RandomCropping._apply must not be called directly — "
            "AugmentationPipeline handles overlap-aligned cropping."
        )


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

    def _sample_view_windows(
        self, T: int,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Sample ``(overlap, view1, view2)`` windows in the TS2Vec style.

        ``overlap = [ol, or)`` is the shared segment both views must cover.
        ``view1 = [v1l, or)`` extends **left** of the overlap; ``view2 =
        [ol, v2r)`` extends **right**.  When the two views are encoded
        independently the overlap positions see different left/right
        context — this is the core contrastive signal that TS2Vec and
        AutoCLS both rely on.

        When ``crop.p == 0`` (or T is too short) both views span the full
        input and the slices degenerate to ``slice(0, T)``.

        Args:
            T: Input time length.

        Returns:
            ``((ol, or), (v1l, v1r), (v2l, v2r))``; each element is a pair
            of integer indices into the input time axis.
        """
        crop_p = self.augmentations["crop"].p

        if crop_p == 0.0 or T < 4:
            return (0, T), (0, T), (0, T)

        overlap_len = max(2, int(round(T * crop_p)))
        if overlap_len >= T:
            return (0, T), (0, T), (0, T)

        overlap_left  = random.randint(0, T - overlap_len)
        overlap_right = overlap_left + overlap_len

        # View 1: extends left of overlap, right edge = overlap_right.
        v1l = random.randint(0, overlap_left)
        v1r = overlap_right

        # View 2: extends right of overlap, left edge = overlap_left.
        v2l = overlap_left
        v2r = random.randint(overlap_right, T)

        return (overlap_left, overlap_right), (v1l, v1r), (v2l, v2r)

    def _apply_chain_with_view(
        self, x: Tensor, view_window: Tuple[int, int],
    ) -> Tensor:
        """Apply all augmentations in order for one view.

        At the ``'crop'`` step the tensor is **structurally sliced** to
        ``x[:, vl:vr, :]`` using the pre-sampled view window — no zero
        padding.  Other augmentations operate on whatever length of tensor
        they happen to receive (augs before crop see the full T, augs after
        crop see ``vr - vl``).  All existing augmentations are length-
        polymorphic so this works transparently.

        Args:
            x: Input tensor of shape (B, T, C).
            view_window: ``(vl, vr)`` — the view's absolute-time range.

        Returns:
            Augmented tensor of shape ``(B, vr - vl, C)`` (or ``(B, T, C)``
            when crop is disabled).
        """
        vl, vr = view_window
        for name in self.order:
            if name == "crop":
                x = x[:, vl:vr, :]
            else:
                x = self.augmentations[name](x)
        return x

    def __call__(
        self, x: Tensor,
    ) -> Tuple[Tensor, Tensor, slice, slice]:
        """Generate two overlap-aligned augmented views of *x*.

        The two returned tensors typically have **different time lengths**
        (each view extends asymmetrically around the shared overlap).  The
        two ``slice`` objects locate the overlap inside each view; the
        downstream contrastive loss should apply them to the encoder
        output so that positives are time-aligned in the original series.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Tuple ``(x1, x2, slice1, slice2)`` where ``x1[:, slice1]`` and
            ``x2[:, slice2]`` correspond to the same absolute time range
            ``[overlap_left, overlap_right)`` of the input.
        """
        T = x.shape[1]
        (ol, or_), view1, view2 = self._sample_view_windows(T)

        x1 = self._apply_chain_with_view(x, view1)
        x2 = self._apply_chain_with_view(x, view2)

        slice1 = slice(ol - view1[0], or_ - view1[0])
        slice2 = slice(ol - view2[0], or_ - view2[0])
        return x1, x2, slice1, slice2
