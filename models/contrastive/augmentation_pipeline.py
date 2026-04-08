"""Re-export AugmentationPipeline from data.augmentations.

The actual implementation lives in ``data/augmentations.py``; this module
provides a single import point for the ``models/contrastive`` package.
"""

from data.augmentations import AugmentationPipeline, AUGMENTATION_ORDERS  # noqa: F401

__all__ = ["AugmentationPipeline", "AUGMENTATION_ORDERS"]
