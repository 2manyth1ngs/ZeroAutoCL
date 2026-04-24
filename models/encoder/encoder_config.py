"""Encoder configuration space for ZeroAutoCL.

The encoder is a fixed dilated-CNN architecture (TS2Vec / AutoCLS style).
Four coarse-grained hyperparameters are searched:

  n_layers  : number of dilated-conv blocks           — choices: 4, 6, 8, 10
  hidden_dim: width of every intermediate layer       — choices: 32, 64, 128
  output_dim: dimensionality of the output embedding  — choices: 64, 128, 320
  mask_mode : training-time timestamp mask applied    — choices: none, binomial, continuous
              after ``input_fc`` (see TS2Vec ``TSEncoder``)

This gives 4 × 3 × 3 × 3 = 108 distinct configurations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union


ENCODER_CHOICES: Dict[str, List[Any]] = {
    "n_layers": [4, 6, 8, 10],
    "hidden_dim": [32, 64, 128],
    "output_dim": [64, 128, 320],
    "mask_mode": ["none", "binomial", "continuous"],
}


@dataclass
class EncoderConfig:
    """Coarse-grained encoder configuration.

    Attributes:
        n_layers: Number of dilated-conv blocks.  Must be in
            ``ENCODER_CHOICES['n_layers']``.
        hidden_dim: Hidden (intermediate) channel width.  Must be in
            ``ENCODER_CHOICES['hidden_dim']``.
        output_dim: Output embedding dimension.  Must be in
            ``ENCODER_CHOICES['output_dim']``.
        mask_mode: Training-time timestamp-level mask mode applied to the
            post-``input_fc`` latent.  ``'binomial'`` matches the TS2Vec
            default; ``'none'`` disables the mask; ``'continuous'`` masks
            short consecutive spans.  Must be in
            ``ENCODER_CHOICES['mask_mode']``.
    """

    n_layers: int = 10
    hidden_dim: int = 64
    output_dim: int = 320
    mask_mode: str = "binomial"

    def __post_init__(self) -> None:
        for field, choices in ENCODER_CHOICES.items():
            value = getattr(self, field)
            if value not in choices:
                raise ValueError(
                    f"EncoderConfig.{field}={value!r} is not in {choices}"
                )

    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Return a plain dict representation."""
        return {
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "mask_mode": self.mask_mode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Union[int, str]]) -> "EncoderConfig":
        """Construct from a plain dict (e.g. loaded from YAML / JSON).

        ``mask_mode`` defaults to ``'binomial'`` if absent so that old
        configs (written before the search dimension existed) default to the
        TS2Vec-aligned behaviour.  Callers that want to preserve the old
        no-mask behaviour of legacy seed data should pass ``mask_mode='none'``
        explicitly.
        """
        return cls(
            n_layers=d["n_layers"],
            hidden_dim=d["hidden_dim"],
            output_dim=d["output_dim"],
            mask_mode=d.get("mask_mode", "binomial"),
        )
