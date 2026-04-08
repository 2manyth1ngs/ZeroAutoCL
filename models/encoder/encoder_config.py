"""Encoder configuration space for ZeroAutoCL.

The encoder is a fixed dilated-CNN architecture (TS2Vec / AutoCLS style).
Only three coarse-grained hyperparameters are searched:

  n_layers  : number of dilated-conv blocks           — choices: 4, 6, 8, 10
  hidden_dim: width of every intermediate layer       — choices: 32, 64, 128
  output_dim: dimensionality of the output embedding  — choices: 64, 128, 320

This gives 4 × 3 × 3 = 36 distinct configurations.
"""

from dataclasses import dataclass
from typing import Dict, List


ENCODER_CHOICES: Dict[str, List[int]] = {
    "n_layers": [4, 6, 8, 10],
    "hidden_dim": [32, 64, 128],
    "output_dim": [64, 128, 320],
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
    """

    n_layers: int = 10
    hidden_dim: int = 64
    output_dim: int = 320

    def __post_init__(self) -> None:
        for field, choices in ENCODER_CHOICES.items():
            value = getattr(self, field)
            if value not in choices:
                raise ValueError(
                    f"EncoderConfig.{field}={value!r} is not in {choices}"
                )

    def to_dict(self) -> Dict[str, int]:
        """Return a plain dict representation."""
        return {
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "EncoderConfig":
        """Construct from a plain dict (e.g. loaded from YAML / JSON)."""
        return cls(
            n_layers=d["n_layers"],
            hidden_dim=d["hidden_dim"],
            output_dim=d["output_dim"],
        )
