"""Encoder configuration space for ZeroAutoCL.

The encoder is a fixed dilated-CNN architecture (TS2Vec / AutoCLS style).
Four coarse-grained hyperparameters parameterise the encoder:

  n_layers  : number of dilated-conv blocks           — values: 4, 6, 8, 10
  hidden_dim: width of every intermediate layer       — values: 32, 64, 128
  output_dim: dimensionality of the output embedding  — values: 64, 128, 320
  mask_mode : training-time timestamp mask applied    — values: none, binomial, continuous
              after ``input_fc`` (see TS2Vec ``TSEncoder``)

Two related but distinct constants:

* :data:`ENCODER_CHOICES` — the FULL set of values ``EncoderConfig`` accepts
  (used for type validation in ``__post_init__``).  Includes ``n_layers=4``
  because the comparator's task-feature extractor
  (``models/comparator/task_feature.py``) uses a fixed 4-layer encoder as
  internal infrastructure that is NOT a search candidate.

* :data:`ENCODER_GRID_CHOICES` — the SUBSET actually enumerated by Stage A
  and sampled by ``sample_encoder_config()``.  Excludes ``n_layers=4``
  because a 4-layer dilated CNN with default dilations has receptive field
  ~31 timesteps, insufficient for the long-range forecasting horizons
  (≥168) used downstream.  Pruning shrinks Stage A's grid from
  4×3×3=36 to 3×3×3=27 (-25%) and shrinks the comparator's encoder
  sub-space symmetrically.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union


ENCODER_CHOICES: Dict[str, List[Any]] = {
    "n_layers": [4, 6, 8, 10],
    "hidden_dim": [32, 64, 128],
    "output_dim": [64, 128, 320],
    "mask_mode": ["none", "binomial", "continuous"],
}

# Stage-A grid + random-sampling subset.  Distinct from ENCODER_CHOICES so
# that infrastructure components (e.g. the lightweight task-feature
# extractor) can keep using ``n_layers=4`` without being valid search
# candidates.
ENCODER_GRID_CHOICES: Dict[str, List[Any]] = {
    "n_layers": [6, 8, 10],
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
