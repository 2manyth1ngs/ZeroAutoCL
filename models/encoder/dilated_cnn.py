"""Parameterised Dilated CNN Encoder for ZeroAutoCL.

Architecture aligned with the TS2Vec reference implementation
(``reference/ts2vec/models/dilated_conv.py`` + ``models/encoder.py``),
with the three coarse-grained hyperparameters (n_layers, hidden_dim,
output_dim) remaining searchable.

TS2Vec architecture
-------------------
input_fc          : Linear(input_dim → hidden_dim)
blocks[0..n]      : ConvBlock with dilation = 2^i, i = 0 … n_layers
                     — blocks 0..n_layers-1 : hidden_dim → hidden_dim
                     — block  n_layers      : hidden_dim → output_dim (final)
repr_dropout      : Dropout(0.1) applied once after the entire conv stack

Each ConvBlock contains **two** ``SamePadConv`` layers with GELU
pre-activation and a residual skip.  When in_channels ≠ out_channels
(or ``final=True``), a 1×1 projector is added on the residual path.

Input  shape: (B, T, C)   — batch, time, channels
Output shape: (B, T, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder_config import EncoderConfig


# ---------------------------------------------------------------------------
# Building blocks  (mirrors reference/ts2vec/models/dilated_conv.py)
# ---------------------------------------------------------------------------

class SamePadConv1d(nn.Module):
    """Conv1d that preserves the input time dimension (same-padding).

    For kernel_size k and dilation d the receptive field is
    ``(k - 1) * d + 1``.  We pad ``receptive_field // 2`` on each side and
    trim 1 step from the right when the receptive field is even, mirroring
    the TS2Vec reference implementation.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        # Trim 1 step on the right when the receptive field is even so that
        # output length == input length.
        self.trim = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.trim:
            out = out[:, :, :-self.trim]
        return out


class ConvBlock(nn.Module):
    """Pre-activation residual block with two dilated convolutions.

    Structure (matches TS2Vec ``ConvBlock``)::

        residual = projector(x) if needed else x
        x → GELU → conv1 → GELU → conv2 → + residual → out

    A 1×1 projector is added on the residual path when ``in_channels !=
    out_channels`` or when ``final=True`` (the last block in the stack).

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Convolution kernel size (default 3).
        dilation: Dilation factor for both conv1 and conv2.
        final: If True, force a projector on the residual path even
            when in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        final: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = SamePadConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply pre-activation residual block.

        Args:
            x: Shape (B, C_in, T).

        Returns:
            Shape (B, C_out, T).
        """
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class DilatedCNNEncoder(nn.Module):
    """Parameterised Dilated CNN encoder aligned with TS2Vec.

    The encoder maps a multivariate time series of shape (B, T, C) to a
    sequence of embeddings of shape (B, T, output_dim).

    Architecture::

        input_fc        : Linear(input_dim → hidden_dim)
        blocks[0..n-1]  : ConvBlock(hidden_dim → hidden_dim, dilation=2^i)
        blocks[n]       : ConvBlock(hidden_dim → output_dim, dilation=2^n, final)
        repr_dropout    : Dropout(0.1)

    Total n_layers+1 ConvBlocks, each containing 2 SamePadConv layers.

    Args:
        input_dim: Number of input channels C (dataset-dependent).
        config: :class:`EncoderConfig` specifying n_layers, hidden_dim,
            output_dim.  If omitted the default config is used.
    """

    def __init__(
        self,
        input_dim: int,
        config: EncoderConfig | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = EncoderConfig()

        self.config = config
        n_layers = config.n_layers
        hidden_dim = config.hidden_dim
        output_dim = config.output_dim

        # ── Input projection (Linear, matches TS2Vec input_fc) ───────────
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # ── Convolutional stack ──────────────────────────────────────────
        # channels[i] defines the output dim of block i.
        # Blocks 0..n_layers-1 : hidden_dim → hidden_dim
        # Block  n_layers      : hidden_dim → output_dim  (final)
        channels = [hidden_dim] * n_layers + [output_dim]
        self.feature_extractor = nn.Sequential(*[
            ConvBlock(
                in_channels=channels[i - 1] if i > 0 else hidden_dim,
                out_channels=channels[i],
                kernel_size=3,
                dilation=2 ** i,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ])

        # ── Repr dropout (single, at the end — matches TS2Vec) ──────────
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of time series.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Embedding tensor of shape (B, T, output_dim).
        """
        h = self.input_fc(x)              # (B, T, hidden_dim)
        h = h.transpose(1, 2)             # (B, hidden_dim, T)
        h = self.repr_dropout(
            self.feature_extractor(h)
        )                                  # (B, output_dim, T)
        return h.transpose(1, 2)           # (B, T, output_dim)

    @classmethod
    def from_config_dict(cls, input_dim: int, config_dict: dict) -> "DilatedCNNEncoder":
        """Convenience constructor from a plain dict.

        Args:
            input_dim: Number of input channels.
            config_dict: Dict with keys 'n_layers', 'hidden_dim', 'output_dim'.

        Returns:
            Initialised :class:`DilatedCNNEncoder`.
        """
        return cls(input_dim, EncoderConfig.from_dict(config_dict))
