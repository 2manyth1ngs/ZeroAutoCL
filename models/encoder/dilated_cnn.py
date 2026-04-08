"""Parameterised Dilated CNN Encoder for ZeroAutoCL.

Architecture (TS2Vec / AutoCLS style)
--------------------------------------
input_proj   : Conv1d(input_dim  → hidden_dim, kernel=1)
layers[0..n) : DilatedConvBlock with dilation = 2^i, i = 0 … n_layers-1
output_proj  : Conv1d(hidden_dim → output_dim, kernel=1)

Each DilatedConvBlock uses ``SamePadConv`` so the sequence length T is
preserved throughout.  A residual connection is added around the block.

Input  shape: (B, T, C)   — batch, time, channels
Output shape: (B, T, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder_config import EncoderConfig


# ---------------------------------------------------------------------------
# Building blocks
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


class DilatedConvBlock(nn.Module):
    """Single residual dilated-conv block.

    Structure::

        x  →  SamePadConv(d) → ReLU → Dropout(0.1)  → + x  →  out

    The residual branch is a plain identity connection; both branches operate
    on tensors of shape (B, hidden_dim, T) throughout.

    Args:
        hidden_dim: Number of channels (same for input and output).
        dilation: Dilation factor for the convolution.
        dropout: Dropout probability applied after ReLU.
    """

    def __init__(
        self,
        hidden_dim: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = SamePadConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply dilated conv with residual skip.

        Args:
            x: Shape (B, hidden_dim, T).

        Returns:
            Shape (B, hidden_dim, T).
        """
        return x + self.dropout(self.relu(self.conv(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class DilatedCNNEncoder(nn.Module):
    """Parameterised Dilated CNN encoder.

    The encoder maps a multivariate time series of shape (B, T, C) to a
    sequence of embeddings of shape (B, T, output_dim).

    Architecture::

        input_proj   : Conv1d(input_dim  → hidden_dim, kernel=1)
        blocks[i]    : DilatedConvBlock(hidden_dim, dilation=2^i)
                       for i in range(n_layers)
        output_proj  : Conv1d(hidden_dim → output_dim, kernel=1)

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

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, dilation=2 ** i)
            for i in range(n_layers)
        ])

        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of time series.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Embedding tensor of shape (B, T, output_dim).
        """
        # Conv1d expects (B, C, T), so we transpose in/out.
        h = x.permute(0, 2, 1)        # (B, C, T)
        h = self.input_proj(h)         # (B, hidden_dim, T)
        for block in self.blocks:
            h = block(h)               # (B, hidden_dim, T)
        h = self.output_proj(h)        # (B, output_dim, T)
        return h.permute(0, 2, 1)      # (B, T, output_dim)

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
