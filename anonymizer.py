"""Anonymizer (forward cINN) and Restorer (inverse cINN) — parameter-shared (Sec. III-B)."""

import torch
import torch.nn as nn
from blocks import CINNBlock


class Anonymizer(nn.Module):
    def __init__(self, mel_channels: int = 80, cond_dim: int = 256, num_blocks: int = 8):
        super().__init__()
        # Project odd channel counts to even
        self.in_proj = nn.Conv2d(mel_channels, mel_channels + mel_channels % 2, 1)
        self._channels = mel_channels + mel_channels % 2
        self.blocks = nn.ModuleList(
            [CINNBlock(self._channels, cond_dim) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv2d(self._channels, mel_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, F, T) mel → anonymized mel xa (same shape)."""
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h)

    def inverse(self, xa: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """xa: anonymized mel + same cond → restored mel xr ≈ x (lossless)."""
        h = self.in_proj(xa)
        for block in reversed(self.blocks):
            h = block.inverse(h, cond)
        return self.out_proj(h)
