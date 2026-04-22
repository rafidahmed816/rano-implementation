"""Anonymizer (forward cINN) and Restorer (inverse cINN) — parameter-shared (§2.2, §2.3).

Input/output shape: (B, 80, T) mel-spectrogram — 1D along time axis.
The Restorer runs the same cINN blocks in REVERSE order (§2.3).
No separate parameters for the Restorer.
"""

import torch
import torch.nn as nn
from blocks import CINNBlock, FixedPermutation


class Anonymizer(nn.Module):
    def __init__(self, mel_channels: int = 80, cond_dim: int = 256, num_blocks: int = 12):
        super().__init__()
        assert mel_channels % 2 == 0, "mel_channels must be even for channel split"
        self.blocks = nn.ModuleList(
            [CINNBlock(mel_channels, cond_dim) for _ in range(num_blocks)]
        )
        # Fixed random permutation between blocks (§8.1, seed varies per block)
        self.perms = nn.ModuleList(
            [FixedPermutation(mel_channels, seed=42 + i) for i in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 80, T) mel → (anonymized mel xa, cumulative log_det).
        cond: (B, 256) speaker embedding.
        log_det: (B,) — sum of log|J| across all cINN blocks.
        """
        h = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for block, perm in zip(self.blocks, self.perms):
            h, ld = block(h, cond)
            log_det_total = log_det_total + ld
            h = perm(h)
        return h, log_det_total

    def inverse(self, xa: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """xa: anonymized mel + same cond → restored mel xr ≡ x (lossless).
        Runs blocks in REVERSE order using inverse equations (§2.3).
        """
        h = xa
        for block, perm in zip(
            reversed(list(self.blocks)), reversed(list(self.perms))
        ):
            h = perm.inverse(h)
            h = block.inverse(h, cond)
        return h
