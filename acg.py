"""Anonymization Condition Generator — INN-based speaker embedding generator (Sec. III-C)."""

import torch
import torch.nn as nn
from blocks import INNBlock


class AnonymizationConditionGenerator(nn.Module):
    """
    INN mapping between speaker embedding space ↔ standard normal latent space.

    Training: maximise log-likelihood of real speaker embeddings (Eq. 4).
    Inference: sample key ~ N(0,1) → reverse INN → anonymous speaker embedding.
    """

    def __init__(self, embed_dim: int = 256, num_blocks: int = 8):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for coupling splits"
        self.blocks = nn.ModuleList([INNBlock(embed_dim) for _ in range(num_blocks)])

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Speaker embedding → latent + cumulative log-det (for NLL loss)."""
        log_det = torch.zeros(s.shape[0], device=s.device)
        z = s
        for block in self.blocks:
            z, ld = block(z)
            log_det = log_det + ld
        return z, log_det

    @torch.no_grad()
    def generate(self, key: torch.Tensor) -> torch.Tensor:
        """key ~ N(0,1) → anonymous speaker embedding sa (inference only)."""
        sa = key
        for block in reversed(self.blocks):
            sa = block.inverse(sa)
        return sa

    def loss(self, s: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """NLL loss: Eq. 4 — LACG = E[||f(s)||²/2 − log|J|] + τ||ω||²."""
        z, log_det = self.forward(s)
        nll = 0.5 * (z ** 2).sum(dim=-1) - log_det
        reg = tau * sum((p ** 2).sum() for p in self.parameters())
        return nll.mean() + reg
