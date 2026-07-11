"""ACG: Anonymization Condition Generator — standalone INN (§2.1).

Learns bijection: real speaker embedding space ↔ N(0,1) latent space.
Forward:  speaker_embedding → z ~ N(0,1)  (used for NLL training).
Inverse:  key ~ N(0,1) → anonymous_speaker_embedding  (inference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import INNBlock, FixedPermutation
class AnonymizationConditionGenerator(nn.Module):
    def __init__(self, embed_dim: int = 256, num_blocks: int = 8):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for coupling splits"
        self.blocks = nn.ModuleList()
        self.perms = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(INNBlock(embed_dim))
            # Fixed random permutation between blocks (§8.1, seed varies per block)
            self.perms.append(FixedPermutation(embed_dim, seed=42 + i))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Speaker embedding → latent + cumulative log-det (for NLL loss)."""
        log_det = torch.zeros(s.shape[0], device=s.device)
        z = s
        for block, perm in zip(self.blocks, self.perms):
            z, ld = block(z)
            z = perm(z)
            log_det = log_det + ld
        return z, log_det

    @torch.no_grad()
    def generate(self, key: torch.Tensor) -> torch.Tensor:
        """key ~ N(0,1) → anonymous speaker embedding (inverse pass).

        The raw INN inverse emits vectors with norm ~11 (it cannot contract the
        norm-16 N(0,1) latent down to the norm-1 unit-sphere the ASV embeddings
        live on). Feeding such conditions to the cINN drives the FiLM modulation
        ~11x too hard, pinning every block's log-scale at the +4 clamp so the
        forward pass multiplies by exp(4)^12 ≈ 1e20 and the mel explodes.
        L2-normalizing here puts the condition in the SAME unit-norm space as the
        real ASV embeddings (s) used by the consistency loss, which is the space
        the cINN must be trained against. Same normalization is applied in both
        anonymize() and restore() (both call generate), so invertibility holds.
        """
        sa = key
        for block, perm in zip(reversed(list(self.blocks)), reversed(list(self.perms))):
            sa = perm.inverse(sa)
            sa = block.inverse(sa)
        return F.normalize(sa, dim=-1)

    def loss(self, s: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """NLL loss: Eq. 4 — LACG = E[||f(s)||²/2 − log|J|].

        The τ||ω||² regularisation from the paper is implemented as Adam
        weight_decay in the optimizer (not as an explicit loss term), which
        is the standard practice for L2 regularisation.  The *tau* argument
        is kept for API compatibility but is unused here.
        """
        z, log_det = self.forward(s)
        nll = 0.5 * (z ** 2).sum(dim=-1) - log_det
        return nll.mean()
