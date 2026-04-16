"""Rano: full speaker anonymization model integrating ACG, Anonymizer, SpeakerEncoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from acg import AnonymizationConditionGenerator
from anonymizer import Anonymizer
from loss import RanoLoss
from speaker_encoder import SpeakerEncoder


class Rano(nn.Module):
    """
    Full Rano model (Fig. 2 in paper).

    Components (all inter-connected during training):
      - acg: generates anonymous speaker embedding from key
      - anonymizer: cINN forward — x + cond → xa
      - restorer: cINN inverse (same weights) — xa + cond → xr
      - asv: differentiable speaker encoder for contrastive + consistency losses
    """

    def __init__(
        self,
        mel_channels: int = 80,
        embed_dim: int = 256,
        num_cinn_blocks: int = 8,
        num_acg_blocks: int = 8,
        hidden: int = 512,
        lambda1: float = 1.0,
        lambda2: float = 5.0,
        margin: float = 0.3,
        acg_tau: float = 0.5,
    ):
        super().__init__()
        self.acg = AnonymizationConditionGenerator(embed_dim, num_acg_blocks)
        self.anonymizer = Anonymizer(mel_channels, embed_dim, num_cinn_blocks)
        self.asv = SpeakerEncoder(mel_channels, embed_dim, hidden)
        self.loss_fn = RanoLoss(lambda1, lambda2, margin)
        self.acg_tau = acg_tau

    # ------------------------------------------------------------------
    # Stage 1: pre-train ACG
    # ------------------------------------------------------------------

    def acg_loss(self, speaker_embeddings: torch.Tensor) -> torch.Tensor:
        """NLL loss for ACG pre-training (Eq. 4)."""
        return self.acg.loss(speaker_embeddings, self.acg_tau)

    # ------------------------------------------------------------------
    # Stage 2: train anonymizer
    # ------------------------------------------------------------------

    def training_step(
        self,
        x: torch.Tensor,
        distance_threshold: float = 0.3,
    ) -> dict[str, torch.Tensor]:
        """
        Algorithm 1 in paper: generate key, anonymize, compute Lcons + Ltri.
        x: (B, 1, F, T) mel-spectrogram batch.
        Returns loss dict with keys: total, consistency, triplet.
        """
        s = self.asv(x)  # original speaker embeddings

        # Sample key until far enough from original (Algorithm 1 lines 2-4)
        cond = self._sample_far_key(s, distance_threshold)

        # Forward process 1: xa = f(x; cond)
        xa = self.anonymizer(x, cond.unsqueeze(-1).unsqueeze(-1))

        # Forward process 2: x_hat = f(x; s) — identity transform expectation
        x_hat = self.anonymizer(x, s.unsqueeze(-1).unsqueeze(-1))

        # Contrastive: anchor = asv(xa), positive = cond, negative = s
        anchor_emb = self.asv(xa.detach())

        return self.loss_fn(x, x_hat, anchor_emb, cond, s)

    def _sample_far_key(self, s: torch.Tensor, d: float) -> torch.Tensor:
        """Sample anonymous embedding sufficiently far from real speaker embedding."""
        device = s.device
        for _ in range(50):  # max retries
            key = torch.randn_like(s)
            cond = self.acg.generate(key)
            dist = 1.0 - torch.nn.functional.cosine_similarity(s, cond).mean()
            if dist.item() > d:
                return cond
        return cond  # fallback: use last sample

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def anonymize(self, x: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Anonymize speech mel-spectrogram.
        Returns (anonymized_mel, cond) — cond needed for restoration.
        """
        cond = self.acg.generate(key)
        c = cond.unsqueeze(-1).unsqueeze(-1)
        xa = self.anonymizer(x, c)
        return xa, cond

    @torch.no_grad()
    def restore(self, xa: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Restore anonymized mel → original mel using the correct key (lossless)."""
        cond = self.acg.generate(key)
        c = cond.unsqueeze(-1).unsqueeze(-1)
        return self.anonymizer.inverse(xa, c)
