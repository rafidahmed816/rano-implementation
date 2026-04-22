"""Rano: full speaker anonymization model integrating ACG, Anonymizer, SpeakerEncoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from acg import AnonymizationConditionGenerator
from anonymizer import Anonymizer
from loss import RanoLoss
from speaker_encoder import AdaINVCSpeakerEncoder


class Rano(nn.Module):
    """
    Full Rano model (Fig. 2 in paper).

    Components (all inter-connected during training):
      - acg: generates anonymous speaker embedding from key (frozen in Stage 2)
      - anonymizer: cINN forward — x + cond → xa
      - restorer: cINN inverse (same weights) — xa + cond → xr
      - asv: AdaIN-VC speaker encoder (frozen in Stage 2)
    """

    def __init__(
        self,
        mel_channels: int = 80,
        embed_dim: int = 256,
        num_cinn_blocks: int = 12,   # §2.2: N_inn = 12
        num_acg_blocks: int = 8,     # §2.1: N_acg = 8
        lambda1: float = 1.0,
        lambda2: float = 5.0,
        margin: float = 0.3,
        acg_tau: float = 0.5,
    ):
        super().__init__()
        self.acg = AnonymizationConditionGenerator(embed_dim, num_acg_blocks)
        self.anonymizer = Anonymizer(mel_channels, embed_dim, num_cinn_blocks)
        self.asv = AdaINVCSpeakerEncoder(mel_channels, embed_dim)
        self.loss_fn = RanoLoss(lambda1, lambda2, margin)
        self.acg_tau = acg_tau

    # ------------------------------------------------------------------
    # Stage 1: pre-train ACG
    # ------------------------------------------------------------------

    def acg_loss(self, speaker_embeddings: torch.Tensor) -> torch.Tensor:
        """NLL loss for ACG pre-training (Eq. 4)."""
        return self.acg.loss(speaker_embeddings, self.acg_tau)

    # ------------------------------------------------------------------
    # Stage 2: train anonymizer (Algorithm 1)
    # ------------------------------------------------------------------

    def training_step(
        self,
        x: torch.Tensor,
        distance_threshold: float = 0.5,   # §7: d = 0.5 (L2)
    ) -> dict[str, torch.Tensor]:
        """
        Algorithm 1 in paper: generate key, anonymize, compute Lcons + Ltri.
        x: (B, 80, T) mel-spectrogram batch.
        Returns loss dict with keys: total, consistency, triplet.
        """
        # Step 2: extract speaker embedding (frozen ASV — §8.1)
        with torch.no_grad():
            s = self.asv(x)

        # Steps 3-5: sample key, compute condition, ensure L2 distance > d
        cond = self._sample_far_key(s, distance_threshold)

        # Step 6: xa = Anonymizer(x, cond) — forward anonymization
        xa, _ = self.anonymizer(x, cond)

        # Step 7: x_hat = Anonymizer(x, s) — consistency check with real embedding
        x_hat, _ = self.anonymizer(x, s)

        # Step 8: L_cons = MSE(x, x_hat)  (Eq. 5)
        # Step 9: emb_ano = ASV(xa) — speaker embedding of anonymized speech
        # NOTE: do NOT detach xa — triplet gradient must flow through anonymizer
        with torch.no_grad():
            anchor_emb = self.asv(xa)

        # Steps 10-11: L_tri + L_total
        return self.loss_fn(x, x_hat, anchor_emb, cond, s)

    def _sample_far_key(self, s: torch.Tensor, d: float) -> torch.Tensor:
        """Sample anonymous embedding with L2 distance > d from real embedding (§3.1 step 5)."""
        device = s.device
        for _ in range(50):  # max retries
            key = torch.randn_like(s)
            with torch.no_grad():
                cond = self.acg.generate(key)
            # Paper §7: threshold d=0.5 is L2 distance
            dist = torch.norm(s - cond, dim=-1).mean()
            if dist.item() > d:
                return cond
        return cond  # fallback: use last sample

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def anonymize(self, x: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Anonymize speech mel-spectrogram (§4.1).
        x: (B, 80, T), key: (B, 256).
        Returns (anonymized_mel, cond) — store key for restoration.
        """
        cond = self.acg.generate(key)
        xa, _ = self.anonymizer(x, cond)
        return xa, cond

    @torch.no_grad()
    def restore(self, xa: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Restore anonymized mel → original mel using the correct key (lossless, §4.2)."""
        cond = self.acg.generate(key)
        return self.anonymizer.inverse(xa, cond)
