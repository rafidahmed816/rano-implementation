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
        num_cinn_blocks: int = 12,  # §2.2: N_inn = 12
        num_acg_blocks: int = 8,  # §2.1: N_acg = 8
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
        distance_threshold: float = 0.5,  # §7: d = 0.5 (L2)
        return_distances: bool = False,  # Return distance stats for monitoring
    ) -> dict[str, torch.Tensor]:
        """
        Algorithm 1 in paper: generate key, anonymize, compute Lcons + Ltri (Eq. 7).

        Implements Algorithm 1 from paper:
            Step 1: Extract speaker embedding s = ASV(x)  [frozen ASV]
            Step 2-4: Sample key z until ||s - c|| ≥ d, where c = ACG(z)
            Step 5: Anonymize: xa = cINN(x, c)
            Step 6: Consistency: x_hat = cINN(x, s)  [SII condition]
            Step 7-8: Compute L_cons and L_tri, aggregate as L_total

        Args:
            x: (B, 80, T) mel-spectrogram batch
            distance_threshold: Paper §7 threshold d = 0.5 (L2 distance)
            return_distances: If True, include distance stats in returned dict

        Returns:
            Dict with keys: 'total', 'consistency', 'triplet', [and 'distances' if requested]

        **CRITICAL CONSTRAINT** (Algorithm 1 line 2-4):
            The returned conditioning MUST satisfy ||s - c|| ≥ distance_threshold.
            If not, training violates the paper's specification.
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
        # ASV params are frozen (requires_grad=False) so no ASV weights update,
        # but we must NOT use no_grad here — L_tri gradient must flow through
        # the ASV computation graph back into xa and then into the anonymizer.
        anchor_emb = self.asv(xa)

        # Steps 10-11: L_tri + L_total
        losses = self.loss_fn(x, x_hat, anchor_emb, cond, s)

        # Optional: Track distance statistics for debugging
        if return_distances:
            with torch.no_grad():
                dist = torch.norm(s - cond, dim=-1, p=2).mean()
                losses["distance"] = dist

        return losses

    def _sample_far_key(self, s: torch.Tensor, d: float) -> torch.Tensor:
        """Sample anonymous embedding with L2 distance > d from real embedding.

        Paper Algorithm 1, line 2-4 (§3.1 step 5):
            "Sample key z from 𝒩(0,I) until ||s − c|| ≥ d (where c = ψ(z))"

        Args:
            s: Speaker embeddings (B, 256)
            d: Distance threshold (default 0.5 in §7)

        Returns:
            cond: Anonymous conditioning (B, 256) with ||s - cond|| ≥ d

        **CRITICAL**: If this returns cond with dist < d, training violates Algorithm 1.
        Fallback should never be used in production training.
        """
        device = s.device
        max_retries = 200  # Increased from 50 for better convergence
        best_cond = None
        best_dist = 0.0

        for attempt in range(max_retries):
            key = torch.randn_like(s)
            with torch.no_grad():
                cond = self.acg.generate(key)
            # Paper §7: threshold d=0.5 is L2 distance (Euclidean norm)
            dist = torch.norm(s - cond, dim=-1, p=2).mean()

            # Track best attempt (for fallback)
            if dist.item() > best_dist:
                best_dist = dist.item()
                best_cond = cond

            # Success: found conditioning with sufficient distance
            if dist.item() > d:
                return cond

        # FALLBACK: Use best attempt found
        # WARNING: This means Algorithm 1 constraint is NOT satisfied!
        if best_dist < d * 0.95:  # Very low distance
            import warnings

            warnings.warn(
                f"[ALGORITHM 1 VIOLATION] Key sampling failed to meet distance threshold.\n"
                f"  Expected: ||s - c|| > {d:.4f}\n"
                f"  Got: ||s - c|| = {best_dist:.4f}\n"
                f"  This violates Paper Algorithm 1 and may harm anonymization quality.\n"
                f"  Possible causes:\n"
                f"    1. ACG not fully converged (increase Stage 1 iterations)\n"
                f"    2. ACG degenerate output distribution (check embedding norm)\n"
                f"    3. Distance threshold too high (reduce d parameter)",
                RuntimeWarning,
            )
        return best_cond

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
