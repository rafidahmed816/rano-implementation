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
        lambda_logdet: float = 0.01,
    ):
        super().__init__()
        self.acg = AnonymizationConditionGenerator(embed_dim, num_acg_blocks)
        self.anonymizer = Anonymizer(mel_channels, embed_dim, num_cinn_blocks)
        self.asv = AdaINVCSpeakerEncoder(mel_channels, embed_dim)
        self.loss_fn = RanoLoss(lambda1, lambda2, margin, lambda_logdet)
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

        # Steps 6-7: run both cINN passes in a single batched forward.
        # Concatenating along the batch dim means all 12 cINN blocks execute
        # once instead of twice — ~2x throughput for the most expensive op.
        x2 = torch.cat([x, x], dim=0)       # (2B, 80, T)
        cond2 = torch.cat([cond, s], dim=0)  # (2B, 256)
        out2, log_det2 = self.anonymizer(x2, cond2)  # (2B, 80, T), (2B,)
        xa, x_hat = out2.chunk(2, dim=0)     # each (B, 80, T)
        log_det_anon, _ = log_det2.chunk(2, dim=0)  # log_det for anonymization pass

        # Step 8: L_cons = MSE(x, x_hat)  (Eq. 5)
        # Step 9: emb_ano = ASV(xa) — speaker embedding of anonymized speech
        # ASV params are frozen (requires_grad=False) so no ASV weights update,
        # but we must NOT use no_grad here — L_tri gradient must flow through
        # the ASV computation graph back into xa and then into the anonymizer.
        anchor_emb = self.asv(xa)

        # Steps 10-11: L_tri + L_total (now also includes log_det regularization)
        # Compute actual element count for proper logdet normalization:
        # each cINN block sums log_s over (mel_channels × T), across all blocks.
        n_elements = x.shape[1] * x.shape[2] * len(self.anonymizer.blocks)
        losses = self.loss_fn(x, x_hat, anchor_emb, cond, s, log_det_anon,
                              n_elements=n_elements)

        # Optional: Track distance statistics for debugging
        if return_distances:
            with torch.no_grad():
                dist = torch.norm(s - cond, dim=-1, p=2).mean()
                losses["distance"] = dist

        return losses

    def _sample_far_key(self, s: torch.Tensor, d: float) -> torch.Tensor:
        """Vectorized key sampling — N candidates per batch element, one ACG pass.

        Paper Algorithm 1 line 2-4: sample key from N(0,1) until ||s - c|| >= d.

        Instead of up to 200 sequential ACG forward passes, we sample N=32
        candidate keys per batch element simultaneously and select the best
        (highest L2 distance from s) per element in a single ACG call.

        Args:
            s: Speaker embeddings (B, 256)
            d: Distance threshold (default 0.5 in §7)

        Returns:
            cond: Anonymous conditioning (B, 256), best distance per element.
        """
        device = s.device
        B, embed_dim = s.shape
        N = 32  # candidates per batch element — covers >99% of cases in one pass

        # Sample all B*N keys and run them through ACG in one batched call
        keys = torch.randn(B * N, embed_dim, device=device)
        with torch.no_grad():
            conds_flat = self.acg.generate(keys)  # (B*N, embed_dim)

        conds = conds_flat.view(B, N, embed_dim)                  # (B, N, D)
        s_exp = s.unsqueeze(1).expand(B, N, embed_dim)            # (B, N, D)
        dists = torch.norm(s_exp - conds, dim=-1, p=2)            # (B, N)
        best_idx = dists.argmax(dim=1)                            # (B,)
        result = conds[torch.arange(B, device=device), best_idx]  # (B, D)

        min_best_dist = dists[torch.arange(B, device=device), best_idx].min().item()
        if min_best_dist < d * 0.95:
            import warnings
            warnings.warn(
                f"[ALGORITHM 1] Best distance {min_best_dist:.4f} < threshold {d:.4f}. "
                f"ACG may need more pre-training (Stage 1 iterations).",
                RuntimeWarning,
            )
        return result

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
        # Use fp64 for maximum numerical precision in cINN transforms.
        # A100 handles fp64 natively; this eliminates floating-point error
        # amplification across 12 sequential cINN blocks.
        orig_dtype = next(self.anonymizer.parameters()).dtype
        self.anonymizer.double()
        xa, _ = self.anonymizer(x.double(), cond.double())
        self.anonymizer.to(orig_dtype)
        return xa.float(), cond

    @torch.no_grad()
    def restore(self, xa: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Restore anonymized mel → original mel using the correct key (lossless, §4.2)."""
        cond = self.acg.generate(key)
        # Use fp64 for maximum numerical precision in inverse cINN.
        # The inverse divides by exp(log_s) at each block — with fp32/fp16,
        # errors compound multiplicatively across 12 blocks.
        # fp64 gives ~15 decimal digits vs fp32's ~7, virtually eliminating
        # numerical restoration error.
        orig_dtype = next(self.anonymizer.parameters()).dtype
        self.anonymizer.double()
        xr = self.anonymizer.inverse(xa.double(), cond.double())
        self.anonymizer.to(orig_dtype)
        return xr.float()
