from __future__ import annotations

import warnings

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
        num_acg_blocks: int = 8,    # §2.1: N_acg = 8
        lambda1: float = 1.0,
        lambda2: float = 5.0,
        margin: float = 0.3,
        acg_tau: float = 0.5,
        lambda_logdet: float = 0.01,
        lambda_anchor: float = 0.0,
        lambda_range: float = 0.0,
    ):
        super().__init__()
        self.acg = AnonymizationConditionGenerator(embed_dim, num_acg_blocks)
        self.anonymizer = Anonymizer(mel_channels, embed_dim, num_cinn_blocks)
        self.asv = AdaINVCSpeakerEncoder(mel_channels, embed_dim)
        self.loss_fn = RanoLoss(lambda1, lambda2, margin, lambda_logdet,
                                lambda_anchor, lambda_range)
        self.acg_tau = acg_tau

        # AMP settings — set by train_stage2.py before training starts so that
        # autocast lives INSIDE training_step rather than wrapping it from
        # outside.  This is required for torch.compile: an external autocast
        # context manager creates a graph-break boundary that causes the CUDA
        # stream queue to back up and deadlock at ~500–680 iterations.
        self._amp_enabled: bool = False
        self._amp_dtype: torch.dtype = torch.float32
        self._device_type: str = "cuda"

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
        return_distances: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Algorithm 1 in paper: generate key, anonymize, compute Lcons + Ltri (Eq. 7).

        FIX (torch.compile deadlock): autocast is applied HERE, inside the
        compiled graph, rather than as an external context manager in the
        training loop.  When torch.compile traces this function it sees a single
        clean graph with no dtype-context boundary.  Applying autocast from
        outside created a graph-break that caused the CUDA stream queue to
        back up and deadlock at ~500–680 iterations.

        Implements Algorithm 1 from paper:
            Step 1: Extract speaker embedding s = ASV(x)  [frozen ASV]
            Step 2-4: Sample key z until ||s - c|| >= d, where c = ACG(z)
            Step 5: Anonymize: xa = cINN(x, c)
            Step 6: Consistency: x_hat = cINN(x, s)  [SII condition]
            Step 7-8: Compute L_cons and L_tri, aggregate as L_total

        Args:
            x: (B, 80, T) mel-spectrogram batch
            distance_threshold: Paper §7 threshold d = 0.5 (L2 distance)
            return_distances: If True, include distance stats in returned dict

        Returns:
            Dict with keys: 'total', 'consistency', 'triplet', [and 'distances' if requested]
        """
        with torch.amp.autocast(
            device_type=self._device_type,
            enabled=self._amp_enabled,
            dtype=self._amp_dtype,
        ):
            return self._training_step_inner(x, distance_threshold, return_distances)

    def _training_step_inner(
        self,
        x: torch.Tensor,
        distance_threshold: float,
        return_distances: bool,
    ) -> dict[str, torch.Tensor]:
        """Inner implementation — called inside autocast by training_step."""
        # Step 2: extract speaker embedding (frozen ASV — §8.1)
        with torch.no_grad():
            s = self.asv(x)

        # Steps 3-5: sample key, compute condition, ensure L2 distance > d
        cond = self._sample_far_key(s, distance_threshold)

        # Steps 6-7: run both cINN passes in a single batched forward.
        # Concatenating along the batch dim means all 12 cINN blocks execute
        # once instead of twice — ~2x throughput for the most expensive op.
        x2 = torch.cat([x, x], dim=0)        # (2B, 80, T)
        cond2 = torch.cat([cond, s], dim=0)   # (2B, 256)
        out2, log_det2 = self.anonymizer(x2, cond2)  # (2B, 80, T), (2B,)
        xa, x_hat = out2.chunk(2, dim=0)      # each (B, 80, T)
        log_det_anon, _ = log_det2.chunk(2, dim=0)

        # Step 8: L_cons = MSE(x, x_hat)  (Eq. 5)
        # ASV params are frozen but we must NOT use no_grad here —
        # L_tri gradient must flow through ASV back into xa and the anonymizer.
        anchor_emb = self.asv(xa)

        # Steps 10-11: L_tri + L_total (also includes log_det regularization)
        n_elements = x.shape[1] * x.shape[2] * len(self.anonymizer.blocks)
        losses = self.loss_fn(
            x, x_hat, anchor_emb, cond, s, log_det_anon, n_elements=n_elements, xa=xa
        )

        if return_distances:
            with torch.no_grad():
                dist = torch.norm(s - cond, dim=-1, p=2).mean()
                losses["distance"] = dist

        return losses

    def _sample_far_key(self, s: torch.Tensor, d: float) -> torch.Tensor:
        """Vectorized key sampling — N candidates per batch element, one ACG pass.

        Paper Algorithm 1 line 2-4: sample key from N(0,1) until ||s - c|| >= d.

        FIX: Removed .item() CPU/GPU sync that happened every training step.
        With torch.compile, calling .item() every step causes the CUDA stream
        queue to back up and deadlock at ~600-680 iterations. The warning is
        now guarded so it only runs outside of compiled graph execution.

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

        conds = conds_flat.view(B, N, embed_dim)               # (B, N, D)
        s_exp = s.unsqueeze(1).expand(B, N, embed_dim)         # (B, N, D)
        dists = torch.norm(s_exp - conds, dim=-1, p=2)         # (B, N)
        best_idx = dists.argmax(dim=1)                         # (B,)
        result = conds[torch.arange(B, device=device), best_idx]  # (B, D)

        # FIX 1: Guard .item() (CPU/GPU sync) behind is_compiling() check.
        # .item() on every step was backing up the CUDA stream queue and
        # deadlocking at ~660 iterations under torch.compile.
        # FIX 2: import warnings moved to top of file — was causing a
        # graph break forcing torch.compile to retrace the graph here.
        if not torch.compiler.is_compiling():
            min_best_dist = dists[
                torch.arange(B, device=device), best_idx
            ].min().item()
            if min_best_dist < d * 0.95:
                warnings.warn(
                    f"[ALGORITHM 1] Best distance {min_best_dist:.4f} < "
                    f"threshold {d:.4f}. ACG may need more pre-training.",
                    RuntimeWarning,
                )

        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def anonymize(
        self, x: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Anonymize speech mel-spectrogram (§4.1).
        x: (B, 80, T), key: (B, 256).
        Returns (anonymized_mel, cond) — store key for restoration.

        Uses float64 for the full forward pass to minimize numerical error
        that would compound during the inverse (restoration).
        """
        cond = self.acg.generate(key)
        # Convert entire anonymizer to float64 so ALL operations (weights,
        # biases, intermediate activations) use double precision.
        orig_dtype = next(self.anonymizer.parameters()).dtype
        self.anonymizer.double()
        xa, _ = self.anonymizer(x.double(), cond.double())
        self.anonymizer.to(orig_dtype)
        return xa.float(), cond

    @torch.no_grad()
    def restore(self, xa: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Restore anonymized mel → original mel using the correct key (§4.2).

        CRITICAL: The entire anonymizer must be in float64 during inverse.
        Just casting input tensors to .double() does NOT work — Conv1d/Linear
        layers cast inputs back to match their weight dtype (float32), so the
        computation still runs in float32 and numerical errors compound
        across 12 cINN blocks, causing values to explode to 1e23.

        Converting the model weights to float64 ensures ALL intermediate
        computations use double precision, reducing per-block error from
        ~1e-7 (fp32) to ~1e-16 (fp64), which stays bounded over 12 blocks.
        """
        cond = self.acg.generate(key)
        # Convert entire anonymizer to float64
        orig_dtype = next(self.anonymizer.parameters()).dtype
        self.anonymizer.double()
        xr = self.anonymizer.inverse(xa.double(), cond.double())
        # Restore to original dtype
        self.anonymizer.to(orig_dtype)
        return xr.float()