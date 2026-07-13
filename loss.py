"""Training losses for Rano — Eq. 5 (consistency) and Eq. 6 (triplet / SDI-differentiation).

Extended with optional log-det Jacobian regularization for better cINN invertibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """Lcons = ||x - f(x; s, θ)||² — SII-consistency (Eq. 5)."""

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x)


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = 1.0 - F.cosine_similarity(anchor, positive)
        d_neg = 1.0 - F.cosine_similarity(anchor, negative)
        return F.relu(self.margin + d_pos - d_neg).mean()


class RanoLoss(nn.Module):
    """
    Total loss for Rano second training stage (Eq. 7):
    Ltotal = λ1 * Lcons + λ2 * Ltri + λ_logdet * L_logdet

    L_logdet = -mean(log_det) encourages volume-preserving cINN transforms,
    improving invertibility and reducing restoration noise.
    """

    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 5.0,
        margin: float = 0.3,
        lambda_logdet: float = 0.01,
        lambda_anchor: float = 0.0,
        lambda_range: float = 0.0,
    ):
        super().__init__()
        self.lcons = ConsistencyLoss()
        self.ltri = TripletLoss(margin)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_logdet = lambda_logdet
        self.lambda_anchor = lambda_anchor
        self.lambda_range = lambda_range

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        anchor_emb: torch.Tensor,
        cond_emb: torch.Tensor,
        orig_emb: torch.Tensor,
        log_det: torch.Tensor | None = None,
        n_elements: int | None = None,
        xa: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cons = self.lcons(x, x_hat)
        tri = self.ltri(anchor_emb, cond_emb, orig_emb)
        total = self.lambda1 * cons + self.lambda2 * tri

        result = {"total": total, "consistency": cons, "triplet": tri}

        # --- Content preservation on the ANONYMIZATION pass (xa = f(x;c)) ---
        # The triplet only constrains the ASV *embedding*, which a garbled mel
        # satisfies as easily as a clean speaker conversion. Without these two
        # terms the model cheats by scrambling phonetic content and blowing mel
        # values out of range ([-43,+29]), giving unintelligible anonymized audio.
        if xa is not None and self.lambda_anchor > 0:
            # Soft anchor: penalize LARGE deviations from x (garbling) while still
            # permitting the moderate change needed for speaker conversion.
            anchor = self.lcons(x, xa)
            result["anchor"] = anchor
            result["total"] = result["total"] + self.lambda_anchor * anchor
        if xa is not None and self.lambda_range > 0:
            # Keep xa inside the original's per-sample dynamic range (hard stop on
            # the blow-up that no vocoder can reconstruct).
            x_lo = x.amin(dim=(1, 2), keepdim=True)
            x_hi = x.amax(dim=(1, 2), keepdim=True)
            rng = (F.relu(xa - x_hi) + F.relu(x_lo - xa)).mean()
            result["range"] = rng
            result["total"] = result["total"] + self.lambda_range * rng

        if log_det is not None and self.lambda_logdet > 0:
            # Normalize log_det by the actual number of elements summed
            # (mel_channels × T × num_cinn_blocks) to get mean log-scale.
            # This keeps values bounded to [-4, 4] (by _EXP_CLAMP in blocks.py).
            if n_elements is not None and n_elements > 0:
                logdet_normalized = log_det / n_elements
            else:
                # Fallback: at least divide by batch size (better than nothing)
                logdet_normalized = log_det / max(log_det.numel(), 1)
            # Squared penalty: penalizes BOTH expansion (log_det > 0) and
            # contraction (log_det < 0), encouraging volume-preserving
            # transforms (det(J) ≈ 1) which improves numerical invertibility.
            # No clamp needed — per-element values are bounded by [-4, 4].
            logdet_loss = (logdet_normalized ** 2).mean()
            result["logdet"] = logdet_loss
            result["total"] = result["total"] + self.lambda_logdet * logdet_loss

        return result
