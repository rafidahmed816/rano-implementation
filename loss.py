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
    ):
        super().__init__()
        self.lcons = ConsistencyLoss()
        self.ltri = TripletLoss(margin)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_logdet = lambda_logdet

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        anchor_emb: torch.Tensor,
        cond_emb: torch.Tensor,
        orig_emb: torch.Tensor,
        log_det: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cons = self.lcons(x, x_hat)
        tri = self.ltri(anchor_emb, cond_emb, orig_emb)
        total = self.lambda1 * cons + self.lambda2 * tri

        result = {"total": total, "consistency": cons, "triplet": tri}

        if log_det is not None and self.lambda_logdet > 0:
            # Penalise large absolute log-det to encourage volume-preserving
            # transforms, which improves numerical invertibility.
            # NOTE: log_det is summed over [channels, time] × all cINN blocks,
            # so we must normalize per-element to keep it on a sane scale.
            # Without this, values reach -20k+ and explode gradients → NaN.
            logdet_per_elem = log_det / max(log_det.numel(), 1)
            logdet_loss = -logdet_per_elem.mean()
            # Safety clamp to prevent any residual instability
            logdet_loss = logdet_loss.clamp(-100.0, 100.0)
            result["logdet"] = logdet_loss
            result["total"] = total + self.lambda_logdet * logdet_loss

        return result
