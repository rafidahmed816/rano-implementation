"""Training losses for Rano — Eq. 5 (consistency) and Eq. 6 (triplet / SDI-differentiation)."""

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
    Ltotal = λ1 * Lcons + λ2 * Ltri
    """

    def __init__(self, lambda1: float = 1.0, lambda2: float = 5.0, margin: float = 0.3):
        super().__init__()
        self.lcons = ConsistencyLoss()
        self.ltri = TripletLoss(margin)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        anchor_emb: torch.Tensor,
        cond_emb: torch.Tensor,
        orig_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cons = self.lcons(x, x_hat)
        tri = self.ltri(anchor_emb, cond_emb, orig_emb)
        total = self.lambda1 * cons + self.lambda2 * tri
        return {"total": total, "consistency": cons, "triplet": tri}
