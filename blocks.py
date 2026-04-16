"""Conditional Invertible Neural Network blocks (Eq. 1 & 2 in paper)."""

import torch
import torch.nn as nn
from einops import rearrange


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block used inside coupling layer sub-nets."""

    def __init__(self, channels: int, growth: int = 32, num_dense: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_dense):
            in_ch = channels + i * growth
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, growth, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.compress = nn.Conv2d(channels + num_dense * growth, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=1)))
        out = self.compress(torch.cat(feats, dim=1))
        return x + self.scale * out


class SubNet(nn.Module):
    """Coupling sub-net: ψ, φ, ρ, η — takes (x, condition) → transform."""

    def __init__(self, in_channels: int, cond_dim: int, out_channels: int):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, in_channels)
        self.rrdb = RRDB(in_channels)
        self.out_proj = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) → (B, C, 1, 1)
        c = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        return self.out_proj(self.rrdb(x + c))


class CINNBlock(nn.Module):
    """
    Single cINN coupling block (Eq. 1 forward, Eq. 2 inverse).

    Split channels into (u, v) → apply affine transforms conditioned on `c`.
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        half = channels // 2
        self.psi = SubNet(half, cond_dim, half)   # exp scale for v→u path
        self.phi = SubNet(half, cond_dim, half)   # shift for v→u path
        self.rho = SubNet(half, cond_dim, half)   # exp scale for u→v path
        self.eta = SubNet(half, cond_dim, half)   # shift for u→v path

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Anonymizer forward pass."""
        u, v = x.chunk(2, dim=1)
        u_out = v * torch.exp(self.psi(v, cond)) + self.phi(v, cond)
        v_out = v * torch.exp(self.rho(u_out, cond)) + self.eta(u_out, cond)
        return torch.cat([u_out, v_out], dim=1)

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Restorer backward pass — exact inverse of forward."""
        u_out, v_out = y.chunk(2, dim=1)
        v = (v_out - self.eta(u_out, cond)) / torch.exp(self.rho(u_out, cond))
        u = (u_out - self.phi(v, cond)) / torch.exp(self.psi(v, cond))
        return torch.cat([u, v], dim=1)


class INNBlock(nn.Module):
    """Unconditional INN block — used inside ACG."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        self.s1 = nn.Sequential(nn.Linear(half, half), nn.Tanh(), nn.Linear(half, half))
        self.t1 = nn.Sequential(nn.Linear(half, half), nn.ReLU(), nn.Linear(half, half))
        self.s2 = nn.Sequential(nn.Linear(half, half), nn.Tanh(), nn.Linear(half, half))
        self.t2 = nn.Sequential(nn.Linear(half, half), nn.ReLU(), nn.Linear(half, half))
        for m in [self.s1, self.s2, self.t1, self.t2]:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, log_det_jacobian)."""
        x1, x2 = x.chunk(2, dim=-1)
        s1, t1 = self.s1(x1), self.t1(x1)
        y2 = x2 * torch.exp(s1) + t1
        s2, t2 = self.s2(y2), self.t2(y2)
        y1 = x1 * torch.exp(s2) + t2
        log_det = (s1 + s2).sum(dim=-1)
        return torch.cat([y1, y2], dim=-1), log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=-1)
        s2, t2 = self.s2(y2), self.t2(y2)
        x1 = (y1 - t2) / torch.exp(s2)
        s1, t1 = self.s1(x1), self.t1(x1)
        x2 = (y2 - t1) / torch.exp(s1)
        return torch.cat([x1, x2], dim=-1)
