"""Conditional Invertible Neural Network blocks (Eq. 1 & 2 in paper).

All operations are 1D along the time axis. Mel shape: (B, 80, T).
"""

import torch
import torch.nn as nn

_EXP_CLAMP = 4.0  # clamp log-scale to [-4, 4] before exp() (§8.1)


# ---------------------------------------------------------------------------
# FiLM conditioning (§2.2)
# ---------------------------------------------------------------------------


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: project condition → (scale, shift)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T), cond: (B, cond_dim)."""
        gamma_beta = self.proj(cond)  # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Dense Block — 4 Conv1d layers with dense connections (§2.2)
# ---------------------------------------------------------------------------


class DenseBlock(nn.Module):
    """4 × [Conv1d → LeakyReLU] with dense connections, then 1×1 compress."""

    def __init__(self, channels: int, growth: int = 32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channels + i * growth, growth, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.compress = nn.Conv1d(channels + 4 * growth, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        return self.compress(torch.cat(feats, dim=1))


# ---------------------------------------------------------------------------
# RRDB — 3 dense blocks + FiLM after each (§2.2)
# ---------------------------------------------------------------------------


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block with FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int, growth: int = 32):
        super().__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(channels, growth) for _ in range(3)])
        self.films = nn.ModuleList([FiLM(cond_dim, channels) for _ in range(3)])
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        for dense, film in zip(self.dense_blocks, self.films):
            h = dense(h)
            h = film(h, cond)
        return x + self.scale * h


# ---------------------------------------------------------------------------
# SubNet — ψ, φ, ρ, η  (3 × RRDB) (§2.2)
# ---------------------------------------------------------------------------


class SubNet(nn.Module):
    """Coupling sub-net with 3 RRDB blocks + FiLM conditioning."""

    def __init__(self, in_channels: int, cond_dim: int, out_channels: int):
        super().__init__()
        self.rrdb_blocks = nn.ModuleList([RRDB(in_channels, cond_dim) for _ in range(3)])
        self.out_proj = nn.Conv1d(in_channels, out_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T), cond: (B, cond_dim) → (B, out_channels, T)."""
        h = x
        for rrdb in self.rrdb_blocks:
            h = rrdb(h, cond)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# Fixed channel permutation (§8.1 — seed=42)
# ---------------------------------------------------------------------------


class FixedPermutation(nn.Module):
    """Fixed random channel permutation for reproducibility."""

    def __init__(self, channels: int, seed: int = 42):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(channels, generator=gen)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.perm]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.inv_perm]


# ---------------------------------------------------------------------------
# CINNBlock — conditional coupling block (Eq. 1 & 2)
# ---------------------------------------------------------------------------


class CINNBlock(nn.Module):
    """
    Single cINN coupling block.

    Forward  (Eq. 1): u_out = u * exp(ψ(v,c)) + φ(v,c)
                      v_out = v * exp(ρ(u_out,c)) + η(u_out,c)
    Inverse  (Eq. 2): v = (v_out − η(u_out,c)) / exp(ρ(u_out,c))
                      u = (u_out − φ(v,c)) / exp(ψ(v,c))
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        half = channels // 2
        self.psi = SubNet(half, cond_dim, half)
        self.phi = SubNet(half, cond_dim, half)
        self.rho = SubNet(half, cond_dim, half)
        self.eta = SubNet(half, cond_dim, half)

    # blocks.py — CINNBlock.forward()
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = x.chunk(2, dim=1)
        log_s1 = self.psi(v, cond).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        u_out = u * torch.exp(log_s1) + self.phi(v, cond)
        log_s2 = self.rho(u_out, cond).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        v_out = v * torch.exp(log_s2) + self.eta(u_out, cond)
        log_det = log_s1.sum(dim=[1, 2]) + log_s2.sum(dim=[1, 2])  # (B,)
        return torch.cat([u_out, v_out], dim=1), log_det

    def inverse(self, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Exact inverse of forward (Eq. 2)."""
        u_out, v_out = y.chunk(2, dim=1)
        log_s2 = self.rho(u_out, cond).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        v = (v_out - self.eta(u_out, cond)) / torch.exp(log_s2)
        log_s1 = self.psi(v, cond).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        u = (u_out - self.phi(v, cond)) / torch.exp(log_s1)
        return torch.cat([u, v], dim=1)


# ---------------------------------------------------------------------------
# INNBlock — unconditional, for ACG (§2.1)
# MLP: [Linear(128,256) → ReLU → Linear(256,256) → ReLU → Linear(256,128)]
# ---------------------------------------------------------------------------


def _acg_mlp(half: int, hidden: int) -> nn.Sequential:
    """ACG sub-net MLP per paper §2.1."""
    net = nn.Sequential(
        nn.Linear(half, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, half),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class INNBlock(nn.Module):
    """Unconditional INN block — used inside ACG."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        hidden = dim  # 256 for dim=256 (§7: ACG sub-net hidden = 256)
        self.s1 = _acg_mlp(half, hidden)
        self.t1 = _acg_mlp(half, hidden)
        self.s2 = _acg_mlp(half, hidden)
        self.t2 = _acg_mlp(half, hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, log_det_jacobian)."""
        x1, x2 = x.chunk(2, dim=-1)
        s1 = self.s1(x1).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        t1 = self.t1(x1)
        y2 = x2 * torch.exp(s1) + t1
        s2 = self.s2(y2).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        t2 = self.t2(y2)
        y1 = x1 * torch.exp(s2) + t2
        log_det = (s1 + s2).sum(dim=-1)
        return torch.cat([y1, y2], dim=-1), log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y1, y2 = y.chunk(2, dim=-1)
        s2 = self.s2(y2).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        t2 = self.t2(y2)
        x1 = (y1 - t2) / torch.exp(s2)
        s1 = self.s1(x1).clamp(-_EXP_CLAMP, _EXP_CLAMP)
        t1 = self.t1(x1)
        x2 = (y2 - t1) / torch.exp(s1)
        return torch.cat([x1, x2], dim=-1)
