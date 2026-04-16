"""Lightweight speaker encoder wrapping AdaIN-VC style ECAPA-TDNN (Sec. III-A)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNNBlock(nn.Module):
    """Time-delay neural network block with dilation."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation,
                              padding=dilation * (kernel - 1) // 2)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class SpeakerEncoder(nn.Module):
    """
    Lightweight speaker encoder: mel-spectrogram → L2-normalised speaker embedding.

    Used both as ASV (contrastive loss) and as real-condition extractor (consistency loss).
    Differentiable — gradients flow through during Rano's second training stage.
    """

    def __init__(self, mel_channels: int = 80, embed_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.tdnns = nn.Sequential(
            TDNNBlock(mel_channels, hidden, 5, 1),
            TDNNBlock(hidden, hidden, 3, 2),
            TDNNBlock(hidden, hidden, 3, 3),
            TDNNBlock(hidden, hidden * 3, 1, 1),
        )
        self.attention = nn.Linear(hidden * 3, 1)
        self.fc = nn.Linear(hidden * 3, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, F, T) → speaker embedding: (B, embed_dim), L2-normalised."""
        # mel expected (B, F, T)
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        h = self.tdnns(mel)                          # (B, H, T)
        w = torch.softmax(self.attention(h.transpose(1, 2)), dim=1)  # (B, T, 1)
        pooled = (h * w.transpose(1, 2)).sum(dim=-1)  # (B, H)
        emb = self.bn(self.fc(pooled))
        return F.normalize(emb, dim=-1)
