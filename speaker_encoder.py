"""Speaker encoders for Rano (§1.2, §2.4).
Training ASV:   AdaIN-VC speaker encoder (pre-trained, frozen).
Evaluation ASV: ECAPA-TDNN from SpeechBrain (separate, never used in training).
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# SpeechBrain tries to create symlinks on Windows which requires elevated privileges.
# Force copy strategy instead (only affects ECAPA-TDNN evaluation encoder).
os.environ.setdefault("SPEECHBRAIN_LOCAL_STRATEGY", "copy")


# ---------------------------------------------------------------------------
# AdaIN-VC Speaker Encoder (training ASV) — §2.4
# Architecture mirrors github.com/jjery2243542/adaptive_voice_conversion
# Load pre-trained weights via: encoder.load_state_dict(torch.load(...))
# ---------------------------------------------------------------------------

class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling: attention-weighted mean + std over time.
    Captures far more speaker information than plain global average pooling."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, in_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        w = torch.softmax(self.attn(x), dim=-1)          # (B, C, T)
        mu = (w * x).sum(-1)                              # (B, C)
        std = ((w * x ** 2).sum(-1) - mu ** 2).clamp(min=1e-4).sqrt()
        return torch.cat([mu, std], dim=1)               # (B, 2C)


class AdaINVCSpeakerEncoder(nn.Module):
    """[UPGRADED] Strong mel-based speaker encoder — x-vector TDNN + attentive
    statistics pooling → 256-d L2-normalised embedding.

    This replaces the original shallow 4-conv + global-average-pool encoder, whose
    weakness ceilinged the anonymization EER (it was easily fooled without the
    change transferring to a real ASV). The class NAME is kept so model.py and the
    train_*.py scripts import it unchanged. Train it with AAMSoftmax (below).
    """

    def __init__(self, mel_channels: int = 80, embed_dim: int = 256, channels: int = 512):
        super().__init__()

        def tdnn(ci, co, k, d):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, dilation=d, padding=((k - 1) // 2) * d),
                nn.BatchNorm1d(co), nn.ReLU(inplace=True),
            )

        self.frame = nn.Sequential(
            tdnn(mel_channels, channels, 5, 1),
            tdnn(channels, channels, 3, 2),
            tdnn(channels, channels, 3, 3),
            tdnn(channels, channels, 1, 1),
            nn.Conv1d(channels, 1500, 1), nn.BatchNorm1d(1500), nn.ReLU(inplace=True),
        )
        self.pool = AttentiveStatsPool(1500)
        self.bn = nn.BatchNorm1d(3000)
        self.fc = nn.Linear(3000, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, 80, T) → speaker embedding: (B, embed_dim), L2-normalised."""
        h = self.frame(mel)
        h = self.pool(h)
        emb = self.fc(self.bn(h))
        return F.normalize(emb, dim=-1)


class AAMSoftmax(nn.Module):
    """Additive Angular Margin softmax (ArcFace) head for speaker classification.
    Produces much more discriminative embeddings than plain cross-entropy.
    Returns logits; use with F.cross_entropy. Discarded after ASV training."""

    def __init__(self, embed_dim: int, num_speakers: int,
                 margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_speakers, embed_dim))
        nn.init.xavier_normal_(self.W)
        self.margin = margin
        self.scale = scale

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        margin_cos = torch.cos(theta + self.margin)
        onehot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        return self.scale * (onehot * margin_cos + (1.0 - onehot) * cos)


# ---------------------------------------------------------------------------
# ECAPA-TDNN wrapper (evaluation only) — §5.1
# Uses SpeechBrain pre-trained model: speechbrain/spkrec-ecapa-voxceleb
# ---------------------------------------------------------------------------

class ECAPATDNNEncoder(nn.Module):
    """
    SpeechBrain ECAPA-TDNN wrapper for evaluation.

    This encoder is used ONLY for computing EER and GVD metrics (§5.1).
    It must NEVER be used during training.

    Requires: pip install speechbrain
    The model is downloaded automatically on first use.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        from huggingface_hub import snapshot_download
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        savedir = "pretrained_models/spkrec-ecapa-voxceleb"
        snapshot_download(
            "speechbrain/spkrec-ecapa-voxceleb",
            local_dir=savedir,
            local_dir_use_symlinks=False,
        )
        # SpeechBrain needs "cuda:0" not "cuda"
        if device.startswith("cuda") and ":" not in device:
            device = "cuda:0"
        self.model = EncoderClassifier.from_hparams(
            source=savedir,
            savedir=savedir,
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, num_samples) raw waveform at 16 kHz → (B, 192) embedding."""
        emb = self.model.encode_batch(wav)
        return F.normalize(emb.squeeze(1), dim=-1)


# ---------------------------------------------------------------------------
# Backward-compatible alias (for existing code that imports SpeakerEncoder)
# ---------------------------------------------------------------------------

SpeakerEncoder = AdaINVCSpeakerEncoder
