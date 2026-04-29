"""Speaker encoders for Rano (§1.2, §2.4).
Training ASV:   AdaIN-VC speaker encoder (pre-trained, frozen).
Evaluation ASV: ECAPA-TDNN from SpeechBrain (separate, never used in training).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# AdaIN-VC Speaker Encoder (training ASV) — §2.4
# Architecture mirrors github.com/jjery2243542/adaptive_voice_conversion
# Load pre-trained weights via: encoder.load_state_dict(torch.load(...))
# ---------------------------------------------------------------------------

class AdaINVCSpeakerEncoder(nn.Module):
    """
    AdaIN-VC style speaker encoder: mel → 256-dim L2-normalised embedding.

    Architecture: stack of 1D convolutions with batch-norm and downsampling,
    followed by global average pooling and a linear projection.

    This must be loaded with pre-trained weights from the AdaIN-VC repo and
    kept FROZEN during all Rano training (§2.4, §8.1).
    """

    def __init__(self, mel_channels: int = 80, embed_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: (80, T) → (128, T)
            nn.Conv1d(mel_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Block 2: (128, T) → (hidden, T/2)
            nn.Conv1d(128, hidden, 4, stride=2, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            # Block 3: (hidden, T/2) → (hidden*2, T/4)
            nn.Conv1d(hidden, hidden * 2, 4, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            # Block 4: (hidden*2, T/4) → (hidden, T/4)
            nn.Conv1d(hidden * 2, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(hidden, embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, 80, T) → speaker embedding: (B, embed_dim), L2-normalised."""
        h = self.encoder(mel)           # (B, hidden, T')
        h = h.mean(dim=-1)             # global average pooling → (B, hidden)
        emb = self.fc(h)               # (B, embed_dim)
        return F.normalize(emb, dim=-1)


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
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.eval()
        # Freeze all parameters
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
