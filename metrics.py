"""
Evaluation metrics (Sec. IV-B):
- EER (Equal Error Rate) — privacy / anonymization quality
- GVD (Gain of Voice Distinctiveness) — speaker diversity preservation
- WER (Word Error Rate) — content retention (via Whisper)
- Pitch Correlation (ρf0) — prosody preservation
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """EER from cosine similarity scores and binary same/different labels."""
    fpr, tpr, _ = _roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer) * 100


def _roc_curve(labels, scores):
    """Minimal ROC curve without sklearn dependency."""
    thresholds = np.sort(np.unique(scores))[::-1]
    fps, tps = [], []
    P = labels.sum()
    N = len(labels) - P
    for t in thresholds:
        pred = scores >= t
        tps.append((pred & labels.astype(bool)).sum() / (P + 1e-9))
        fps.append((pred & ~labels.astype(bool)).sum() / (N + 1e-9))
    return np.array(fps), np.array(tps), thresholds


def compute_gvd(
    orig_embeddings: dict[str, torch.Tensor],
    anon_embeddings: dict[str, torch.Tensor],
) -> float:
    """
    Gain of Voice Distinctiveness in dB (Sec. IV-B).
    orig/anon_embeddings: {speaker_id: (N, D) tensor of speaker embeddings}.
    """
    def mean_dist(emb_dict: dict) -> float:
        keys = list(emb_dict.keys())
        dists = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = emb_dict[keys[i]].mean(0)
                b = emb_dict[keys[j]].mean(0)
                dists.append(1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        return float(np.mean(dists)) if dists else 0.0

    d_orig = mean_dist(orig_embeddings)
    d_anon = mean_dist(anon_embeddings)
    if d_orig < 1e-9:
        return 0.0
    return 10 * np.log10(d_anon / (d_orig + 1e-9))


def compute_pitch_correlation(f0_orig: np.ndarray, f0_anon: np.ndarray) -> float:
    """Pearson correlation between original and anonymized pitch sequences (ρf0)."""
    min_len = min(len(f0_orig), len(f0_anon))
    f0_orig, f0_anon = f0_orig[:min_len], f0_anon[:min_len]
    voiced = (f0_orig > 0) & (f0_anon > 0)
    if voiced.sum() < 2:
        return 0.0
    corr = np.corrcoef(f0_orig[voiced], f0_anon[voiced])[0, 1]
    return float(corr)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Token-level WER between reference and hypothesis transcripts."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    # Dynamic programming edit distance
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        d[i, 0] = i
    for j in range(len(hyp) + 1):
        d[0, j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(ref), len(hyp)] / (len(ref) + 1e-9) * 100


def extract_f0(wav: np.ndarray, sr: int = 22050, hop: int = 256) -> np.ndarray:
    """Extract fundamental frequency using autocorrelation (RAPT-like)."""
    try:
        import librosa
        f0, _, _ = librosa.pyin(wav, fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=hop)
        return np.nan_to_num(f0)
    except ImportError:
        return np.zeros(len(wav) // hop)
