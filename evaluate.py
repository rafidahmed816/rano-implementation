"""Evaluate Rano: EER, GVD, pitch correlation (§IV-B, Table I).

Usage:
  python evaluate.py \
    --acg_checkpoint checkpoints/acg/acg_best.pt \
    --anonymizer_ckpt checkpoints/rano/anonymizer_final.pt \
    --test_dir data/test/
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from model import Rano
from audio import MelProcessor
from speaker_encoder import ECAPATDNNEncoder

_ECAPA_SR = 16000  # ECAPA-TDNN trained at 16 kHz


# ---------------------------------------------------------------------------
# Metrics (inlined — no metrics.py dependency)
# ---------------------------------------------------------------------------

def compute_eer(scores: list[float], labels: list[int]) -> float:
    """EER (%) from cosine similarity scores + binary same-speaker labels."""
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer) * 100


def compute_gvd(orig_mean: dict, anon_mean: dict) -> float:
    """GVD (dB): gain in voice distinctiveness after anonymization."""
    spks = list(orig_mean.keys())
    if len(spks) < 2:
        return 0.0

    def mean_cosine_dist(embs):
        vecs = F.normalize(torch.stack(list(embs.values())), dim=-1)
        dists = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dists.append((1 - (vecs[i] * vecs[j]).sum().item()))
        return float(np.mean(dists)) if dists else 0.0

    d_orig = mean_cosine_dist(orig_mean)
    d_anon = mean_cosine_dist(anon_mean)
    return float(10 * np.log10(d_anon / d_orig)) if d_orig > 0 else 0.0


def extract_f0(wav_np: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    import librosa
    f0, _, _ = librosa.pyin(
        wav_np.astype(np.float64),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    return f0


def compute_pitch_correlation(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    n = min(len(f0_a), len(f0_b))
    f0_a, f0_b = f0_a[:n], f0_b[:n]
    voiced = ~(np.isnan(f0_a) | np.isnan(f0_b))
    if voiced.sum() < 2:
        return 0.0
    return float(np.corrcoef(f0_a[voiced], f0_b[voiced])[0, 1])


# ---------------------------------------------------------------------------
# Model loading (mirrors quick_infer.py)
# ---------------------------------------------------------------------------

def _load_model(args, device):
    model = Rano(embed_dim=args.embed_dim, num_cinn_blocks=args.num_cinn_blocks).to(device)

    acg_path = Path(args.acg_checkpoint)
    if acg_path.exists():
        model.acg.load_state_dict(torch.load(acg_path, map_location=device))
        print(f"Loaded ACG from {acg_path}")
    else:
        print(f"[WARN] ACG not found: {acg_path} — using random weights")

    anon_path = Path(args.anonymizer_ckpt)
    if anon_path.exists():
        model.anonymizer.load_state_dict(torch.load(anon_path, map_location=device))
        print(f"Loaded Anonymizer from {anon_path}")
    else:
        raise FileNotFoundError(f"Anonymizer checkpoint not found: {anon_path}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(device=device)
    model = _load_model(args, device)

    # ECAPA-TDNN: evaluation ASV — completely separate from training ASV (§IV-B).
    # Auto-downloaded from speechbrain/spkrec-ecapa-voxceleb on first run.
    sb_device = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"
    eval_asv = ECAPATDNNEncoder(device=sb_device).to(device)

    speaker_keys: dict[str, torch.Tensor] = {}
    orig_embs_per_utt: dict[str, list] = defaultdict(list)   # for EER
    anon_embs_per_utt: dict[str, list] = defaultdict(list)   # for EER
    orig_embs_mean: dict[str, list] = defaultdict(list)      # for GVD
    anon_embs_mean: dict[str, list] = defaultdict(list)      # for GVD
    pitch_corrs: list[float] = []

    test_files = sorted(
        f for ext in ("*.wav", "*.flac")
        for f in Path(args.test_dir).rglob(ext)
    )
    if not test_files:
        raise FileNotFoundError(f"No wav/flac files found in {args.test_dir}")

    with torch.no_grad():
        for path in tqdm(test_files, desc="Evaluating"):
            spk = path.parent.name
            wav, sr = torchaudio.load(str(path))
            wav = processor.resample(wav.mean(0), sr)   # (T,) at 22050 Hz

            mel = processor.wav_to_mel(wav).to(device)  # (1, 80, T_mel)

            if spk not in speaker_keys:
                speaker_keys[spk] = torch.randn(1, args.embed_dim, device=device)
            xa, _ = model.anonymize(mel, speaker_keys[spk])  # (1, 80, T_mel)

            # Vocode anonymized mel → waveform at 22050 Hz
            xa_wav_22k = processor.mel_to_wav(xa.cpu()).squeeze(0)  # (T_wav,)

            # Resample both to 16 kHz for ECAPA-TDNN
            wav_16k = torchaudio.functional.resample(wav, processor.sample_rate, _ECAPA_SR)
            xa_wav_16k = torchaudio.functional.resample(xa_wav_22k, processor.sample_rate, _ECAPA_SR)

            orig_emb = eval_asv(wav_16k.unsqueeze(0).to(device)).squeeze(0).cpu()
            anon_emb = eval_asv(xa_wav_16k.unsqueeze(0).to(device)).squeeze(0).cpu()

            orig_embs_per_utt[spk].append(orig_emb)
            anon_embs_per_utt[spk].append(anon_emb)
            orig_embs_mean[spk].append(orig_emb)
            anon_embs_mean[spk].append(anon_emb)

            # Pitch correlation at 22050 Hz
            f0_orig = extract_f0(wav.numpy(), processor.sample_rate, processor.hop_length)
            f0_anon = extract_f0(xa_wav_22k.numpy(), processor.sample_rate, processor.hop_length)
            pitch_corrs.append(compute_pitch_correlation(f0_orig, f0_anon))

    # --- EER ---
    # Enroll = mean original embedding per speaker.
    # Test = per-utterance anonymized embeddings.
    # Label = 1 if enroll speaker == test speaker.
    enroll = {s: torch.stack(v).mean(0) for s, v in orig_embs_per_utt.items()}
    eer_scores, eer_labels = [], []
    for enroll_spk, enroll_emb in enroll.items():
        for test_spk, utt_embs in anon_embs_per_utt.items():
            for utt_emb in utt_embs:
                score = F.cosine_similarity(
                    enroll_emb.unsqueeze(0), utt_emb.unsqueeze(0)
                ).item()
                eer_scores.append(score)
                eer_labels.append(1 if enroll_spk == test_spk else 0)

    eer = (
        compute_eer(eer_scores, eer_labels)
        if len(set(eer_labels)) == 2
        else float("nan")
    )

    # --- GVD ---
    orig_mean = {s: torch.stack(v).mean(0) for s, v in orig_embs_mean.items()}
    anon_mean = {s: torch.stack(v).mean(0) for s, v in anon_embs_mean.items()}
    gvd = compute_gvd(orig_mean, anon_mean)

    # --- Print results ---
    rho_f0 = float(np.mean(pitch_corrs))
    print(f"\n{'='*42}")
    print(f"  Speakers evaluated : {len(enroll)}")
    print(f"  Utterances         : {len(test_files)}")
    print(f"{'='*42}")
    print(f"  EER         : {eer:.2f}%   (↑ = better anonymization)")
    print(f"  GVD         : {gvd:.2f} dB  (↑ = more diverse anonymous voices)")
    print(f"  Pitch corr  : {rho_f0:.3f}   (→ 1 = SII preserved)")
    print(f"{'='*42}")
    print("  Note: run Whisper separately for WER (SII intelligibility)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--acg_checkpoint",  default="checkpoints/acg/acg_best.pt")
    p.add_argument("--anonymizer_ckpt", required=True)
    p.add_argument("--test_dir",        required=True)
    p.add_argument("--embed_dim",       type=int, default=256)
    p.add_argument("--num_cinn_blocks", type=int, default=12)
    args = p.parse_args()
    evaluate(args)
