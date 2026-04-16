"""Evaluate Rano: EER, WER, GVD, pitch correlation (Sec. IV-B, Table I)."""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from model import Rano
from audio import MelProcessor
from metrics import compute_eer, compute_gvd, compute_pitch_correlation, extract_f0
from speaker_encoder import SpeakerEncoder


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()

    model = Rano(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Use fixed per-speaker keys for speaker-level anonymization
    speaker_keys: dict[str, torch.Tensor] = {}

    orig_embeddings: dict[str, list] = defaultdict(list)
    anon_embeddings: dict[str, list] = defaultdict(list)
    pitch_corrs = []
    eer_scores, eer_labels = [], []

    test_files = sorted(Path(args.test_dir).rglob("*.wav")) + sorted(
        Path(args.test_dir).rglob("*.flac")
    )

    eval_asv = SpeakerEncoder(embed_dim=args.embed_dim).to(device)
    eval_asv.load_state_dict(torch.load(args.eval_asv_checkpoint, map_location=device))
    eval_asv.eval()

    with torch.no_grad():
        for path in tqdm(test_files, desc="Evaluating"):
            spk = path.parent.name
            wav, sr = torchaudio.load(path)
            wav = processor.resample(wav.mean(0), sr)
            mel = processor.wav_to_mel(wav).to(device)

            if spk not in speaker_keys:
                speaker_keys[spk] = torch.randn(1, args.embed_dim, device=device)
            key = speaker_keys[spk]

            xa, _ = model.anonymize(mel, key)

            orig_emb = eval_asv(mel.squeeze(0)).cpu()
            anon_emb = eval_asv(xa.squeeze(0)).cpu()

            orig_embeddings[spk].append(orig_emb)
            anon_embeddings[spk].append(anon_emb)

            # Pitch correlation
            wav_np = wav.numpy()
            xa_wav = processor.mel_to_wav_grifflim(xa.cpu())
            f0_orig = extract_f0(wav_np, processor.sample_rate, processor.hop_length)
            f0_anon = extract_f0(
                xa_wav.squeeze().numpy(), processor.sample_rate, processor.hop_length
            )
            pitch_corrs.append(compute_pitch_correlation(f0_orig, f0_anon))

    # Aggregate embeddings per speaker
    orig_mean = {k: torch.stack(v).mean(0) for k, v in orig_embeddings.items()}
    anon_mean = {k: torch.stack(v).mean(0) for k, v in anon_embeddings.items()}

    gvd = compute_gvd(orig_mean, anon_mean)
    rho_f0 = float(np.mean(pitch_corrs))

    print(f"\n{'='*40}")
    print(f"GVD:             {gvd:.2f} dB  (ideal ≥ 0)")
    print(f"Pitch Corr ρf0:  {rho_f0:.3f}  (ideal → 1)")
    print(f"  (Run ASV evaluation separately for EER)")
    print(f"  (Run Whisper separately for WER)")
    print(f"{'='*40}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eval_asv_checkpoint", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--output_dir", type=str, default="eval_outputs")
    args = p.parse_args()
    evaluate(args)
