"""
Security evaluation: simulate key-attack restoration attempts (§4.3, Table III).
Measures speaker similarity and MCD between illegally restored and original speech.

Expected outcome: Simspk < 15% and MCD > 5 dB for all fake-key attempts.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from model import Rano
from audio import MelProcessor
from speaker_encoder import ECAPATDNNEncoder


def mcd(mel_orig: torch.Tensor, mel_restored: torch.Tensor) -> float:
    """Mel-Cepstral Distortion between two mel-spectrograms (in dB)."""
    diff = (mel_orig - mel_restored) ** 2
    return float((10 / np.log(10)) * torch.sqrt(2 * diff.mean()).item())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()) * 100


def run_security_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()

    model = Rano(embed_dim=args.embed_dim, num_cinn_blocks=args.num_cinn_blocks).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # §5.1: Use ECAPA-TDNN for evaluation
    eval_asv = ECAPATDNNEncoder(device=str(device))

    test_files = (
        sorted(Path(args.test_dir).rglob("*.wav")) + sorted(Path(args.test_dir).rglob("*.flac"))
    )[: args.num_samples]

    # Attack distance buckets matching Table III
    buckets = [(0.3, 0.4), (0.2, 0.3), (0.1, 0.2)]
    results = {b: {"sim": [], "mcd": []} for b in buckets}

    with torch.no_grad():
        for path in tqdm(test_files, desc="Security eval"):
            wav, sr = torchaudio.load(path)
            wav = processor.resample(wav.mean(0), sr)
            mel = processor.wav_to_mel(wav).to(device)
            if mel.dim() == 4:
                mel = mel.squeeze(1)

            # Anonymize with correct key
            true_key = torch.randn(1, args.embed_dim, device=device)
            xa, true_cond = model.anonymize(mel, true_key)

            # §4.3: attempt restoration with 100 randomly sampled fake keys
            for _ in range(args.attack_trials):
                fake_key = torch.randn(1, args.embed_dim, device=device)
                fake_cond = model.acg.generate(fake_key)
                d_key = float(
                    1 - torch.nn.functional.cosine_similarity(true_cond, fake_cond).item()
                )

                for lo, hi in buckets:
                    if lo < d_key <= hi:
                        xr_fake = model.anonymizer.inverse(xa, fake_cond)

                        # Compare using ECAPA-TDNN on waveforms
                        orig_wav = processor.mel_to_wav_grifflim(mel.cpu())
                        fake_wav = processor.mel_to_wav_grifflim(xr_fake.cpu())

                        orig_wav_16k = torchaudio.functional.resample(
                            orig_wav.squeeze(), processor.sample_rate, 16000
                        )
                        fake_wav_16k = torchaudio.functional.resample(
                            fake_wav.squeeze(), processor.sample_rate, 16000
                        )

                        orig_emb = eval_asv(orig_wav_16k.unsqueeze(0)).squeeze()
                        fake_emb = eval_asv(fake_wav_16k.unsqueeze(0)).squeeze()

                        results[(lo, hi)]["sim"].append(cosine_sim(orig_emb, fake_emb))
                        results[(lo, hi)]["mcd"].append(mcd(mel.squeeze(), xr_fake.squeeze()))

    print("\nSecurity Evaluation Results (Table III):")
    print(f"{'Case':<25} {'Sim_spk (%)':<12} {'MCD (dB)':<10}")
    print("-" * 50)
    for (lo, hi), data in results.items():
        if data["sim"]:
            sim = np.mean(data["sim"])
            m = np.mean(data["mcd"])
            status = "✓" if sim < 15 and m > 5 else "✗"
            print(f"{lo:.1f} < D_key < {hi:.1f}         {sim:>8.2f}%   {m:>8.2f}  {status}")
        else:
            print(f"{lo:.1f} < D_key < {hi:.1f}         {'N/A':>12} {'N/A':>10}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_cinn_blocks", type=int, default=12)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--attack_trials", type=int, default=100)  # §4.3: 100 fake keys
    args = p.parse_args()
    run_security_eval(args)
