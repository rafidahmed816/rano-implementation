"""
Security evaluation: simulate key-attack restoration attempts (Sec. IV-D, Table III).
Measures speaker similarity and MCD between illegally restored and original speech.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from model import Rano
from audio import MelProcessor
from speaker_encoder import SpeakerEncoder


def mcd(mel_orig: torch.Tensor, mel_restored: torch.Tensor) -> float:
    """Mel-Cepstral Distortion between two mel-spectrograms (in dB)."""
    diff = (mel_orig - mel_restored) ** 2
    return float((10 / np.log(10)) * torch.sqrt(2 * diff.mean()).item())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()) * 100


def run_security_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()

    model = Rano(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    eval_asv = SpeakerEncoder(embed_dim=args.embed_dim).to(device)
    eval_asv.load_state_dict(torch.load(args.eval_asv_checkpoint, map_location=device))
    eval_asv.eval()

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

            # Anonymize with correct key
            true_key = torch.randn(1, args.embed_dim, device=device)
            xa, true_cond = model.anonymize(mel, true_key)

            # Attack: try to restore with fake keys
            for _ in range(args.attack_trials):
                fake_key = torch.randn(1, args.embed_dim, device=device)
                fake_cond = model.acg.generate(fake_key)
                d_key = float(
                    1 - torch.nn.functional.cosine_similarity(true_cond, fake_cond).item()
                )

                for lo, hi in buckets:
                    if lo < d_key <= hi:
                        # Restore with wrong key
                        xr_fake = model.anonymizer.inverse(
                            xa, fake_cond.unsqueeze(-1).unsqueeze(-1)
                        )
                        # Compare with original
                        orig_emb = eval_asv(mel.squeeze(0))
                        fake_emb = eval_asv(xr_fake.squeeze(0))
                        results[(lo, hi)]["sim"].append(cosine_sim(orig_emb, fake_emb))
                        results[(lo, hi)]["mcd"].append(mcd(mel.squeeze(), xr_fake.squeeze()))

    print("\nSecurity Evaluation Results (Table III):")
    print(f"{'Case':<25} {'Sim_spk (%)':>12} {'MCD (dB)':>10}")
    print("-" * 50)
    for (lo, hi), data in results.items():
        if data["sim"]:
            sim = np.mean(data["sim"])
            m = np.mean(data["mcd"])
            print(f"{lo:.1f} < D_key < {hi:.1f}         {sim:>12.2f} {m:>10.2f}")
        else:
            print(f"{lo:.1f} < D_key < {hi:.1f}         {'N/A':>12} {'N/A':>10}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eval_asv_checkpoint", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--attack_trials", type=int, default=10)
    args = p.parse_args()
    run_security_eval(args)
