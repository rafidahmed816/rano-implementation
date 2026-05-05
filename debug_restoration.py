"""
Comprehensive debugging script for restoration pipeline.

Traces: original audio → anonymization → restoration
Checks: numerical stability, invertibility, vocoding quality
"""

import torch
import torchaudio
from pathlib import Path
import argparse
import json
from tqdm import tqdm

from model import Rano
from audio import MelProcessor


def debug_single_file(args):
    """Debug single file through full pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(use_hifigan=args.use_hifigan)
    model = _load_model(args, device)

    # Load original audio
    print(f"\n[1] Loading audio: {args.input}")
    wav_orig, sr = torchaudio.load(args.input)
    wav_orig = processor.resample(wav_orig.mean(0), sr)
    print(
        f"    Original waveform: shape={wav_orig.shape}, min={wav_orig.min():.6f}, "
        f"max={wav_orig.max():.6f}, mean={wav_orig.mean():.6f}"
    )

    # Convert to mel
    print(f"\n[2] Converting to mel-spectrogram")
    mel_orig = processor.wav_to_mel(wav_orig).to(device)
    if mel_orig.dim() == 4:
        mel_orig = mel_orig.squeeze(1)
    print(f"    Mel shape: {mel_orig.shape}")
    print(
        f"    Mel stats: min={mel_orig.min():.6f}, max={mel_orig.max():.6f}, "
        f"mean={mel_orig.mean():.6f}, std={mel_orig.std():.6f}"
    )

    # Generate random key
    print(f"\n[3] Generating anonymization key")
    key = torch.randn(1, args.embed_dim, device=device)
    print(f"    Key shape: {key.shape}, norm={torch.norm(key).item():.6f}")

    # Anonymize
    print(f"\n[4] Anonymizing mel-spectrogram")
    with torch.no_grad():
        mel_anon, cond = model.anonymize(mel_orig, key)
    print(
        f"    Anonymized mel: min={mel_anon.min():.6f}, max={mel_anon.max():.6f}, "
        f"mean={mel_anon.mean():.6f}, std={mel_anon.std():.6f}"
    )
    print(f"    Has NaN: {torch.isnan(mel_anon).any()}, Has Inf: {torch.isinf(mel_anon).any()}")
    print(f"    Condition embedding: norm={torch.norm(cond).item():.6f}")

    # Vocod anonymized mel
    print(f"\n[5] Vocoding anonymized mel-spectrogram")
    wav_anon = processor.mel_to_wav(mel_anon.cpu())
    print(
        f"    Anonymized waveform: shape={wav_anon.shape}, min={wav_anon.min():.6f}, "
        f"max={wav_anon.max():.6f}, mean={wav_anon.mean():.6f}"
    )
    if wav_anon.abs().max() < 1e-5:
        print(f"    [WARNING] Anonymized waveform is nearly silent!")

    # Restore
    print(f"\n[6] Restoring mel-spectrogram (should recover original)")
    with torch.no_grad():
        mel_restored = model.restore(mel_anon.to(device), key)
    print(
        f"    Restored mel: min={mel_restored.min():.6f}, max={mel_restored.max():.6f}, "
        f"mean={mel_restored.mean():.6f}, std={mel_restored.std():.6f}"
    )
    print(
        f"    Has NaN: {torch.isnan(mel_restored).any()}, Has Inf: {torch.isinf(mel_restored).any()}"
    )

    # Check invertibility
    print(f"\n[7] Checking invertibility (mel_orig vs mel_restored)")
    diff = (mel_orig - mel_restored).abs()
    print(f"    L1 diff: {diff.mean().item():.6e}")
    print(f"    L2 diff: torch.norm(diff): {torch.norm(diff).item():.6e}")
    print(f"    Max diff: {diff.max().item():.6e}")
    if diff.mean().item() > 1e-3:
        print(f"    [WARNING] High reconstruction error! cINN may not be invertible.")

    # Vocod restored mel
    print(f"\n[8] Vocoding restored mel-spectrogram")
    wav_restored = processor.mel_to_wav(mel_restored.cpu())
    print(
        f"    Restored waveform: shape={wav_restored.shape}, min={wav_restored.min():.6f}, "
        f"max={wav_restored.max():.6f}, mean={wav_restored.mean():.6f}"
    )
    if wav_restored.abs().max() < 1e-5:
        print(f"    [ERROR] Restored waveform is nearly silent!")
        print(f"    This is the root cause of blank audio output.")

    # Compare original vs restored
    print(f"\n[9] Comparing original waveform vs restored waveform")
    print(
        f"    Original: shape={wav_orig.shape}, min={wav_orig.min():.6f}, "
        f"max={wav_orig.max():.6f}"
    )
    print(
        f"    Restored: shape={wav_restored.shape}, min={wav_restored.min():.6f}, "
        f"max={wav_restored.max():.6f}"
    )

    # Save debug files
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if wav_orig.abs().max() > 1e-5:
        torchaudio.save(
            str(out_dir / "01_original.wav"), wav_orig.unsqueeze(0), processor.sample_rate
        )
    if wav_anon.abs().max() > 1e-5:
        torchaudio.save(str(out_dir / "02_anonymized.wav"), wav_anon, processor.sample_rate)
    if wav_restored.abs().max() > 1e-5:
        torchaudio.save(str(out_dir / "03_restored.wav"), wav_restored, processor.sample_rate)

    # Save mel spectrograms for inspection
    torch.save(
        {"orig": mel_orig, "anon": mel_anon, "restored": mel_restored}, out_dir / "mel_debug.pt"
    )
    print(f"\n[10] Debug files saved to {out_dir}/")
    print(f"      - 01_original.wav")
    print(f"      - 02_anonymized.wav")
    print(f"      - 03_restored.wav")
    print(f"      - mel_debug.pt (torch tensor dict)")


def debug_full_pipeline(args):
    """Debug full anonymize → restore pipeline from directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(use_hifigan=args.use_hifigan)
    model = _load_model(args, device)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.wav"))[:3]  # Debug first 3 files

    print(f"\n[DEBUG] Full pipeline test with {len(files)} files")

    for file_idx, path in enumerate(files):
        print(f"\n{'='*70}")
        print(f"File {file_idx + 1}/{len(files)}: {path.name}")
        print(f"{'='*70}")

        try:
            # Load and anonymize
            wav, sr = torchaudio.load(path)
            wav = processor.resample(wav.mean(0), sr)
            mel = processor.wav_to_mel(wav).to(device)
            if mel.dim() == 4:
                mel = mel.squeeze(1)

            key = torch.randn(1, args.embed_dim, device=device)
            mel_anon, _ = model.anonymize(mel, key)
            wav_anon = processor.mel_to_wav(mel_anon.cpu())

            # Save anonymized
            anon_path = output_dir / f"{path.stem}_anon.wav"
            torchaudio.save(str(anon_path), wav_anon, processor.sample_rate)
            print(f"  ✓ Anonymized: {anon_path.name} (energy: {wav_anon.abs().max():.6f})")

            # Restore
            mel_restored = model.restore(mel_anon.to(device), key)
            diff = (mel - mel_restored).abs().mean()
            print(f"  ✓ Restored mel: L1 diff = {diff:.6e}")

            wav_restored = processor.mel_to_wav(mel_restored.cpu())
            rest_path = output_dir / f"{path.stem}_restored.wav"
            torchaudio.save(str(rest_path), wav_restored, processor.sample_rate)
            print(f"  ✓ Restored: {rest_path.name} (energy: {wav_restored.abs().max():.6f})")

            if wav_restored.abs().max() < 1e-5:
                print(f"  ✗ [WARNING] Restored waveform is silent!")

        except Exception as e:
            print(f"  ✗ Error: {e}")


def _load_model(args, device):
    model = Rano(embed_dim=args.embed_dim, num_cinn_blocks=args.num_cinn_blocks).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Debug restoration pipeline")
    p.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="single: debug one file; batch: debug multiple files",
    )
    p.add_argument("--input", required=True, help="Input audio file or directory")
    p.add_argument("--output", required=True, help="Output directory for debug files")
    p.add_argument("--checkpoint", required=True, help="Checkpoint file (rano_final.pt)")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_cinn_blocks", type=int, default=12)
    p.add_argument(
        "--use_hifigan",
        action="store_true",
        default=True,
        help="Use HiFi-GAN vocoder (default: True)",
    )

    args = p.parse_args()

    if args.mode == "single":
        debug_single_file(args)
    else:
        debug_full_pipeline(args)
