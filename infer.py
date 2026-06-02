"""
CLI: anonymize or restore audio files (§4.1, §4.2).

Usage:
  # Anonymize (speaker-level: same key per speaker folder)
  python infer.py anonymize --input audio/ --output anon/ --checkpoint rano_final.pt

  # Restore (requires the same key used during anonymization)
  python infer.py restore --input anon/ --output restored/ --checkpoint rano_final.pt --key_file keys.json
"""

import argparse
import json
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

from model import Rano
from audio import MelProcessor


def anonymize_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()
    model = _load_model(args, device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.input).rglob("*.wav")) + sorted(Path(args.input).rglob("*.flac"))

    speaker_keys: dict[str, torch.Tensor] = {}
    key_store: dict[str, list] = {}

    with torch.no_grad():
        for path in tqdm(files, desc="Anonymizing"):
            spk = path.parent.name
            if spk not in speaker_keys:
                speaker_keys[spk] = torch.randn(1, args.embed_dim, device=device)

            key = speaker_keys[spk]
            # Load audio using soundfile (no torchcodec needed)
            wav_np, sr = sf.read(str(path))
            wav = torch.from_numpy(wav_np).float()
            if wav.dim() == 2:
                wav = wav.mean(1)
            wav = processor.resample(wav, sr)
            mel = processor.wav_to_mel(wav).to(device)
            # Ensure (B, 80, T) shape
            if mel.dim() == 4:
                mel = mel.squeeze(1)

            xa, _ = model.anonymize(mel, key)

            out_path = out_dir / path.relative_to(args.input).with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Use HiFi-GAN vocoder (§IV-A: "We utilize Hifi-GAN [39] as the vocoder")
            anon_wav = processor.mel_to_wav(xa.cpu())
            # Convert tensor to numpy for soundfile
            anon_wav_np = anon_wav.squeeze(0).numpy() if anon_wav.dim() > 1 else anon_wav.numpy()
            sf.write(str(out_path), anon_wav_np, processor.sample_rate)

            key_store[spk] = speaker_keys[spk].cpu().tolist()

    key_file = out_dir / "keys.json"
    with open(key_file, "w") as f:
        json.dump(key_store, f)
    print(f"Keys saved to {key_file} — keep secret for restoration!")


def restore_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()
    model = _load_model(args, device)

    with open(args.key_file) as f:
        key_store = json.load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.input).rglob("*.wav"))

    with torch.no_grad():
        for path in tqdm(files, desc="Restoring"):
            spk = path.parent.name
            if spk not in key_store:
                print(f"  [WARN] No key for speaker {spk}, skipping {path}")
                continue
            key = torch.tensor(key_store[spk], device=device)
            if key.dim() == 1:
                key = key.unsqueeze(0)
            # Load audio using soundfile (no torchcodec needed)
            wav_np, sr = sf.read(str(path))
            wav = torch.from_numpy(wav_np).float()
            if wav.dim() == 2:
                wav = wav.mean(1)
            wav = processor.resample(wav, sr)
            mel = processor.wav_to_mel(wav).to(device)
            if mel.dim() == 4:
                mel = mel.squeeze(1)

            xr = model.restore(mel, key)

            # Debug: Check mel-spectrogram statistics before vocoding
            print(
                f"[{spk}] Restored mel: min={xr.min():.6f}, max={xr.max():.6f}, "
                f"mean={xr.mean():.6f}, std={xr.std():.6f}"
            )
            print(
                f"  Original mel:  min={mel.min():.6f}, max={mel.max():.6f}, "
                f"mean={mel.mean():.6f}, std={mel.std():.6f}"
            )
            has_nan = torch.isnan(xr).any()
            has_inf = torch.isinf(xr).any()
            print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

            if has_nan or has_inf:
                print(f"  [ERROR] Numerical issue detected! Clipping to safe range...")
                # Clip to reasonable mel range to prevent vocoder instability
                mel_min, mel_max = -12.0, 12.0
                xr = torch.clamp(xr, mel_min, mel_max)
                print(f"  Clipped to [{mel_min}, {mel_max}]")

            out_path = out_dir / path.relative_to(args.input)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Use HiFi-GAN vocoder (§IV-A: "We utilize Hifi-GAN [39] as the vocoder")
            restored_wav = processor.mel_to_wav(xr.cpu())

            # Check vocoded waveform
            print(
                f"  Vocoded wav: min={restored_wav.min():.6f}, max={restored_wav.max():.6f}, mean={restored_wav.mean():.6f}"
            )

            # Save using soundfile (no torchcodec needed)
            restored_wav_np = restored_wav.squeeze(0).numpy() if restored_wav.dim() > 1 else restored_wav.numpy()
            sf.write(str(out_path), restored_wav_np, processor.sample_rate)


def _load_model(args, device):
    model = Rano(embed_dim=args.embed_dim, num_cinn_blocks=args.num_cinn_blocks).to(device)
    
    # Load anonymizer checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device)
    print(f"[INFO] Loaded anonymizer checkpoint: {args.checkpoint}")
    print(f"[INFO] Anonymizer state_dict keys (first 5): {list(state_dict.keys())[:5]}")
    
    # Handle torch.compile() wrapped checkpoints (_orig_mod. prefix)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        print("[INFO] Detected torch.compile() wrapping, removing _orig_mod. prefix...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # Load into model.anonymizer (checkpoint contains only anonymizer weights)
    model.anonymizer.load_state_dict(state_dict, strict=False)
    
    # Load ACG checkpoint (required for restoration)
    if hasattr(args, 'acg_checkpoint') and args.acg_checkpoint:
        try:
            acg_state = torch.load(args.acg_checkpoint, map_location=device)
            print(f"[INFO] Loaded ACG checkpoint: {args.acg_checkpoint}")
            print(f"[INFO] ACG state_dict keys: {list(acg_state.keys())[:3]}")
            
            # Handle torch.compile() wrapped ACG checkpoint too
            if any(k.startswith("_orig_mod.") for k in acg_state.keys()):
                print("[INFO] Detected torch.compile() wrapping in ACG, removing _orig_mod. prefix...")
                acg_state = {k.replace("_orig_mod.", ""): v for k, v in acg_state.items()}
            
            model.acg.load_state_dict(acg_state, strict=False)
            print("[OK] ACG loaded successfully")
        except FileNotFoundError:
            print(f"[WARN] ACG checkpoint not found: {args.acg_checkpoint}, using random initialization")
    
    model.eval()
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    for cmd in ["anonymize", "restore"]:
        sp = sub.add_parser(cmd)
        sp.add_argument("--input", required=True)
        sp.add_argument("--output", required=True)
        sp.add_argument("--checkpoint", required=True)
        sp.add_argument("--acg_checkpoint", type=str, default="checkpoints/acg/acg_best.pt",
                        help="ACG checkpoint for conditioning (required for restoration)")
        sp.add_argument("--embed_dim", type=int, default=256)
        sp.add_argument("--num_cinn_blocks", type=int, default=12)
        if cmd == "restore":
            sp.add_argument("--key_file", required=True)

    args = p.parse_args()
    if args.command == "anonymize":
        anonymize_dir(args)
    elif args.command == "restore":
        restore_dir(args)
    else:
        p.print_help()