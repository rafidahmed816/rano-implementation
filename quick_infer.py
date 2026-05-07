"""
CLI: anonymize or restore audio files (§4.1, §4.2).
Same as infer.py but loads ACG and Anonymizer checkpoints separately,
so you can run inference with a mid-training anonymizer_stepXXXXX.pt.

Usage:
  # Anonymize
  python quick_infer.py anonymize --input audio/ --output anon/ --anonymizer_ckpt checkpoints/rano/anonymizer_step10000.pt

  # Restore (requires the same key used during anonymization)
  python quick_infer.py restore --input anon/ --output restored/ --anonymizer_ckpt checkpoints/rano/anonymizer_step10000.pt --key_file anon/keys.json
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm

from model import Rano
from audio import MelProcessor


def _save_audio(path: Path, wav_np: np.ndarray, sr: int, fmt: str) -> Path:
    """Save audio in requested format. Returns actual path written."""
    out = path.with_suffix(f".{fmt}")
    if fmt == "flac":
        sf.write(str(out), wav_np, sr, format="FLAC", subtype="PCM_16")
    elif fmt == "mp3":
        t = torch.from_numpy(wav_np).float().unsqueeze(0)  # (1, T)
        torchaudio.save(str(out), t, sr, format="mp3")
    else:
        sf.write(str(out), wav_np, sr)
    return out


def anonymize_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(device=device)
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
            wav_np, sr = sf.read(str(path))
            wav = torch.from_numpy(wav_np).float()
            if wav.ndim == 2:
                wav = wav.mean(1)  # stereo → mono
            wav = processor.resample(wav, sr)
            mel = processor.wav_to_mel(wav).to(device)
            # Ensure (B, 80, T) shape
            if mel.dim() == 4:
                mel = mel.squeeze(1)

            xa, _ = model.anonymize(mel, key)

            out_path = out_dir / path.relative_to(args.input).with_suffix(f".{args.format}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            anon_wav = processor.mel_to_wav(xa.cpu())
            _save_audio(out_path, anon_wav.squeeze(0).numpy(), processor.sample_rate, args.format)

            # Save exact anonymized mel tensor — required for lossless restoration.
            # Any vocoder is lossy: xa → wav → mel ≠ xa, which breaks the cINN inverse.
            mel_path = out_path.with_suffix(".mel.pt")
            torch.save(xa.cpu(), mel_path)

            key_store[spk] = speaker_keys[spk].cpu().tolist()

    key_file = out_dir / "keys.json"
    with open(key_file, "w") as f:
        json.dump(key_store, f)
    print(f"Keys saved to {key_file} — keep secret for restoration!")


def restore_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(device=device)
    model = _load_model(args, device)

    with open(args.key_file) as f:
        key_store = json.load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for ext in ("*.wav", "*.flac", "*.mp3")
                   for f in Path(args.input).rglob(ext))

    with torch.no_grad():
        for path in tqdm(files, desc="Restoring"):
            spk = path.parent.name
            if spk not in key_store:
                # Fallback: if only one speaker key exists (flat output structure), use it
                if len(key_store) == 1:
                    spk = next(iter(key_store))
                else:
                    print(f"  [WARN] No key for speaker {spk}, skipping {path}")
                    continue
            key = torch.tensor(key_store[spk], device=device)
            if key.dim() == 1:
                key = key.unsqueeze(0)

            # Load saved anonymized mel tensor (saved during anonymize step).
            # The cINN inverse requires the exact xa tensor — not a re-vocoded
            # version — because Griffin-Lim is lossy and breaks invertibility.
            mel_path = path.with_suffix(".mel.pt")
            if mel_path.exists():
                mel = torch.load(mel_path, map_location=device)
            else:
                print(f"  [WARN] No .mel.pt found for {path.name}; re-converting from wav "
                      f"(restoration quality will be degraded — re-run anonymize to fix).")
                wav_np, sr = sf.read(str(path))
                wav = torch.from_numpy(wav_np).float()
                if wav.ndim == 2:
                    wav = wav.mean(1)
                wav = processor.resample(wav, sr)
                mel = processor.wav_to_mel(wav).to(device)
                if mel.dim() == 4:
                    mel = mel.squeeze(1)

            xr = model.restore(mel, key)

            out_path = out_dir / path.relative_to(args.input)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            restored_wav = processor.mel_to_wav(xr.cpu())
            _save_audio(out_path, restored_wav.squeeze(0).numpy(), processor.sample_rate, args.format)


def _load_model(args, device):
    """
    Loads ACG and Anonymizer checkpoints separately.
    This is the only difference from infer.py's _load_model.
    """
    model = Rano(embed_dim=args.embed_dim, num_cinn_blocks=args.num_cinn_blocks).to(device)

    # Load ACG (Stage 1 checkpoint)
    acg_path = Path(args.acg_checkpoint)
    if acg_path.exists():
        model.acg.load_state_dict(torch.load(acg_path, map_location=device))
        print(f"Loaded ACG from {acg_path}")
    else:
        print(f"[WARN] ACG checkpoint not found: {acg_path} — using random weights")

    # Load Anonymizer (mid-training or final)
    anon_path = Path(args.anonymizer_ckpt)
    if anon_path.exists():
        model.anonymizer.load_state_dict(torch.load(anon_path, map_location=device))
        print(f"Loaded Anonymizer from {anon_path}")
    else:
        raise FileNotFoundError(f"Anonymizer checkpoint not found: {anon_path}")

    model.eval()
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    for cmd in ["anonymize", "restore"]:
        sp = sub.add_parser(cmd)
        sp.add_argument("--input",           required=True)
        sp.add_argument("--output",          required=True)
        sp.add_argument("--acg_checkpoint",  default="checkpoints/acg/acg_final.pt")
        sp.add_argument("--anonymizer_ckpt", default="checkpoints/rano/anonymizer_step10000.pt")
        sp.add_argument("--embed_dim",       type=int, default=256)
        sp.add_argument("--num_cinn_blocks", type=int, default=12)
        sp.add_argument("--format", choices=["wav", "flac", "mp3"], default="flac",
                        help="Output audio format (flac=lossless, mp3=needs ffmpeg)")
        if cmd == "restore":
            sp.add_argument("--key_file", required=True)

    args = p.parse_args()
    if args.command == "anonymize":
        anonymize_dir(args)
    elif args.command == "restore":
        restore_dir(args)
    else:
        p.print_help()
