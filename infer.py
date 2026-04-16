"""
CLI: anonymize or restore audio files.

Usage:
  # Anonymize (speaker-level: same key per speaker folder)
  uv run scripts/infer.py anonymize --input audio/ --output anon/ --checkpoint rano_final.pt

  # Restore (requires the same key used during anonymization)
  uv run scripts/infer.py restore --input anon/ --output restored/ --checkpoint rano_final.pt --key_file keys.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
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
            wav, sr = torchaudio.load(path)
            wav = processor.resample(wav.mean(0), sr)
            mel = processor.wav_to_mel(wav).to(device)
            xa, _ = model.anonymize(mel, key)

            out_path = out_dir / path.relative_to(args.input).with_suffix(".wav")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            anon_wav = processor.mel_to_wav_grifflim(xa.cpu())
            torchaudio.save(str(out_path), anon_wav, processor.sample_rate)

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
            key = torch.tensor(key_store[spk], device=device).unsqueeze(0)
            wav, sr = torchaudio.load(path)
            wav = processor.resample(wav.mean(0), sr)
            mel = processor.wav_to_mel(wav).to(device)
            xr = model.restore(mel, key)

            out_path = out_dir / path.relative_to(args.input)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            restored_wav = processor.mel_to_wav_grifflim(xr.cpu())
            torchaudio.save(str(out_path), restored_wav, processor.sample_rate)


def _load_model(args, device):
    model = Rano(embed_dim=args.embed_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
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
        sp.add_argument("--embed_dim", type=int, default=256)
        if cmd == "restore":
            sp.add_argument("--key_file", required=True)

    args = p.parse_args()
    if args.command == "anonymize":
        anonymize_dir(args)
    elif args.command == "restore":
        restore_dir(args)
    else:
        p.print_help()
