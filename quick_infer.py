
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


# ============================================================
# MEL CLAMPING CONSTANTS
# HiFi-GAN LJSpeech was trained on log-mel values in this range.
# The cINN pushes some frames below -11.5 dB which HiFi-GAN has
# never seen — it produces pure noise for those frames.
# Clamping restores them to the valid vocoder input range without
# affecting speaker identity (which is encoded in mel shape, not
# in extreme out-of-range low-energy frames).
# ============================================================
MEL_MIN = -11.5
MEL_MAX =  2.0


# ============================================================
# AUDIO SAVE HELPER
# ============================================================

def _save_audio(path: Path, wav_np: np.ndarray, sr: int, fmt: str) -> Path:
    out = path.with_suffix(f".{fmt}")
    if fmt == "flac":
        sf.write(str(out), wav_np, sr, format="FLAC", subtype="PCM_16")
    elif fmt == "mp3":
        t = torch.from_numpy(wav_np).float().unsqueeze(0)
        torchaudio.save(str(out), t, sr, format="mp3")
    else:
        sf.write(str(out), wav_np, sr)
    return out


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def _clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k.replace("_orig_mod.", "", 1)
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        cleaned[k] = v
    return cleaned


def _safe_load(model_part: torch.nn.Module, ckpt_path, device, name: str = "model"):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{name} checkpoint not found: {ckpt_path}")
    print(f"Loading {name}: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
    ckpt = _clean_state_dict(ckpt)
    missing, unexpected = model_part.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  [WARN] {name} missing keys: {missing[:3]}")
    if unexpected:
        print(f"  [WARN] {name} unexpected keys: {unexpected[:3]}")
    print(f"  {name} loaded OK")


def _load_model(args, device: torch.device) -> Rano:
    # FIX: num_cinn_blocks default corrected to 12 (was 8 — caused silent
    # architecture mismatch when loading a checkpoint trained with 12 blocks).
    model = Rano(
        embed_dim=args.embed_dim,
        num_cinn_blocks=args.num_cinn_blocks,
    ).to(device)
    _safe_load(model.acg,        args.acg_checkpoint,  device, "ACG")
    _safe_load(model.anonymizer, args.anonymizer_ckpt, device, "Anonymizer")
    model.eval()
    return model


# ============================================================
# ANONYMIZATION
# ============================================================

def anonymize_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(device=device)
    model = _load_model(args, device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = (
        sorted(Path(args.input).rglob("*.wav")) +
        sorted(Path(args.input).rglob("*.flac"))
    )
    if not files:
        raise RuntimeError(f"No .wav or .flac files found in: {args.input}")
    print(f"Found {len(files)} files to anonymize.")

    speaker_keys: dict[str, torch.Tensor] = {}
    key_store:    dict[str, list]          = {}

    with torch.no_grad():
        for path in tqdm(files, desc="Anonymizing"):
            spk = path.parent.name

            # One fixed random key per speaker = speaker-level anonymization.
            # model.anonymize() passes this through ACG internally.
            if spk not in speaker_keys:
                speaker_keys[spk] = torch.randn(1, args.embed_dim, device=device)

            key = speaker_keys[spk]  # (1, embed_dim) Gaussian

            wav_np, sr = sf.read(str(path))
            wav = torch.from_numpy(wav_np).float()
            if wav.ndim == 2:
                wav = wav.mean(1)

            wav = processor.resample(wav, sr)
            mel = processor.wav_to_mel(wav).to(device)
            if mel.dim() == 4:
                mel = mel.squeeze(1)

            # Anonymize: key → ACG → cond → cINN forward (float64 internally)
            xa, _ = model.anonymize(mel, key)

            # FIX: clamp xa to HiFi-GAN's valid input range BEFORE vocoder.
            # The cINN pushes some frames to -20 dB; HiFi-GAN LJSpeech was
            # never trained below -11.5 dB and produces noise for those frames.
            # Clamp BEFORE saving .mel.pt so restoration uses clean xa too.
            xa = xa.clamp(MEL_MIN, MEL_MAX)

            # Build output path, mirroring input folder structure
            out_path = out_dir / path.relative_to(args.input).with_suffix(
                f".{args.format}"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save anonymized audio
            anon_wav = processor.mel_to_wav(xa.cpu())
            _save_audio(out_path, anon_wav.squeeze(0).numpy(),
                        processor.sample_rate, args.format)

            # Save clamped mel tensor for lossless restoration
            # (avoids vocoder round-trip error during restore)
            torch.save(xa.cpu(), out_path.with_suffix(".mel.pt"))

            # Save key as list for JSON serialisation
            key_store[spk] = speaker_keys[spk].cpu().squeeze(0).tolist()

    keys_path = out_dir / "keys.json"
    with open(keys_path, "w") as f:
        json.dump(key_store, f, indent=2)
    print(f"\nAnonymization complete.")
    print(f"Keys saved to: {keys_path}")
    print(f"Audio saved to: {out_dir}")
    print(f"\nTo restore, run:")
    print(f"  python quick_infer.py restore \\")
    print(f"    --input {out_dir} \\")
    print(f"    --output {out_dir}/restored \\")
    print(f"    --acg_checkpoint {args.acg_checkpoint} \\")
    print(f"    --anonymizer_ckpt {args.anonymizer_ckpt} \\")
    print(f"    --key_file {keys_path}")


# ============================================================
# RESTORATION
# ============================================================

def restore_dir(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(device=device)
    model = _load_model(args, device)

    with open(args.key_file) as f:
        key_store: dict[str, list] = json.load(f)
    print(f"Loaded keys for {len(key_store)} speaker(s): {list(key_store.keys())}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ("*.wav", "*.flac", "*.mp3"):
        files += list(Path(args.input).rglob(ext))
    if not files:
        raise RuntimeError(f"No audio files found in: {args.input}")
    print(f"Found {len(files)} files to restore.")

    with torch.no_grad():
        for path in tqdm(files, desc="Restoring"):
            spk = path.parent.name

            # Handle speaker name mismatch gracefully
            if spk not in key_store:
                if len(key_store) == 1:
                    spk = list(key_store.keys())[0]
                else:
                    print(f"  [SKIP] No key for speaker '{spk}': {path}")
                    continue

            # FIX: unsqueeze(0) to get (1, embed_dim) — was (embed_dim,) which
            # caused a shape mismatch inside model.restore() → ACG → cINN.
            key = torch.tensor(
                key_store[spk], dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, embed_dim)

            # Prefer saved .mel.pt (exact anonymized mel, no vocoder round-trip)
            # Fall back to re-extracting mel from the anonymized audio file.
            mel_path = path.with_suffix(".mel.pt")
            if mel_path.exists():
                xa = torch.load(mel_path, map_location=device)
                if xa.dim() == 4:
                    xa = xa.squeeze(1)
            else:
                print(f"  [INFO] No .mel.pt for {path.name} — re-extracting mel from audio")
                wav_np, sr = sf.read(str(path))
                wav = torch.from_numpy(wav_np).float()
                if wav.ndim == 2:
                    wav = wav.mean(1)
                wav = processor.resample(wav, sr)
                xa = processor.wav_to_mel(wav).to(device)
                if xa.dim() == 4:
                    xa = xa.squeeze(1)
                # Clamp here too in case .mel.pt was missing and we re-extracted
                xa = xa.clamp(MEL_MIN, MEL_MAX)

            # Restore: key → ACG → same cond → cINN inverse (float64 internally)
            xr = model.restore(xa, key)

            out_path = out_dir / path.relative_to(args.input)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            restored = processor.mel_to_wav(xr.cpu())
            _save_audio(out_path, restored.squeeze(0).numpy(),
                        processor.sample_rate, args.format)

    print(f"\nRestoration complete. Output: {out_dir}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RANO quick inference — anonymize or restore speech"
    )
    sub = p.add_subparsers(dest="command")

    for cmd in ["anonymize", "restore"]:
        sp = sub.add_parser(cmd)
        sp.add_argument("--input",   required=True,  help="Input audio directory")
        sp.add_argument("--output",  required=True,  help="Output directory")
        sp.add_argument("--acg_checkpoint",   default="checkpoints/acg/acg_final.pt")
        sp.add_argument("--anonymizer_ckpt",  default="checkpoints/rano/anonymizer_best.pt")
        sp.add_argument("--embed_dim",        type=int, default=256)
        # FIX: was 8 — must match the num_cinn_blocks used during training (12)
        sp.add_argument("--num_cinn_blocks",  type=int, default=12)
        sp.add_argument("--format", choices=["wav", "flac", "mp3"], default="wav")
        if cmd == "restore":
            sp.add_argument("--key_file", required=True,
                            help="keys.json produced by the anonymize step")

    args = p.parse_args()

    if args.command == "anonymize":
        anonymize_dir(args)
    elif args.command == "restore":
        restore_dir(args)
    else:
        p.print_help()
