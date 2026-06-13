"""Debug mel <-> wav conversion and vocoder behavior.

Saves reconstructions (Griffin-Lim and `mel_to_wav`) and prints mel statistics.

Usage examples (PowerShell):
  python scripts/debug_mel_vocoder.py --file test_audio/121726/121-121726-0000.flac --out_dir debug_out --use_grifflim
  python scripts/debug_mel_vocoder.py --file vocoderinference/cond_swap/baseline/121-121726-0000_anon.pt --out_dir debug_out
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from audio import MelProcessor


def to_mono(wav: np.ndarray) -> np.ndarray:
    w = np.asarray(wav, dtype=np.float32)
    if w.ndim == 2:
        return w.mean(axis=1)
    return w


def load_mel_from_audio(path: Path, processor: MelProcessor) -> torch.Tensor:
    wav_np, sr = sf.read(str(path))
    wav_np = to_mono(wav_np)
    wav_t = processor.resample(torch.tensor(wav_np).float(), sr)
    mel = processor.wav_to_mel(wav_t)  # (B, 80, T) or (80, T)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    return mel


def load_mel_from_file(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, np.ndarray):
        t = torch.from_numpy(obj)
    else:
        t = obj
    if not torch.is_tensor(t):
        raise RuntimeError(f"Unsupported mel file type: {type(obj)}")
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t


def mel_stats(m: torch.Tensor) -> dict:
    arr = m.cpu().numpy()
    return {
        "shape": arr.shape,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "finite": bool(np.all(np.isfinite(arr))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Input audio (.wav/.flac) or mel (.pt/.npy)")
    p.add_argument("--out_dir", default="vocoderinference/debug", help="Output folder")
    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--use_grifflim", action="store_true", help="Force Griffin-Lim (skip heavy vocoder)" )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = MelProcessor(device=None, use_hifigan=not args.use_grifflim, sample_rate=args.sample_rate)

    pth = Path(args.file)
    if pth.suffix.lower() in (".wav", ".flac"):
        mel = load_mel_from_audio(pth, processor)
    elif pth.suffix.lower() in (".pt", ".pth"):
        mel = load_mel_from_file(pth)
    elif pth.suffix.lower() == ".npy":
        arr = np.load(pth)
        mel = torch.from_numpy(arr)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
    else:
        raise RuntimeError(f"Unsupported input type: {pth.suffix}")

    stats = mel_stats(mel)
    print("Mel stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    stem = pth.stem

    # Save mel as numpy for inspection
    np.save(out_dir / f"{stem}_mel.npy", mel.cpu().numpy())

    # Griffin-Lim reconstruction (always available)
    try:
        wav_gl = processor.mel_to_wav_grifflim(mel.cpu())
        wav_gl = wav_gl.squeeze().numpy()
        sf.write(str(out_dir / f"{stem}_grifflim.wav"), wav_gl, args.sample_rate)
        print(f"Saved Griffin-Lim recon -> {out_dir / (stem + '_grifflim.wav')}")
    except Exception as exc:
        print(f"Griffin-Lim recon failed: {exc}")

    # Try mel_to_wav (may use HiFi-GAN / transformers if available)
    try:
        wav_voc = processor.mel_to_wav(mel.cpu())
        wav_voc = wav_voc.squeeze().numpy()
        sf.write(str(out_dir / f"{stem}_mel_to_wav.wav"), wav_voc, args.sample_rate)
        print(f"Saved mel_to_wav recon -> {out_dir / (stem + '_mel_to_wav.wav')}")
    except Exception as exc:
        print(f"mel_to_wav (neural vocoder) failed or unavailable: {exc}")

    # Quick sanity checks on reconstructions
    try:
        import numpy as _np
        for name in [f"{stem}_grifflim.wav", f"{stem}_mel_to_wav.wav"]:
            p = out_dir / name
            if not p.exists():
                continue
            w, sr = sf.read(p)
            w = to_mono(w)
            print(f"{name}: len={len(w)} samples ({len(w)/args.sample_rate:.2f}s), mean={float(_np.mean(w)):.6f}, std={float(_np.std(w)):.6f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
