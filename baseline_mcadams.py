"""VoicePrivacy Challenge baseline B2: McAdams-coefficient anonymization.

A signal-processing anonymizer (no training, no GPU): LPC-analyse each frame, take
the poles, raise the phase angle of each complex pole pair to the power `mcadams`,
resynthesise. Shifting formant positions changes perceived speaker identity while
leaving the content largely intact.

This is the standard literature baseline to compare a learned anonymizer against.
It writes the SAME <spk>/<utt>_orig.wav / _anon.wav layout evaluate3.py produces, so
wer_from_audio.py and the held-out attacker scripts run on the output unchanged.

  python baseline_mcadams.py --test_dir test_wer_set --out_dir eval_mcadams
  python wer_from_audio.py --audio_dir eval_mcadams --test_dir test_wer_set --whisper_model large
"""
from __future__ import annotations
import argparse, glob, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import scipy.signal
import soundfile as sf


def mcadams_anonymize(x: np.ndarray, sr: int, winlen_ms: float = 20.0,
                      shift_ms: float = 10.0, lp_order: int = 20,
                      mcadams: float = 0.8) -> np.ndarray:
    """VPC B2. x: mono float waveform -> anonymized waveform (same length-ish)."""
    eps = np.finfo(np.float32).eps
    x = np.asarray(x, dtype=np.float64) + eps
    winlen = int(np.floor(winlen_ms * 0.001 * sr))
    shift = int(np.floor(shift_ms * 0.001 * sr))
    if len(x) < winlen * 2:
        return x.astype(np.float32)

    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)
    n_frames = 1 + int(np.floor((len(x) - winlen) / shift))
    out = np.zeros(len(x))

    lpc_fn = getattr(librosa, "lpc", None) or librosa.core.lpc
    for m in range(1, n_frames):
        frame = x[m * shift: m * shift + winlen] * win
        if len(frame) < winlen:
            break
        try:
            a_lpc = lpc_fn(frame + eps, order=lp_order)
        except Exception:
            continue
        poles = scipy.signal.tf2zpk(np.array([1.0]), a_lpc)[1]
        idx_imag = np.where(~np.isreal(poles))[0]
        if idx_imag.size == 0:
            continue
        # one index per conjugate pair
        idx_pair = idx_imag[np.arange(0, idx_imag.size, 2)]

        new_ang = np.angle(poles[idx_pair]) ** mcadams          # THE McAdams transform
        new_ang = np.clip(new_ang, 0.0, np.pi)

        new_poles = poles.copy()
        for k, i in enumerate(idx_pair):
            r = np.abs(poles[i])
            new_poles[i] = r * np.exp(1j * new_ang[k])
            if i + 1 < len(new_poles):                          # its conjugate
                new_poles[i + 1] = np.abs(poles[i + 1]) * np.exp(-1j * new_ang[k])

        a_new = np.real(np.poly(new_poles))
        res = scipy.signal.lfilter(a_lpc, np.array([1.0]), frame)      # LPC residual
        rec = scipy.signal.lfilter(np.array([1.0]), a_new, res) * win  # resynthesise
        sl = np.arange(m * shift, m * shift + len(rec))
        out[sl] += rec

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.95
    return out.astype(np.float32)


def main(args):
    files = sorted(glob.glob(os.path.join(args.test_dir, "*", "*.flac"))) + \
            sorted(glob.glob(os.path.join(args.test_dir, "*", "*.wav")))
    if not files:
        raise SystemExit(f"No audio under {args.test_dir}")
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    print(f"McAdams (alpha={args.mcadams}) on {len(files)} utterances -> {out_root}")

    for i, f in enumerate(files, 1):
        p = Path(f)
        parts = p.stem.split("-")
        spk = parts[0] if len(parts) >= 2 and parts[0].isdigit() else p.parent.name
        w, sr = sf.read(f)
        if np.asarray(w).ndim > 1:
            w = np.mean(w, axis=1)
        if sr != args.sr:
            w = librosa.resample(np.asarray(w, dtype=np.float64), orig_sr=sr, target_sr=args.sr)
            sr = args.sr
        d = out_root / spk; d.mkdir(parents=True, exist_ok=True)
        peak = np.max(np.abs(w)) or 1.0
        sf.write(str(d / f"{p.stem}_orig.wav"), (w / peak * 0.95).astype(np.float32), sr)
        sf.write(str(d / f"{p.stem}_anon.wav"), mcadams_anonymize(w, sr, mcadams=args.mcadams), sr)
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    print("done. Score it with wer_from_audio.py and the held-out attacker scripts.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True)
    p.add_argument("--out_dir", default="eval_mcadams")
    p.add_argument("--sr", type=int, default=22050, help="match the system under comparison")
    p.add_argument("--mcadams", type=float, default=0.8, help="VPC B2 default 0.8")
    main(p.parse_args())
