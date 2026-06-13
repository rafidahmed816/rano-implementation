"""
Quick test: measure vocoder baseline WER (no anonymization).

This measures the WER introduced purely by the mel round-trip:
  wav → mel-spectrogram → vocoder → wav' → Whisper ASR → WER

Purpose: Establishes the "floor" WER that any anonymization system
using this vocoder cannot go below. The RANO paper's ground-truth
WER is 6.58% (Table I), so your vocoder baseline should be similar.

Usage:
  python test_vocoder_wer.py --test_dir data/test-clean --vocoder hifigan
  python test_vocoder_wer.py --test_dir data/test-clean --vocoder griffinlim
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from audio import MelProcessor

# Mel clamping for HiFi-GAN (matches quick_infer.py)
MEL_MIN = -11.5
MEL_MAX = 2.0


def load_transcripts(test_dir: str) -> dict:
    """Load ground truth transcripts from LibriSpeech .trans.txt files."""
    transcripts = {}
    for f in Path(test_dir).rglob("*.trans.txt"):
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1].lower()
    return transcripts


def calculate_wer(ref: str, hyp: str) -> float:
    """Standard WER via dynamic programming."""
    r, h = ref.lower().split(), hyp.lower().split()
    if len(r) == 0:
        return 0.0
    dp = np.zeros((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return float(dp[-1][-1] / max(1, len(r)))


def peak_normalize(wav: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(wav))
    if max_val > 1e-8:
        return wav / max_val
    return wav


def main():
    p = argparse.ArgumentParser(
        description="Test vocoder baseline WER (no anonymization)"
    )
    p.add_argument("--test_dir", required=True, help="LibriSpeech test directory")
    p.add_argument(
        "--vocoder",
        choices=["hifigan", "griffinlim"],
        default="hifigan",
        help="Vocoder to test. hifigan = paper §IV-A (default).",
    )
    p.add_argument(
        "--max_samples", type=int, default=50, help="Max utterances to test"
    )
    p.add_argument(
        "--whisper_model", default="large", help="Whisper model size"
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Vocoder: {args.vocoder}")

    use_hifigan = args.vocoder == "hifigan"
    processor = MelProcessor(
        device=device, use_hifigan=use_hifigan, sample_rate=22050
    )
    proc_sr = 22050

    # Load Whisper
    import whisper

    whisper_model = whisper.load_model(args.whisper_model).to(device)
    print(f"Whisper ({args.whisper_model}) loaded.")

    transcripts = load_transcripts(args.test_dir)
    print(f"Loaded {len(transcripts)} transcripts.")

    files = list(Path(args.test_dir).rglob("*.flac")) + list(
        Path(args.test_dir).rglob("*.wav")
    )
    files = files[: args.max_samples]
    print(f"Testing {len(files)} utterances.\n")

    def transcribe(wav: np.ndarray, sr: int) -> str:
        wav = np.asarray(wav, dtype=np.float32)
        if sr != 16000:
            wav = torchaudio.functional.resample(
                torch.tensor(wav), sr, 16000
            ).numpy()
        wav = peak_normalize(wav)
        audio = whisper.pad_or_trim(wav)
        n_mels = whisper_model.dims.n_mels
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(device)
        opts = whisper.DecodingOptions(
            language="en", without_timestamps=True, fp16=False
        )
        return whisper.decode(whisper_model, mel, opts).text

    wer_original = []
    wer_roundtrip = []
    per_utterance = []

    start = time.time()
    with torch.no_grad():
        for path in tqdm(files, desc="Testing vocoder WER"):
            ref_text = transcripts.get(path.stem, "")
            if not ref_text:
                continue

            wav_np, sr = sf.read(str(path))
            if wav_np.ndim == 2:
                wav_np = wav_np.mean(axis=1)

            # Resample to processor sample rate
            wav_proc = torchaudio.functional.resample(
                torch.tensor(wav_np, dtype=torch.float32), sr, proc_sr
            ).numpy()

            if len(wav_proc) < 1024:
                continue

            # 1. Transcribe original audio (no processing)
            hyp_orig = transcribe(wav_np, sr)
            wer_o = calculate_wer(ref_text, hyp_orig) * 100

            # 2. Round-trip: wav → mel → vocoder → wav'
            wav_t = torch.tensor(wav_proc, dtype=torch.float32)
            mel = processor.wav_to_mel(wav_t).to(device)

            if use_hifigan:
                mel_clamped = mel.clamp(MEL_MIN, MEL_MAX)
                recon_wav = processor.mel_to_wav(mel_clamped.cpu())
            else:
                recon_wav = processor.mel_to_wav(mel.cpu())

            recon_np = peak_normalize(recon_wav.squeeze(0).numpy())

            # 3. Transcribe reconstructed audio
            hyp_recon = transcribe(recon_np, proc_sr)
            wer_r = calculate_wer(ref_text, hyp_recon) * 100

            wer_original.append(wer_o)
            wer_roundtrip.append(wer_r)

            per_utterance.append(
                {
                    "file": path.stem,
                    "ref": ref_text[:60],
                    "wer_orig": wer_o,
                    "wer_roundtrip": wer_r,
                }
            )

    elapsed = time.time() - start

    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" Vocoder Baseline WER Test — {args.vocoder}")
    print(f"{'='*65}")
    print(f"  Utterances tested:    {len(wer_original)}")
    print(f"  Time:                 {elapsed:.1f}s")
    print()
    print(f"  Original audio WER:   {np.mean(wer_original):>6.1f}%  (no processing)")
    print(f"  Vocoder round-trip:   {np.mean(wer_roundtrip):>6.1f}%  (wav->mel->vocoder->wav')")
    print(f"  WER from vocoder:     {np.mean(wer_roundtrip) - np.mean(wer_original):>+6.1f}%")
    print()
    print(f"  Paper ground-truth:     6.58%  (Table I)")
    print(f"  Paper RANO WER:        11.91%  (Table I)")
    print(f"{'='*65}")

    # Show worst utterances
    per_utterance.sort(key=lambda x: x["wer_roundtrip"], reverse=True)
    print(f"\n  Top-5 WORST round-trip WER:")
    for u in per_utterance[:5]:
        print(
            f"    {u['file']}: orig={u['wer_orig']:5.1f}%  "
            f"roundtrip={u['wer_roundtrip']:5.1f}%  "
            f"ref: {u['ref']}..."
        )

    # Show best utterances
    per_utterance.sort(key=lambda x: x["wer_roundtrip"])
    print(f"\n  Top-5 BEST round-trip WER:")
    for u in per_utterance[:5]:
        print(
            f"    {u['file']}: orig={u['wer_orig']:5.1f}%  "
            f"roundtrip={u['wer_roundtrip']:5.1f}%  "
            f"ref: {u['ref']}..."
        )

    # Summary verdict
    print()
    if np.mean(wer_roundtrip) < 10:
        print("  ✓ Vocoder quality is GOOD — baseline WER is close to paper.")
        print("    → Any remaining WER gap is from the cINN, not the vocoder.")
    elif np.mean(wer_roundtrip) < 15:
        print("  ~ Vocoder quality is FAIR — some WER overhead from vocoder.")
        print("    → Consider fine-tuning HiFi-GAN on your mel configuration.")
    else:
        print("  ✗ Vocoder quality is POOR — significant WER from vocoder alone.")
        print("    → Check mel config matches HiFi-GAN training config.")
        print("    → Try SpeechBrain tts-hifigan-ljspeech if using SpeechT5.")


if __name__ == "__main__":
    main()
