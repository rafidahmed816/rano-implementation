"""
WER Decomposition Test (Normalized Scoring)

Three measurements, all using Whisper's official EnglishTextNormalizer
so punctuation/number/contraction mismatches don't inflate WER:

  TEST 1: Raw audio baseline
          wav → Whisper → WER
          (Whisper's inherent error rate on this audiobook text)

  TEST 2: Vocoder-only roundtrip (no cINN)
          wav → mel → vocoder (Griffin-Lim / Perturbed-Phase) → wav2 → Whisper → WER

  TEST 3: (reference only — paste your existing 44.3% anonymized number)
          wav → mel → cINN → vocoder → wav2 → Whisper → WER

Report deltas:
  Vocoder cost = TEST2 - TEST1
  cINN cost    = TEST3 - TEST2
"""

import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from audio import MelProcessor
from evaluate3 import (
    pseudo_inverse_vocoder, load_librispeech_transcripts,
    to_mono, peak_normalize, resample_to
)

# ============================================================
# NORMALIZED WER (fixes punctuation/number/contraction mismatches)
# ============================================================
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

def calculate_wer(ref: str, hyp: str) -> float:
    ref_n = normalizer(ref)
    hyp_n = normalizer(hyp)
    r, h = ref_n.split(), hyp_n.split()
    if len(r) == 0:
        return 0.0
    dp = np.zeros((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1): dp[i][0] = i
    for j in range(len(h) + 1): dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return float(dp[-1][-1] / max(1, len(r)))


# ============================================================
# SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

proc_sr = 16000  # match main eval for apples-to-apples comparison
processor = MelProcessor(device=device, use_hifigan=False, sample_rate=proc_sr)

import whisper
print("Loading Whisper large-v2...")
whisper_model = whisper.load_model("large-v2").to(device)

def transcribe(wav, sr):
    wav = np.asarray(wav, dtype=np.float32)
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000).numpy()
    wav = peak_normalize(wav)
    audio = whisper.pad_or_trim(wav)
    n_mels = whisper_model.dims.n_mels
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(device)
    opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
    return whisper.decode(whisper_model, mel, opts).text


# Load ground truth transcripts
test_dir = "test_audio"
transcripts = load_librispeech_transcripts(test_dir)
print(f"Loaded {len(transcripts)} transcripts")

files = list(Path(test_dir).rglob("*.flac")) + list(Path(test_dir).rglob("*.wav"))
print(f"Found {len(files)} files\n")


# ============================================================
# TEST 1: RAW AUDIO BASELINE
# wav → Whisper → WER (no mel, no vocoder, no cINN)
# ============================================================
print("=" * 60)
print("TEST 1: RAW AUDIO BASELINE")
print("  wav → Whisper → WER")
print("=" * 60)

raw_wers = []
for path in files:
    ref_text = transcripts.get(path.stem, "")
    if not ref_text:
        continue

    wav_np, sr = sf.read(str(path))
    wav_np = to_mono(wav_np)
    wav_16k = resample_to(wav_np, sr, 16000)
    if len(wav_16k) < 1024:
        continue

    hyp = transcribe(wav_16k, 16000)
    w = calculate_wer(ref_text, hyp) * 100
    raw_wers.append(w)
    print(f"  {path.stem}: raw_WER={w:.1f}%  ref: {ref_text[:50]}...")

raw_baseline = np.mean(raw_wers)
print(f"\n  >>> Raw audio WER (Whisper baseline): {raw_baseline:.1f}%")


# ============================================================
# TEST 2: VOCODER-ONLY ROUNDTRIP (NO cINN)
# wav → mel → vocoder → wav2 → Whisper → WER
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: VOCODER-ONLY ROUNDTRIP (no cINN, no anonymization)")
print("  wav → mel → pseudo_inverse_vocoder → wav2 → Whisper → WER")
print("=" * 60)

vocoder_wers_gl = []
vocoder_wers_pp = []

for path in files:
    ref_text = transcripts.get(path.stem, "")
    if not ref_text:
        continue

    wav_np, sr = sf.read(str(path))
    wav_np = to_mono(wav_np)
    wav_16k = resample_to(wav_np, sr, proc_sr)
    if len(wav_16k) < 1024:
        continue

    wav_t = torch.tensor(wav_16k).float()
    mel = processor.wav_to_mel(wav_t).to(device)

    # Griffin-Lim roundtrip
    roundtrip_gl = pseudo_inverse_vocoder(
        mel, orig_wav=wav_16k, sr=proc_sr,
        n_fft=1024, hop_length=256, n_mels=80,
        mode="griffinlim", griffinlim_iters=128, apply_post_filter=False,
    )
    roundtrip_gl = peak_normalize(roundtrip_gl)
    hyp_gl = transcribe(roundtrip_gl, proc_sr)
    w_gl = calculate_wer(ref_text, hyp_gl) * 100
    vocoder_wers_gl.append(w_gl)

    # Perturbed-phase roundtrip
    roundtrip_pp = pseudo_inverse_vocoder(
        mel, orig_wav=wav_16k, sr=proc_sr,
        n_fft=1024, hop_length=256, n_mels=80,
        mode="perturbed_phase", pitch_shift_semitones=3.5,
        griffinlim_iters=128, apply_post_filter=False,
    )
    roundtrip_pp = peak_normalize(roundtrip_pp)
    hyp_pp = transcribe(roundtrip_pp, proc_sr)
    w_pp = calculate_wer(ref_text, hyp_pp) * 100
    vocoder_wers_pp.append(w_pp)

    print(f"  {path.stem}: GL_WER={w_gl:.1f}%  PP_WER={w_pp:.1f}%  ref: {ref_text[:50]}...")

gl_mean = np.mean(vocoder_wers_gl)
pp_mean = np.mean(vocoder_wers_pp)


# ============================================================
# SUMMARY / DECOMPOSITION
# ============================================================
ANON_WER = 44.3  # paste your existing full-pipeline (vocoder + cINN) result here

print("\n" + "=" * 60)
print("DECOMPOSITION SUMMARY (normalized WER)")
print("=" * 60)
print(f"  TEST 1 — Raw audio baseline:           {raw_baseline:.1f}%")
print(f"  TEST 2 — Vocoder-only (Griffin-Lim):   {gl_mean:.1f}%")
print(f"  TEST 2 — Vocoder-only (Perturbed-Ph):  {pp_mean:.1f}%")
print(f"  TEST 3 — Full pipeline (cINN+vocoder): {ANON_WER:.1f}%  (from previous run)")
print("-" * 60)
print(f"  Vocoder cost   (Test2 - Test1):  GL={gl_mean - raw_baseline:+.1f}pp   PP={pp_mean - raw_baseline:+.1f}pp")
print(f"  cINN cost      (Test3 - Test2):  GL={ANON_WER - gl_mean:+.1f}pp   PP={ANON_WER - pp_mean:+.1f}pp")
print("=" * 60)