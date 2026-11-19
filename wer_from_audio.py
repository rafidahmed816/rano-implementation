"""Compute WER with Whisper LARGE from an already-generated eval audio dir.

Why separate: evaluate3.py loads Whisper in fp32 (~6 GB for large) ON TOP of Rano +
ECAPA + the vocoder -- that does not fit an 8 GB card. Running it as a second pass
gives Whisper the whole GPU, and loading it in fp16 halves it to ~3 GB, so LARGE fits
comfortably. Uses the exact same normalizer + edit-distance WER as evaluate3.

Reports BOTH:
  orig WER  -- Whisper on the clean originals = the ASR floor on this data
  anon WER  -- Whisper on the anonymized audio = the number you report
(the gap is what the anonymizer+vocoder actually cost)

  python wer_from_audio.py --audio_dir eval_neural_final --test_dir test_wer_set
  python wer_from_audio.py --audio_dir eval_neural_final --test_dir test_wer_set --whisper_model medium
"""
from __future__ import annotations
import argparse, glob, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torchaudio

from evaluate3 import calculate_wer, load_librispeech_transcripts, to_mono, peak_normalize


def transcribe_fp16(model, wav: np.ndarray, sr: int, half: bool) -> str:
    import whisper
    wav = np.asarray(wav, dtype=np.float32)
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000).numpy()
    wav = peak_normalize(wav)
    audio = whisper.pad_or_trim(wav)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    if half:
        mel = mel.half()
    opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=half)
    return whisper.decode(model, mel, opts).text


def main(args):
    import whisper
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    transcripts = load_librispeech_transcripts(args.test_dir)
    if not transcripts:
        raise SystemExit(f"No .trans.txt found under {args.test_dir} -- WER needs transcripts.")
    print(f"Loaded {len(transcripts)} ground-truth transcripts.")

    print(f"Loading Whisper {args.whisper_model} on {dev} (fp16={dev=='cuda'}) ...", flush=True)
    model = whisper.load_model(args.whisper_model, device=dev)
    half = (dev == "cuda") and not args.fp32
    if half:
        # 1.55B params fp16 ~3 GB (vs ~6 GB fp32) -- large then fits an 8 GB card.
        # Whisper's own LayerNorm does `super().forward(x.float())`, so if its weights
        # are fp16 you get "expected scalar type Float but found Half". Keep every
        # LayerNorm in fp32 and half the rest; Whisper's Linear/Conv1d already cast
        # their weights to the input dtype, so the mixed graph is consistent.
        model = model.half()
        for m in model.modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.float()
    if dev == "cuda":
        print(f"  VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    files = sorted(glob.glob(os.path.join(args.audio_dir, "*", "*_anon.wav")))
    if not files:
        raise SystemExit(f"No *_anon.wav under {args.audio_dir}")
    print(f"Scoring {len(files)} utterances from {args.audio_dir}\n", flush=True)

    wer_anon, wer_orig, missing = [], [], 0
    for i, af in enumerate(files, 1):
        stem = Path(af).name.replace("_anon.wav", "")
        ref = transcripts.get(stem)
        if not ref:
            missing += 1
            continue
        wa, sr = sf.read(af)
        wer_anon.append(calculate_wer(ref, transcribe_fp16(model, to_mono(wa), sr, half)) * 100.0)
        of = af.replace("_anon.wav", "_orig.wav")
        if os.path.exists(of):
            wo, sro = sf.read(of)
            wer_orig.append(calculate_wer(ref, transcribe_fp16(model, to_mono(wo), sro, half)) * 100.0)
        if i % 20 == 0:
            print(f"  {i}/{len(files)} ...", flush=True)

    print("\n" + "=" * 62)
    print(f"WER -- Whisper {args.whisper_model}   [{args.audio_dir}]")
    print("=" * 62)
    if wer_orig:
        print(f"  orig WER (clean originals, ASR floor) : {np.mean(wer_orig):6.2f}%   [{len(wer_orig)} utts]")
    print(f"  anon WER (anonymized, REPORT THIS)    : {np.mean(wer_anon):6.2f}%   [{len(wer_anon)} utts]")
    if wer_orig:
        print(f"  cost of anonymization + vocoder       : {np.mean(wer_anon)-np.mean(wer_orig):6.2f} pts")
    if missing:
        print(f"  [warn] {missing} files had no matching transcript and were skipped")
    print("=" * 62)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True, help="eval output dir containing <spk>/<utt>_anon.wav")
    p.add_argument("--test_dir", required=True, help="test set containing .trans.txt files")
    p.add_argument("--whisper_model", default="large")
    p.add_argument("--fp32", action="store_true",
                   help="Force fp32 Whisper (~2x VRAM). Only if fp16 output looks wrong.")
    main(p.parse_args())
