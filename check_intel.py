"""Intelligibility check — is the ANONYMIZED mel actually speech, or garbled?

Compares WER of the original mel vs the anonymized mel, both vocoded with the
SAME Griffin-Lim (consistent phase). This isolates the MODEL from the vocoder:

  origMel WER low  + anonMel WER low   -> content preserved (GOOD, fix worked)
  origMel WER low  + anonMel WER high  -> the model garbles content (BAD)

Needs a transcript-bearing test set (e.g. built by build_vctk_eval.py).

  python check_intel.py --anonymizer_ckpt <ckpt> --test_dir test_local --n 8
"""
from __future__ import annotations
import argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf

from model import Rano
from audio import MelProcessor
from evaluate3 import pseudo_inverse_vocoder, calculate_wer, load_librispeech_transcripts


def _load(p, mod, dev):
    sd = torch.load(p, map_location=dev, weights_only=False)
    sd = sd.get("state_dict", sd)
    mod.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anonymizer_ckpt", required=True)
    ap.add_argument("--acg_checkpoint", default="checkpoints/acg/acg_final.pt")
    ap.add_argument("--test_dir", default="test_local")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--whisper_model", default="medium")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = MelProcessor()
    m = Rano(embed_dim=256, num_cinn_blocks=12).to(dev).eval()
    _load(args.acg_checkpoint, m.acg, dev)
    _load(args.anonymizer_ckpt, m.anonymizer, dev)

    import whisper
    wm = whisper.load_model(args.whisper_model).to(dev)
    tr = load_librispeech_transcripts(args.test_dir)

    def wer(wav, ref):
        try:
            w = torchaudio.functional.resample(torch.tensor(wav).float(), 22050, 16000).numpy()
            w = w / (np.abs(w).max() + 1e-8)
            txt = wm.transcribe(w.astype("float32"), language="en", temperature=0.0)["text"]
            return calculate_wer(ref, txt) * 100
        except Exception:
            return float("nan")

    og, an = [], []
    for f in sorted(Path(args.test_dir).rglob("*.wav"))[: args.n]:
        ref = tr.get(f.stem)
        if not ref:
            continue
        w, sr = sf.read(f); w22 = proc.resample(torch.tensor(w).float(), sr)
        mel = proc.wav_to_mel(w22).to(dev)
        xa, _ = m.anonymize(mel, torch.randn(1, 256, device=dev))
        og.append(wer(pseudo_inverse_vocoder(mel, orig_wav=w22.numpy(), sr=22050, mode="griffinlim"), ref))
        an.append(wer(pseudo_inverse_vocoder(xa, orig_wav=w22.numpy(), sr=22050, mode="griffinlim"), ref))

    print(f"\n=== Intelligibility ({args.anonymizer_ckpt}) ===")
    print(f"  origMel WER = {np.nanmean(og):5.1f}%   (vocoder baseline; should be low)")
    print(f"  anonMel WER = {np.nanmean(an):5.1f}%   (the model; want CLOSE to origMel)")
    gap = np.nanmean(an) - np.nanmean(og)
    print(f"  gap = {gap:5.1f}pp  -> " +
          ("content preserved ✓" if gap < 40 else "still garbling — tune lambda_anchor/range"))


if __name__ == "__main__":
    main()
