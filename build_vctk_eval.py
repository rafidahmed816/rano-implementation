"""Build a VCTK evaluation set WITH transcripts (so WER works in evaluate3).

VCTK stores audio and text in parallel trees:
  <root>/wav48/pXXX/pXXX_YYY.wav
  <root>/txt/pXXX/pXXX_YYY.txt   (one sentence per file)

evaluate3.load_librispeech_transcripts() only understands LibriSpeech-style
'<spk>-<chap>.trans.txt' files ("uttid text" per line). So this copies N
speakers x K utterances of audio into <out>/pXXX/ and writes a matching
<out>/pXXX/pXXX.trans.txt, letting evaluate3 + --compute_wer work unchanged.

Usage:
  python build_vctk_eval.py --vctk_root VCTK-Corpus --out test_local \
      --num_speakers 15 --per_speaker 8
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vctk_root", required=True, help="VCTK-Corpus root (has wav48/ and txt/)")
    ap.add_argument("--out", default="test_local")
    ap.add_argument("--num_speakers", type=int, default=15)
    ap.add_argument("--per_speaker", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.vctk_root)
    wav_root = root / "wav48"
    txt_root = root / "txt"
    if not wav_root.is_dir():
        raise SystemExit(f"Expected {wav_root} (VCTK wav48/). Point --vctk_root at the VCTK-Corpus root.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spk_dirs = sorted([p for p in wav_root.iterdir() if p.is_dir()])
    total = 0
    for spk_dir in spk_dirs[: args.num_speakers]:
        spk = spk_dir.name
        wavs = sorted(spk_dir.glob("*.wav")) + sorted(spk_dir.glob("*.flac"))
        dst = out / spk
        lines = []
        copied = 0
        for w in wavs:
            if copied >= args.per_speaker:
                break
            txt = txt_root / spk / (w.stem + ".txt")   # parallel transcript
            if not txt.exists():
                continue
            text = txt.read_text(encoding="utf-8").strip()
            if not text:
                continue
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(w, dst / w.name)
            lines.append(f"{w.stem} {text}")            # LibriSpeech .trans.txt line
            copied += 1
            total += 1
        if lines:
            (dst / f"{spk}.trans.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_spk = sum(1 for d in out.iterdir() if d.is_dir())
    print(f"Built {out}: {n_spk} speakers, {total} utterances (with transcripts).")


if __name__ == "__main__":
    main()
