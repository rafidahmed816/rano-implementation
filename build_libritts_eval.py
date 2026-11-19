"""Build a LibriTTS evaluation set WITH transcripts (so WER / check_intel work).

LibriTTS layout:
  <root>/<subset>/<speaker>/<chapter>/<spk>_<chap>_<utt>_<seg>.wav
  <root>/<subset>/<speaker>/<chapter>/<spk>_<chap>_<utt>_<seg>.normalized.txt

The transcript sits NEXT TO each wav (not a parallel tree like VCTK). This copies
N speakers x K utts into <out>/<speaker>/ and writes a LibriSpeech-style
<out>/<speaker>/<speaker>.trans.txt so evaluate3 / check_intel read it unchanged.

  python build_libritts_eval.py --libritts_root /mnt/storage/data/LibriTTS \
      --subset train-clean-100 --out test_multi_wer --num_speakers 20 --per_speaker 6
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libritts_root", required=True)
    ap.add_argument("--subset", default="train-clean-100")
    ap.add_argument("--out", default="test_multi_wer")
    ap.add_argument("--num_speakers", type=int, default=20)
    ap.add_argument("--per_speaker", type=int, default=6)
    args = ap.parse_args()

    src = Path(args.libritts_root) / args.subset
    if not src.is_dir():
        raise SystemExit(f"{src} not found. Check --libritts_root / --subset.")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    spk_dirs = sorted([p for p in src.iterdir() if p.is_dir()])
    total = 0
    for spk_dir in spk_dirs[: args.num_speakers]:
        spk = spk_dir.name
        wavs = sorted(spk_dir.rglob("*.wav"))[: args.per_speaker]
        dst = out / spk
        lines = []
        for w in wavs:
            norm = w.parent / (w.stem + ".normalized.txt")   # transcript next to wav
            if not norm.exists():
                continue
            text = norm.read_text(encoding="utf-8").strip()
            if not text:
                continue
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(w, dst / w.name)
            lines.append(f"{w.stem} {text}")
            total += 1
        if lines:
            (dst / f"{spk}.trans.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_spk = sum(1 for d in out.iterdir() if d.is_dir())
    print(f"Built {out}: {n_spk} speakers, {total} utterances (with transcripts).")


if __name__ == "__main__":
    main()
