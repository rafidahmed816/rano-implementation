"""Assemble a multi-speaker evaluation set for EER measurement.

EER needs MANY distinct real speakers. This copies a balanced subset
(N speakers x K utterances each) from a LibriSpeech root into a flat
<out>/<speaker>/ layout, plus the matching .trans.txt files so --compute_wer
still works. Speakers/utterances are picked deterministically (sorted) for
reproducibility.

Usage:
  python build_eval_set.py --src data/LibriSpeech/train-clean-100 \
      --out test_multi --num_speakers 20 --per_speaker 6
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="LibriSpeech subset root (speaker/chapter/*.flac)")
    ap.add_argument("--out", default="test_multi")
    ap.add_argument("--num_speakers", type=int, default=20)
    ap.add_argument("--per_speaker", type=int, default=6)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spk_dirs = sorted([p for p in src.iterdir() if p.is_dir()])
    if not spk_dirs:
        raise SystemExit(f"No speaker folders under {src}")

    chosen_speakers = spk_dirs[: args.num_speakers]
    total = 0
    trans_copied: set[str] = set()

    for spk_dir in chosen_speakers:
        flacs = sorted(spk_dir.rglob("*.flac"))[: args.per_speaker]
        if not flacs:
            continue
        dst_dir = out / spk_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in flacs:
            shutil.copy2(f, dst_dir / f.name)
            # copy the chapter transcript once (for optional --compute_wer)
            for trans in f.parent.glob("*.trans.txt"):
                key = str(trans)
                if key not in trans_copied:
                    shutil.copy2(trans, dst_dir / trans.name)
                    trans_copied.add(key)
            total += 1

    n_spk = sum(1 for d in out.iterdir() if d.is_dir())
    print(f"Built {out}: {n_spk} speakers, {total} utterances, "
          f"{len(trans_copied)} transcript files.")


if __name__ == "__main__":
    main()
