"""Vocoder-free anonymization check — the truth serum for a Rano checkpoint.

Measures how far the cINN moves speaker identity in the training-ASV embedding
space, with NO vocoder, phase, or pitch-shift confounds. Point it at any
anonymizer checkpoint during/after training.

  cos(ASV(xa), ASV(x)) = 1.0  -> passthrough (NO anonymization)
  cos(ASV(xa), ASV(x)) -> 0   -> strong identity change

Usage:
  python check_anon.py --anonymizer_ckpt checkpoints/rano_warmstart/anonymizer_best.pt \
      --test_dir test_multi --n 60
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

from model import Rano
from audio import MelProcessor


def _load(path, module, device):
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if any(k.startswith("anonymizer.") for k in sd):
        sd = {k[len("anonymizer."):]: v for k, v in sd.items() if k.startswith("anonymizer.")}
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    inc = module.load_state_dict(sd, strict=False)
    if inc.missing_keys:
        raise RuntimeError(f"{path}: {len(inc.missing_keys)} missing keys (e.g. {inc.missing_keys[:3]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anonymizer_ckpt", required=True)
    ap.add_argument("--acg_checkpoint", default="checkpoints/acg/acg_final.pt")
    ap.add_argument("--asv_checkpoint", default="checkpoints/asv.pt")
    ap.add_argument("--test_dir", default="test_multi")
    ap.add_argument("--n", type=int, default=60, help="max utterances")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = MelProcessor()
    m = Rano(embed_dim=256, num_cinn_blocks=12).to(dev).eval()
    _load(args.acg_checkpoint, m.acg, dev)
    _load(args.asv_checkpoint, m.asv, dev)
    _load(args.anonymizer_ckpt, m.anonymizer, dev)

    files = sorted(Path(args.test_dir).rglob("*.flac"))[: args.n]
    if not files:
        raise SystemExit(f"No .flac under {args.test_dir}")

    cos_xa_x, cos_xa_c = [], []
    with torch.no_grad():
        for f in files:
            w, sr = sf.read(f)
            w = proc.resample(torch.tensor(w).float(), sr)
            mel = proc.wav_to_mel(w).to(dev)
            key = torch.randn(1, 256, device=dev)
            xa, c = m.anonymize(mel, key)
            ex, exa = m.asv(mel), m.asv(xa)
            cos_xa_x.append(F.cosine_similarity(exa, ex).item())
            cos_xa_c.append(F.cosine_similarity(exa, c).item())

    x_x = float(np.mean(cos_xa_x))
    print(f"\n=== Vocoder-free anonymization ({len(files)} utts, {args.anonymizer_ckpt}) ===")
    print(f"  cos(ASV(xa), ASV(x))   = {x_x:.3f}   (1.0=passthrough, 0=strong change)")
    print(f"  cos(ASV(xa), target c) = {np.mean(cos_xa_c):.3f}   (1.0=fully moved to target)")
    print(f"  => identity moved {(1 - x_x) * 100:.1f}% off the original")
    if x_x > 0.9:
        print("  VERDICT: still a PASSTHROUGH — keep training / raise margin,lambda2.")
    elif x_x > 0.5:
        print("  VERDICT: partial anonymization — progressing, keep going.")
    else:
        print("  VERDICT: strong anonymization — the cINN is doing the work.")


if __name__ == "__main__":
    main()
