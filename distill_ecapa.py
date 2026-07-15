"""Distill SpeechBrain ECAPA-TDNN (waveform, 192-d) into the mel-domain
AdaINVCSpeakerEncoder (256-d), so Stage B/C can train the anonymizer against an
ECAPA-ALIGNED target WITHOUT a differentiable vocoder in the loop.

Why: rano_v2 fooled its OWN homegrown ASV (train-space cos ~ -0.09) but an
independent ECAPA still linked anon->orig at 25% EER (generalization gap). The
GL control proved mel<->identity is tightly aligned (cos 0.875), so a mel student
CAN mimic ECAPA's cosine geometry. Train Stage C against this student and fooling
it should transfer to fooling ECAPA at eval.

Output `asv_ecapa.pt` is a DROP-IN --asv_checkpoint: it is exactly an
AdaINVCSpeakerEncoder state_dict (the distill head is NOT saved), so Stage B and
Stage C load it strict, unchanged.

  python distill_ecapa.py \
    --vctk_root /mnt/storage/data/VCTK-Corpus-0.92 \
    --libritts_root /mnt/storage/data/LibriTTS --libritts_subset train-clean-100 \
    --output checkpoints/asv_ecapa.pt --steps 30000 --batch_size 64 --num_workers 8

Watch ECAPA-align climb (~0.7-0.9) and l_rel fall. Auto-resumes from output_dir.
"""
from __future__ import annotations
import argparse, sys, time, types
from pathlib import Path

# speechbrain eagerly loads its optional k2-fsa (ASR) integration during model
# init; k2 isn't installed and we don't use it. Stub it so `import k2` succeeds
# and the ECAPA teacher load doesn't crash. Harmless if a real k2 is present
# (setdefault keeps it), so this is safe on the cloud too.
sys.modules.setdefault("k2", types.ModuleType("k2"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from audio import MelProcessor
from speaker_encoder import AdaINVCSpeakerEncoder
from evaluate3 import load_asv          # SAME ECAPA loader the eval uses
from train_vocoder import WavSegments   # reuse the tested wav-segment dataset

# On Python 3.14, torch custom-op registration probes sys.modules via
# inspect.getmodule -> hasattr(mod, "__file__"). speechbrain's LazyModule raises
# ImportError there for optional integrations that aren't installed (k2, wordemb,
# ...), crashing the ECAPA load. Convert that to AttributeError so hasattr()
# returns False cleanly. No-op where speechbrain already loads fine (e.g. cloud).
try:
    from speechbrain.utils import importutils as _sb_iu
    _sb_orig_getattr = _sb_iu.LazyModule.__getattr__
    def _sb_safe_getattr(self, name):
        try:
            return _sb_orig_getattr(self, name)
        except ImportError:
            raise AttributeError(name)
    _sb_iu.LazyModule.__getattr__ = _sb_safe_getattr
except Exception:
    pass


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = MelProcessor(sample_rate=args.sr)
    proc.mel_transform = proc.mel_transform.to(dev)

    roots = [args.vctk_root]
    if args.libritts_root:
        roots.append(str(Path(args.libritts_root) / args.libritts_subset)
                     if args.libritts_subset else args.libritts_root)
    ds = WavSegments(roots, args.segment, args.sr)
    print(f"Distill data: {len(ds)} files | segment {args.segment} (~{args.segment/args.sr:.1f}s)",
          flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(args.num_workers > 0))

    student = AdaINVCSpeakerEncoder(mel_channels=80, embed_dim=args.embed_dim).to(dev)
    head = nn.Linear(args.embed_dim, 192).to(dev)     # readout to ECAPA dim; distill-only, NOT saved
    print("Loading ECAPA-TDNN teacher (frozen) ...", flush=True)
    extract_ecapa = load_asv(torch.device(dev))       # wav(B,T) -> (B,192) numpy, no_grad

    opt = torch.optim.AdamW(list(student.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.parent / (out.stem + ".state.pt")
    tmp_path = out.parent / (out.stem + ".state.tmp")
    step = 0
    if state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        student.load_state_dict(st["student"]); head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); step = st["step"]
        print(f"Resumed distillation from step {step}", flush=True)

    student.train()
    data_iter = iter(loader); t0 = time.time()
    while step < args.steps:
        try:
            wav = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); wav = next(data_iter)
        wav = wav.to(dev, non_blocking=True)                  # (B, segment) @ sr
        with torch.no_grad():
            mel = proc.wav_to_mel(wav)                        # (B, 80, T)
            wav16 = torchaudio.functional.resample(wav, args.sr, 16000)
            e = extract_ecapa(wav16)                          # (B, 192) numpy, frozen
            e = F.normalize(torch.from_numpy(e).float().to(dev), dim=-1)

        h = student(mel)                                      # (B, 256), L2-norm
        p = F.normalize(head(h), dim=-1)                      # (B, 192)
        l_cos = (1.0 - (p * e).sum(-1)).mean()                # match ECAPA direction
        l_rel = F.mse_loss(h @ h.t(), e @ e.t())              # match cosine GEOMETRY (triplet uses this)
        loss = l_cos + args.lambda_rel * l_rel
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()

        step += 1
        if step % args.log_every == 0:
            align = (p * e).sum(-1).mean().item()
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"step={step:6d}  l_cos={l_cos.item():.4f}  l_rel={l_rel.item():.4f}  "
                  f"ECAPA-align={align:.3f}  {sps:.1f} it/s  "
                  f"ETA={(args.steps-step)/max(sps,1e-6)/3600:.2f}h", flush=True)
        if step % args.save_every == 0:
            torch.save(student.state_dict(), out)             # drop-in AdaINVCSpeakerEncoder
            torch.save({"student": student.state_dict(), "head": head.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, tmp_path)
            tmp_path.replace(state_path)
            print(f"  [ckpt] {out} @ step {step}", flush=True)

    torch.save(student.state_dict(), out)
    print(f"\nDone. Distilled ECAPA-aligned ASV -> {out}", flush=True)
    print("Next: retrain the ACG (Stage B) on THIS asv, then a FRESH Stage C.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vctk_root", required=True)
    p.add_argument("--libritts_root", default=None)
    p.add_argument("--libritts_subset", default=None, help="e.g. train-clean-100")
    p.add_argument("--output", default="checkpoints/asv_ecapa.pt")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--segment", type=int, default=64000, help="~2.9s; ECAPA needs a few seconds")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_rel", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2000)
    train(p.parse_args())
