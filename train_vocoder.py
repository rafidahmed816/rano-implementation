"""Train a HiFi-GAN vocoder on THIS project's mel (via audio.MelProcessor), so it
inverts exactly the features the anonymizer produces. Any speech works as data
(VCTK + LibriTTS). Designed to fit an 8GB GPU (segment 8192, small batch, fp32).

FROM SCRATCH (slow, needs ~1M steps for clean speech):
  python train_vocoder.py --vctk_root VCTK-Corpus --libritts_root LibriTTS \
      --libritts_subset test-clean --output_dir checkpoints/vocoder \
      --batch_size 8 --steps 400000

FINE-TUNE a pretrained universal HiFi-GAN (fast, ~10-30k steps) -- RECOMMENDED.
Our Generator/MPD/MSD are architecturally identical to jik876's UNIVERSAL_V1, so
its weights load with strict=True; fine-tuning only re-maps the mel domain (our
torchaudio htk/power-2 mel vs their slaney/power-1 mel -- same audio underneath):
  python train_vocoder.py --vctk_root /mnt/storage/data/VCTK-Corpus-0.92 \
      --libritts_root /mnt/storage/data/LibriTTS --libritts_subset train-clean-100 \
      --output_dir /mnt/storage/checkpoints/vocoder_ft \
      --pretrained_g pretrained/UNIVERSAL_V1/g_02500000 \
      --pretrained_do pretrained/UNIVERSAL_V1/do_02500000 \
      --batch_size 24 --steps 30000

A local checkpoint in output_dir ALWAYS takes priority over --pretrained_* (so
resuming a fine-tune run continues from where it stopped; it does not reset to the
pretrained weights). Infer later with vocode_mel() (bottom).
"""
from __future__ import annotations
import argparse, itertools, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

from audio import MelProcessor
from hifigan import (Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
                     feature_loss, discriminator_loss, generator_loss)


class WavSegments(Dataset):
    """Random fixed-length wav segments from VCTK/LibriTTS, resampled to sr."""

    def __init__(self, roots, segment=8192, sr=22050):
        self.files = []
        for r in roots:
            if r:
                self.files += sorted(Path(r).rglob("*.wav")) + sorted(Path(r).rglob("*.flac"))
        if not self.files:
            raise SystemExit("No .wav/.flac found under the given roots.")
        self.segment = segment
        self.proc = MelProcessor(sample_rate=sr)
        random.seed(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                w, sr = sf.read(self.files[idx])
                w = torch.tensor(w).float()
                if w.ndim > 1:
                    w = w.mean(-1)
                w = self.proc.resample(w, sr)
                if w.shape[-1] < self.segment:
                    w = F.pad(w, (0, self.segment - w.shape[-1]))
                else:
                    s = random.randint(0, w.shape[-1] - self.segment)
                    w = w[s:s + self.segment]
                return w
            except Exception:
                idx = random.randint(0, len(self.files) - 1)
        raise RuntimeError("Repeated decode failures in WavSegments.")


def warm_start(G, mpd, msd, g_path, do_path, dev):
    """Load pretrained jik876 HiFi-GAN weights into our (architecturally identical)
    nets to fine-tune instead of training from scratch. G is required and loaded
    strict=True (any mismatch is a real error you must see); discriminators are
    optional and loaded leniently (fresh D still fine-tunes fine, it just re-learns)."""
    gsd = torch.load(g_path, map_location=dev, weights_only=False)
    gsd = gsd["generator"] if isinstance(gsd, dict) and "generator" in gsd else gsd
    G.load_state_dict(gsd, strict=True)   # architecture verified identical to UNIVERSAL_V1
    print(f"[pretrained] generator loaded from {g_path}  ({len(gsd)} tensors, strict=True OK)")
    if do_path:
        dsd = torch.load(do_path, map_location=dev, weights_only=False)
        try:
            mpd.load_state_dict(dsd["mpd"], strict=True)
            msd.load_state_dict(dsd["msd"], strict=True)
            print(f"[pretrained] MPD+MSD loaded from {do_path}  (warm discriminators)")
        except Exception as e:
            print(f"[pretrained][warn] discriminator load failed ({e}); using FRESH "
                  f"discriminators (fine-tune still works, D just re-learns).")
    else:
        print("[pretrained] no --pretrained_do -> discriminators start FRESH "
              "(ok; pass do_02500000 for a warmer, faster start).")


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = MelProcessor(sample_rate=args.sr)
    proc.mel_transform = proc.mel_transform.to(dev)  # mel on GPU (differentiable loss)

    roots = [args.vctk_root]
    if args.libritts_root:
        roots.append(str(Path(args.libritts_root) / args.libritts_subset)
                     if args.libritts_subset else args.libritts_root)
    ds = WavSegments(roots, args.segment, args.sr)
    print(f"Vocoder data: {len(ds)} files")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(args.num_workers > 0))

    G = Generator().to(dev)
    mpd = MultiPeriodDiscriminator().to(dev)
    msd = MultiScaleDiscriminator().to(dev)
    optG = torch.optim.AdamW(G.parameters(), args.lr, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(itertools.chain(mpd.parameters(), msd.parameters()),
                             args.lr, betas=(0.8, 0.99))
    schG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=0.999)
    schD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma=0.999)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    step = 0
    ckpt = out / "vocoder_state.pt"
    if ckpt.exists():
        # A run in progress ALWAYS wins over --pretrained_* (pretrained is already baked in).
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        G.load_state_dict(st["gen"]); mpd.load_state_dict(st["mpd"]); msd.load_state_dict(st["msd"])
        optG.load_state_dict(st["optG"]); optD.load_state_dict(st["optD"])
        schG.load_state_dict(st["schG"]); schD.load_state_dict(st["schD"])
        step = st["step"]
        print(f"Resumed vocoder from step {step}")
    elif args.pretrained_g:
        # First launch of a fine-tune run: warm-start weights, keep fresh optimizers at --lr.
        warm_start(G, mpd, msd, args.pretrained_g, args.pretrained_do, dev)
        print(f"Fine-tuning from pretrained HiFi-GAN at lr={args.lr}, starting at step 0.")

    G.train(); mpd.train(); msd.train()
    data_iter = iter(loader)
    t0 = time.time()
    while step < args.steps:
        try:
            y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); y = next(data_iter)
        y = y.to(dev, non_blocking=True)               # (B, segment)
        mel = proc.wav_to_mel(y)                        # (B, 80, F) — generator input
        y = y.unsqueeze(1)                              # (B, 1, segment)
        y_g = G(mel)                                    # (B, 1, ~F*256)
        L = min(y.shape[-1], y_g.shape[-1])
        y, y_g = y[..., :L], y_g[..., :L]

        # --- Discriminator ---
        optD.zero_grad(set_to_none=True)
        yr_p, yg_p, _, _ = mpd(y, y_g.detach())
        yr_s, yg_s, _, _ = msd(y, y_g.detach())
        loss_d = discriminator_loss(yr_p, yg_p) + discriminator_loss(yr_s, yg_s)
        loss_d.backward(); optD.step()

        # --- Generator ---
        optG.zero_grad(set_to_none=True)
        mel_g = proc.wav_to_mel(y_g.squeeze(1))
        Fm = min(mel.shape[-1], mel_g.shape[-1])
        loss_mel = F.l1_loss(mel[..., :Fm], mel_g[..., :Fm]) * 45.0
        yr_p, yg_p, fr_p, fg_p = mpd(y, y_g)
        yr_s, yg_s, fr_s, fg_s = msd(y, y_g)
        loss_fm = feature_loss(fr_p, fg_p) + feature_loss(fr_s, fg_s)
        loss_adv = generator_loss(yg_p) + generator_loss(yg_s)
        loss_g = loss_adv + loss_fm + loss_mel
        loss_g.backward(); optG.step()

        step += 1
        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"step={step}  mel={loss_mel.item()/45:.3f}  fm={loss_fm.item():.2f}  "
                  f"g_adv={loss_adv.item():.2f}  d={loss_d.item():.2f}  "
                  f"{sps:.1f} it/s  ETA={ (args.steps-step)/max(sps,1e-6)/3600:.1f}h", flush=True)
        if step % args.save_every == 0:
            schG.step(); schD.step()
            torch.save({"gen": G.state_dict(), "mpd": mpd.state_dict(), "msd": msd.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "schG": schG.state_dict(), "schD": schD.state_dict(), "step": step},
                       ckpt.with_suffix(".pt.tmp"))
            ckpt.with_suffix(".pt.tmp").replace(ckpt)
            torch.save({"gen": G.state_dict(), "step": step}, out / "generator.pt")
            print(f"  [ckpt] saved at step {step}", flush=True)

    print("Vocoder training complete.")


@torch.no_grad()
def vocode_mel(mel: torch.Tensor, generator_ckpt: str, device: str = "cuda") -> torch.Tensor:
    """Load a trained generator and vocode a log-mel (B,80,T) -> waveform (B, T_wav)."""
    G = Generator().to(device).eval()
    st = torch.load(generator_ckpt, map_location=device, weights_only=False)
    G.load_state_dict(st["gen"] if "gen" in st else st)
    G.remove_wn()
    return G(mel.to(device)).squeeze(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vctk_root", required=True)
    p.add_argument("--libritts_root", default=None)
    p.add_argument("--libritts_subset", default=None, help="e.g. test-clean or train-clean-100")
    p.add_argument("--pretrained_g", default=None,
                   help="Fine-tune: path to jik876 UNIVERSAL_V1 generator ckpt (g_02500000).")
    p.add_argument("--pretrained_do", default=None,
                   help="Fine-tune: optional discriminator ckpt (do_02500000) to warm-start MPD+MSD.")
    p.add_argument("--output_dir", default="checkpoints/vocoder")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--segment", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=8, help="drop to 4 if 8GB OOMs")
    p.add_argument("--steps", type=int, default=400000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2000)
    train(p.parse_args())
