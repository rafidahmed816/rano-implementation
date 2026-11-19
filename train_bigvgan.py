"""Train the BigVGAN-style vocoder (SnakeBeta generator + MPD + MRD) on THIS
project's mel (audio.MelProcessor). Meant for the cloud GPU (bigger batch).

  python train_bigvgan.py \
      --vctk_root /mnt/storage/data/VCTK-Corpus-0.92 \
      --libritts_root /mnt/storage/data/LibriTTS --libritts_subset train-clean-100 \
      --output_dir /mnt/storage/checkpoints/bigvgan --batch_size 16 --steps 400000

Auto-resumes from output_dir. Infer with vocode_mel() (bottom).
"""
from __future__ import annotations
import argparse, itertools, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audio import MelProcessor
from train_vocoder import WavSegments  # reuse the wav-segment dataset
from hifigan import MultiPeriodDiscriminator, feature_loss, discriminator_loss, generator_loss
from bigvgan import BigVGAN, MultiResolutionDiscriminator


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = MelProcessor(sample_rate=args.sr)
    proc.mel_transform = proc.mel_transform.to(dev)

    roots = [args.vctk_root]
    if args.libritts_root:
        roots.append(str(Path(args.libritts_root) / args.libritts_subset)
                     if args.libritts_subset else args.libritts_root)
    ds = WavSegments(roots, args.segment, args.sr)
    print(f"BigVGAN data: {len(ds)} files")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True,
                        persistent_workers=(args.num_workers > 0))

    G = BigVGAN().to(dev)
    mpd = MultiPeriodDiscriminator().to(dev)
    mrd = MultiResolutionDiscriminator().to(dev)
    optG = torch.optim.AdamW(G.parameters(), args.lr, betas=(0.8, 0.99))
    optD = torch.optim.AdamW(itertools.chain(mpd.parameters(), mrd.parameters()),
                             args.lr, betas=(0.8, 0.99))
    schG = torch.optim.lr_scheduler.ExponentialLR(optG, gamma=0.999)
    schD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma=0.999)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    step = 0
    ckpt = out / "bigvgan_state.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        G.load_state_dict(st["gen"]); mpd.load_state_dict(st["mpd"]); mrd.load_state_dict(st["mrd"])
        optG.load_state_dict(st["optG"]); optD.load_state_dict(st["optD"])
        schG.load_state_dict(st["schG"]); schD.load_state_dict(st["schD"])
        step = st["step"]
        print(f"Resumed BigVGAN from step {step}")

    G.train(); mpd.train(); mrd.train()
    data_iter = iter(loader)
    t0 = time.time()
    while step < args.steps:
        try:
            y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); y = next(data_iter)
        y = y.to(dev, non_blocking=True)
        mel = proc.wav_to_mel(y)
        y = y.unsqueeze(1)
        y_g = G(mel)
        L = min(y.shape[-1], y_g.shape[-1])
        y, y_g = y[..., :L], y_g[..., :L]

        # Discriminator
        optD.zero_grad(set_to_none=True)
        yr_p, yg_p, _, _ = mpd(y, y_g.detach())
        yr_r, yg_r, _, _ = mrd(y, y_g.detach())
        loss_d = discriminator_loss(yr_p, yg_p) + discriminator_loss(yr_r, yg_r)
        loss_d.backward(); optD.step()

        # Generator
        optG.zero_grad(set_to_none=True)
        mel_g = proc.wav_to_mel(y_g.squeeze(1))
        Fm = min(mel.shape[-1], mel_g.shape[-1])
        loss_mel = F.l1_loss(mel[..., :Fm], mel_g[..., :Fm]) * 45.0
        yr_p, yg_p, fr_p, fg_p = mpd(y, y_g)
        yr_r, yg_r, fr_r, fg_r = mrd(y, y_g)
        loss_fm = feature_loss(fr_p, fg_p) + feature_loss(fr_r, fg_r)
        loss_adv = generator_loss(yg_p) + generator_loss(yg_r)
        (loss_adv + loss_fm + loss_mel).backward(); optG.step()

        step += 1
        if step % args.log_every == 0:
            sps = args.log_every / (time.time() - t0); t0 = time.time()
            print(f"step={step}  mel={loss_mel.item()/45:.3f}  fm={loss_fm.item():.2f}  "
                  f"g_adv={loss_adv.item():.2f}  d={loss_d.item():.2f}  {sps:.1f} it/s  "
                  f"ETA={(args.steps-step)/max(sps,1e-6)/3600:.1f}h", flush=True)
        if step % args.save_every == 0:
            schG.step(); schD.step()
            torch.save({"gen": G.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                        "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "schG": schG.state_dict(), "schD": schD.state_dict(), "step": step},
                       ckpt.with_suffix(".pt.tmp"))
            ckpt.with_suffix(".pt.tmp").replace(ckpt)
            torch.save({"gen": G.state_dict(), "step": step}, out / "generator.pt")
            print(f"  [ckpt] saved at step {step}", flush=True)

    print("BigVGAN training complete.")


@torch.no_grad()
def vocode_mel(mel: torch.Tensor, generator_ckpt: str, device: str = "cuda") -> torch.Tensor:
    """Load a trained BigVGAN generator and vocode a log-mel (B,80,T) -> (B, T_wav)."""
    G = BigVGAN().to(device).eval()
    st = torch.load(generator_ckpt, map_location=device, weights_only=False)
    G.load_state_dict(st["gen"] if "gen" in st else st)
    G.remove_wn()
    return G(mel.to(device)).squeeze(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vctk_root", required=True)
    p.add_argument("--libritts_root", default=None)
    p.add_argument("--libritts_subset", default=None)
    p.add_argument("--output_dir", default="/mnt/storage/checkpoints/bigvgan")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--segment", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=16, help="48GB fits ~16-24")
    p.add_argument("--steps", type=int, default=400000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=2000)
    train(p.parse_args())
