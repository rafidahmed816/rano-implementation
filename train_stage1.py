"""Stage 1: Pre-train ACG on real speaker embeddings (Sec. III-E, Fig. 3a)."""

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from acg import AnonymizationConditionGenerator
from audio import MelProcessor
from data import build_dataset
from speaker_encoder import SpeakerEncoder


def train_acg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()

    dataset = build_dataset(
        vctk_root=args.vctk_root,
        libritts_root=args.libritts_root,
        split="train",
        processor=processor,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True)

    asv = SpeakerEncoder(embed_dim=args.embed_dim).to(device)
    asv_ckpt = Path(args.asv_checkpoint)
    if asv_ckpt.exists():
        asv.load_state_dict(torch.load(asv_ckpt, map_location=device))
    asv.eval()

    acg = AnonymizationConditionGenerator(
        embed_dim=args.embed_dim, num_blocks=args.num_acg_blocks
    ).to(device)
    optimizer = Adam(acg.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in tqdm(range(1, args.iterations + 1), desc="ACG pre-training"):
        batch = next(iter(loader))
        mel = batch["mel"].to(device)

        with torch.no_grad():
            s = asv(mel.squeeze(1))

        loss = acg.loss(s, tau=args.tau)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(acg.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0:
            tqdm.write(f"step={step}  acg_loss={loss.item():.4f}")
            torch.save(acg.state_dict(), out_dir / f"acg_step{step}.pt")

    torch.save(acg.state_dict(), out_dir / "acg_final.pt")
    print(f"ACG saved to {out_dir / 'acg_final.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vctk_root", type=str, default=None)
    p.add_argument("--libritts_root", type=str, default=None)
    p.add_argument("--asv_checkpoint", type=str, default="checkpoints/asv.pt")
    p.add_argument("--output_dir", type=str, default="checkpoints/acg")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_acg_blocks", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_step", type=int, default=20_000)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    train_acg(args)
