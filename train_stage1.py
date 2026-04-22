"""Stage 1: Pre-train ACG on real speaker embeddings (§3, Stage 1)."""

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from acg import AnonymizationConditionGenerator
from audio import MelProcessor
from data import build_dataset
from speaker_encoder import AdaINVCSpeakerEncoder


def train_acg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()

    dataset = build_dataset(
        vctk_root=args.vctk_root,
        libritts_root=args.libritts_root,
        split="train",
        libritts_subsets=args.librispeech_subsets,
        processor=processor,
        validate_librispeech=args.validate_dataset,
        fail_on_validation_error=not args.allow_invalid_dataset,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    asv = AdaINVCSpeakerEncoder(embed_dim=args.embed_dim).to(device)
    asv_ckpt = Path(args.asv_checkpoint)
    if asv_ckpt.exists():
        asv.load_state_dict(torch.load(asv_ckpt, map_location=device))
    asv.eval()
    for p in asv.parameters():
        p.requires_grad_(False)

    acg = AnonymizationConditionGenerator(
        embed_dim=args.embed_dim, num_blocks=args.num_acg_blocks
    ).to(device)
    # §3 Stage 1: Adam with weight decay τ=0.5 for L2 regularisation, NO scheduler
    optimizer = Adam(acg.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8,
                     weight_decay=args.tau)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    data_iter = iter(loader)
    for step in tqdm(range(1, args.iterations + 1), desc="ACG pre-training"):
        # Proper iterator cycling (fix: was re-creating iterator every step)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        mel = batch["mel"].to(device)

        with torch.no_grad():
            s = asv(mel.squeeze(1) if mel.dim() == 4 else mel)

        loss = acg.loss(s, tau=args.tau)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(acg.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0:
            tqdm.write(f"step={step}  acg_loss={loss.item():.4f}")

        # §3: save every 10,000 iterations; keep best
        if step % 10_000 == 0:
            torch.save(acg.state_dict(), out_dir / f"acg_step{step}.pt")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(acg.state_dict(), out_dir / "acg_best.pt")

    torch.save(acg.state_dict(), out_dir / "acg_final.pt")
    print(f"ACG saved to {out_dir / 'acg_final.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vctk_root", type=str, default=None)
    p.add_argument("--libritts_root", type=str, default="data")
    p.add_argument(
        "--librispeech_root",
        type=str,
        default=None,
        help="Alias for --libritts_root; supports LibriSpeech-style layout.",
    )
    p.add_argument(
        "--librispeech_subsets",
        nargs="+",
        default=["train-clean-100"],
        help="Subsets under root. If root already points to subset folder, keep default.",
    )
    p.add_argument(
        "--validate_dataset",
        action="store_true",
        help="Validate LibriSpeech transcript/audio alignment before training.",
    )
    p.add_argument(
        "--allow_invalid_dataset",
        action="store_true",
        help="Continue even when validation finds issues.",
    )
    p.add_argument("--asv_checkpoint", type=str, default="checkpoints/asv.pt")
    p.add_argument("--output_dir", type=str, default="checkpoints/acg")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_acg_blocks", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)       # §7: batch=64
    p.add_argument("--iterations", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    if args.librispeech_root and not args.libritts_root:
        args.libritts_root = args.librispeech_root
    train_acg(args)
