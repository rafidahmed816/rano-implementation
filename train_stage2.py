import argparse
from pathlib import Path
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Rano
from audio import MelProcessor
from data import build_dataset


def train_rano(args):
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

    model = Rano(
        mel_channels=args.mel_channels,
        embed_dim=args.embed_dim,
        num_cinn_blocks=args.num_cinn_blocks,
        num_acg_blocks=args.num_acg_blocks,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        margin=args.margin,
    ).to(device)

    # Load pre-trained ACG and freeze it (§3: freeze after Stage 1)
    acg_ckpt = Path(args.acg_checkpoint)
    if acg_ckpt.exists():
        model.acg.load_state_dict(torch.load(acg_ckpt, map_location=device))
    for p in model.acg.parameters():
        p.requires_grad_(False)

    # Load pre-trained ASV and freeze it (§8.1: never update ASV)
    asv_ckpt = Path(args.asv_checkpoint)
    if asv_ckpt.exists():
        model.asv.load_state_dict(torch.load(asv_ckpt, map_location=device))
    for p in model.asv.parameters():
        p.requires_grad_(False)

    # §3 Stage 2: Adam + StepLR(step_size=50000, gamma=0.5)
    optimizer = Adam(model.anonymizer.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    writer = SummaryWriter(args.log_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad()
    data_iter = iter(loader)
    for step in tqdm(range(1, args.iterations + 1), desc="Rano training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        mel = batch["mel"].to(device)
        # Ensure (B, 80, T) shape
        if mel.dim() == 4:
            mel = mel.squeeze(1)

        losses = model.training_step(mel, distance_threshold=args.distance_threshold)

        # Divide loss by accumulate_steps so the accumulated gradient matches the mean over the effective batch
        loss_to_backprop = losses["total"] / args.accumulate_steps
        loss_to_backprop.backward()

        if step % args.accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.anonymizer.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 100 == 0:
            writer.add_scalar("loss/total", losses["total"].item(), step)
            writer.add_scalar("loss/consistency", losses["consistency"].item(), step)
            writer.add_scalar("loss/triplet", losses["triplet"].item(), step)

        if step % 5000 == 0:
            tqdm.write(
                f"step={step}  total={losses['total'].item():.4f}"
                f"  cons={losses['consistency'].item():.4f}"
                f"  tri={losses['triplet'].item():.4f}"
            )

        # §3: save every 10,000 iterations
        if step % 10_000 == 0:
            torch.save(model.anonymizer.state_dict(), out_dir / f"anonymizer_step{step}.pt")

    torch.save(model.anonymizer.state_dict(), out_dir / "anonymizer_final.pt")
    torch.save(model.state_dict(), out_dir / "rano_final.pt")
    print(f"Rano saved to {out_dir / 'rano_final.pt'}")


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
    p.add_argument("--acg_checkpoint", type=str, default="checkpoints/acg/acg_final.pt")
    p.add_argument("--asv_checkpoint", type=str, default="checkpoints/asv.pt")
    p.add_argument("--output_dir", type=str, default="checkpoints/rano")
    p.add_argument("--log_dir", type=str, default="logs/rano")
    p.add_argument("--mel_channels", type=int, default=80)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_cinn_blocks", type=int, default=12)  # §2.2: 12 blocks
    p.add_argument("--num_acg_blocks", type=int, default=8)
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=5.0)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=4, help="Physical batch size per step.")
    p.add_argument(
        "--accumulate_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients. Effective batch size = batch_size * accumulate_steps.",
    )
    p.add_argument("--iterations", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_step", type=int, default=50_000)  # §7: step_size=50000
    p.add_argument("--distance_threshold", type=float, default=0.5)  # §7: d=0.5
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    if args.librispeech_root and not args.libritts_root:
        args.libritts_root = args.librispeech_root
    train_rano(args)
