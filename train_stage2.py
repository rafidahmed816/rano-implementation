"""Stage 2: Train Anonymizer cINN (§3, Algorithm 1).

Key improvements over original:
  - Checkpoint resume: saves/loads optimizer, scheduler, scaler, and step state
  - AMP enabled by default on CUDA (~1.5-2x faster on RTX 4060)
  - Fixed iteration count: 200,000 (was 20,000 due to typo)
  - Validation loss every N steps for training visibility
  - Log-det Jacobian regularization for better cINN invertibility
  - Saves best checkpoint by validation loss
"""

import argparse
import json
import time
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


def _save_training_state(
    path: Path,
    model: Rano,
    optimizer: Adam,
    scheduler: StepLR,
    scaler,
    step: int,
    best_val_loss: float,
):
    """Save full training state for resumption."""
    state = {
        "step": step,
        "best_val_loss": best_val_loss,
        "anonymizer_state_dict": model.anonymizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def _load_training_state(
    path: Path,
    model: Rano,
    optimizer: Adam,
    scheduler: StepLR,
    scaler,
    device: torch.device,
) -> tuple[int, float]:
    """Load training state. Returns (start_step, best_val_loss)."""
    state = torch.load(path, map_location=device)
    model.anonymizer.load_state_dict(state["anonymizer_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    return state["step"], state.get("best_val_loss", float("inf"))


@torch.no_grad()
def _validate(model: Rano, val_loader, device, use_amp, num_batches=10):
    """Run validation and return average losses."""
    model.anonymizer.eval()
    totals = {"total": 0.0, "consistency": 0.0, "triplet": 0.0}
    count = 0
    data_iter = iter(val_loader)
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        mel = batch["mel"].to(device, non_blocking=True)
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        # Temporarily enable grad for the validation forward pass
        # because training_step needs gradients internally for ASV→anonymizer flow.
        # We use a simpler validation: just compute consistency loss.
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            x = mel
            with torch.no_grad():
                s = model.asv(x)
            # Consistency check: cINN(x, s) should ≈ x
            x_hat, _ = model.anonymizer(x, s)
            val_cons = torch.nn.functional.mse_loss(x_hat, x)
            totals["consistency"] += val_cons.item()
            totals["total"] += val_cons.item()
        count += 1
    model.anonymizer.train()
    if count == 0:
        return {k: 0.0 for k in totals}
    return {k: v / count for k, v in totals.items()}


def train_rano(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()

    # --- Build train dataset ---
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
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    # --- Build validation dataset ---
    val_dataset = build_dataset(
        vctk_root=args.vctk_root,
        libritts_root=args.libritts_root,
        split="test",
        libritts_subsets=args.librispeech_subsets,
        processor=processor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    # --- Build model ---
    model = Rano(
        mel_channels=args.mel_channels,
        embed_dim=args.embed_dim,
        num_cinn_blocks=args.num_cinn_blocks,
        num_acg_blocks=args.num_acg_blocks,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        margin=args.margin,
        lambda_logdet=args.lambda_logdet,
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

    # Optional: compile anonymizer for faster GPU kernel fusion (PyTorch 2.0+)
    if args.compile and hasattr(torch, "compile"):
        print("Compiling anonymizer with torch.compile() -- first step will be slow...")
        model.anonymizer = torch.compile(model.anonymizer)

    # §3 Stage 2: Adam + StepLR(step_size=50000, gamma=0.5)
    optimizer = Adam(model.anonymizer.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    # Mixed precision: enabled by default on CUDA
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled.")

    writer = SummaryWriter(args.log_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint ---
    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            start_step, best_val_loss = _load_training_state(
                resume_path, model, optimizer, scheduler, scaler, device
            )
            print(f"  Resumed at step {start_step}, best_val_loss={best_val_loss:.6f}")
        else:
            print(f"[WARN] Resume checkpoint not found: {resume_path}, training from scratch.")
    elif (out_dir / "training_state.pt").exists():
        # Auto-resume from latest training state if it exists
        print(f"Auto-resuming from {out_dir / 'training_state.pt'}")
        start_step, best_val_loss = _load_training_state(
            out_dir / "training_state.pt", model, optimizer, scheduler, scaler, device
        )
        print(f"  Resumed at step {start_step}, best_val_loss={best_val_loss:.6f}")

    # --- Training loop ---
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    step_times = []

    print(f"\n{'='*60}")
    print(f"Training config:")
    print(f"  Device:           {device}")
    print(f"  AMP:              {use_amp}")
    print(f"  Batch size:       {args.batch_size} (physical) x {args.accumulate_steps} (accum) = {args.batch_size * args.accumulate_steps} (effective)")
    print(f"  Steps:            {start_step} -> {args.iterations}")
    print(f"  LR:               {args.lr} (StepLR every {args.lr_step})")
    print(f"  Lambda logdet:    {args.lambda_logdet}")
    print(f"  Checkpoint dir:   {out_dir}")
    print(f"{'='*60}\n")

    for step in tqdm(range(start_step + 1, args.iterations + 1), desc="Rano training", initial=start_step, total=args.iterations):
        step_start = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        mel = batch["mel"].to(device, non_blocking=True)
        # Ensure (B, 80, T) shape
        if mel.dim() == 4:
            mel = mel.squeeze(1)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            losses = model.training_step(mel, distance_threshold=args.distance_threshold)

        # Divide loss by accumulate_steps so the accumulated gradient matches the mean over the effective batch
        loss_to_backprop = losses["total"] / args.accumulate_steps
        if scaler is not None:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        if step % args.accumulate_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.anonymizer.parameters(), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        step_time = time.time() - step_start
        step_times.append(step_time)

        # --- Logging ---
        if step % 100 == 0:
            writer.add_scalar("loss/total", losses["total"].item(), step)
            writer.add_scalar("loss/consistency", losses["consistency"].item(), step)
            writer.add_scalar("loss/triplet", losses["triplet"].item(), step)
            if "logdet" in losses:
                writer.add_scalar("loss/logdet", losses["logdet"].item(), step)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], step)

            avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])
            remaining_steps = args.iterations - step
            eta_hours = (remaining_steps * avg_step_time) / 3600
            writer.add_scalar("perf/step_time", avg_step_time, step)

        if step % 1000 == 0:
            avg_step_time = sum(step_times[-1000:]) / len(step_times[-1000:])
            remaining_steps = args.iterations - step
            eta_hours = (remaining_steps * avg_step_time) / 3600
            logdet_str = f"  logdet={losses['logdet'].item():.4f}" if "logdet" in losses else ""
            tqdm.write(
                f"step={step}  total={losses['total'].item():.4f}"
                f"  cons={losses['consistency'].item():.4f}"
                f"  tri={losses['triplet'].item():.4f}"
                f"{logdet_str}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
                f"  step_t={avg_step_time:.2f}s"
                f"  ETA={eta_hours:.1f}h"
            )

        # --- Validation ---
        if step % args.val_every == 0:
            val_losses = _validate(model, val_loader, device, use_amp)
            writer.add_scalar("val/consistency", val_losses["consistency"], step)
            tqdm.write(
                f"  [VAL] step={step}  consistency={val_losses['consistency']:.6f}"
            )
            if val_losses["consistency"] < best_val_loss:
                best_val_loss = val_losses["consistency"]
                torch.save(model.anonymizer.state_dict(), out_dir / "anonymizer_best.pt")
                tqdm.write(f"  [VAL] New best! Saved anonymizer_best.pt (val_cons={best_val_loss:.6f})")

        # --- Checkpointing ---
        # Save every 1k steps for safety against power loss
        if step % 1000 == 0:
            torch.save(model.anonymizer.state_dict(), out_dir / f"anonymizer_step{step}.pt")
            _save_training_state(
                out_dir / "training_state.pt",
                model, optimizer, scheduler, scaler, step, best_val_loss,
            )
            tqdm.write(f"  [CKPT] Saved step {step} checkpoint + training state")

    # --- Final save ---
    torch.save(model.anonymizer.state_dict(), out_dir / "anonymizer_final.pt")
    _save_training_state(
        out_dir / "training_state.pt",
        model, optimizer, scheduler, scaler, args.iterations, best_val_loss,
    )

    # Also save a combined Rano checkpoint for convenience
    torch.save(model.state_dict(), out_dir / "rano_final.pt")
    print(f"Rano saved to {out_dir / 'rano_final.pt'}")
    print(f"Best validation consistency loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vctk_root", type=str, default=None)
    p.add_argument("--libritts_root", type=str, default=None)
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
    p.add_argument("--lambda_logdet", type=float, default=0.01,
                    help="Weight for log-det Jacobian regularization (0 to disable).")
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=4, help="Physical batch size per step.")
    p.add_argument(
        "--accumulate_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients. Effective batch size = batch_size * accumulate_steps.",
    )
    p.add_argument("--iterations", type=int, default=200_000)  # §7: 200k iterations
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_step", type=int, default=50_000)  # §7: step_size=50000
    p.add_argument("--distance_threshold", type=float, default=0.5)  # §7: d=0.5
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_every", type=int, default=500,
                    help="Run validation every N steps (also saves best model if improved).")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to training_state.pt to resume from. If not set, auto-resumes if training_state.pt exists in output_dir.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision (fp16) training. Enabled by default.",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision training.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Compile anonymizer with torch.compile() (PyTorch 2.0+). Slow first step, faster thereafter.",
    )
    args = p.parse_args()

    # Handle --no_amp override
    if args.no_amp:
        args.amp = False

    if args.librispeech_root and not args.libritts_root:
        args.libritts_root = args.librispeech_root
    train_rano(args)
