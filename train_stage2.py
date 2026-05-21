"""Stage 2: Train Anonymizer cINN (§3, Algorithm 1).

torch.compile deadlock fix (root cause analysis):
  - model.anonymizer is replaced with a compiled wrapper when --compile is used.
  - _validate() was calling model.anonymizer(mel, s) — i.e. the COMPILED object —
    in eval() mode. CUDA-graph-backed compiled modules try to recapture their graph
    when called in a different mode, which deadlocks because training CUDA kernels
    are still enqueued on the stream. This explains the exact step-499 hang
    (val_every=500 → step 500 triggers validation, tqdm shows 499).
  - Fix: keep _uncompiled_anonymizer (the original Anonymizer module) and use
    ONLY that inside _validate(). The compiled version is only used for training.
  - Fix: never call .eval()/.train() on the compiled module. The Anonymizer has
    no BatchNorm or Dropout, so eval/train mode makes zero numerical difference.
  - Fix: use torch.compile mode="default" (Inductor), NOT "reduce-overhead".
    "reduce-overhead" uses CUDA Graphs which require strict shape/mode invariance.
    "default" (Inductor) is more flexible and handles graph breaks gracefully.
  - Fix: autocast lives INSIDE model.training_step (set via model._amp_* attrs)
    so the compiled graph sees a clean boundary with no external context manager.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Rano
from audio import MelProcessor
from data import build_dataset


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

# Set VERBOSE = True to re-enable per-step diagnostic logging for debugging.
# When False, _log() is a zero-cost no-op — no string formatting, no I/O.
VERBOSE = False


def _log(msg: str, flush: bool = True) -> None:
    """Timestamped diagnostic print — only active when VERBOSE = True."""
    if not VERBOSE:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=flush, file=sys.stderr)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_training_state(
    path: Path,
    model: Rano,
    optimizer: Adam,
    scheduler: StepLR,
    scaler,
    step: int,
    best_val_loss: float,
) -> None:
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
    state = torch.load(path, map_location=device, weights_only=False)
    model.anonymizer.load_state_dict(state["anonymizer_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in state:
        scaler.load_state_dict(state["scaler_state_dict"])
    return state["step"], state.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Validation  — uses the UNCOMPILED anonymizer to avoid CUDA-graph deadlocks
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: Rano,
    uncompiled_anonymizer,          # original Anonymizer BEFORE torch.compile()
    val_loader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    step: int,
    num_batches: int = 10,
) -> dict[str, float]:
    """Run validation using the UNCOMPILED anonymizer module.

    KEY FIX: We deliberately pass the original (pre-compile) Anonymizer object.
    Calling the compiled wrapper in eval() mode triggers CUDA-graph recapture
    which deadlocks against in-flight training kernels on the same stream.
    The uncompiled anonymizer has identical weights (shared nn.Module state)
    and produces numerically identical output.

    We also do NOT call .eval()/.train() — the Anonymizer has no BatchNorm or
    Dropout, so mode switching has zero effect and only risks graph invalidation.
    """
    _log(f"[VAL] step={step} — entering validation (uncompiled anonymizer, no mode switch)")

    totals = {"consistency": 0.0}
    count = 0
    data_iter = iter(val_loader)

    for batch_idx in range(num_batches):
        _log(f"[VAL] step={step} batch {batch_idx+1}/{num_batches} — loading data")
        try:
            batch = next(data_iter)
        except StopIteration:
            _log(f"[VAL] step={step} — val_loader exhausted after {batch_idx} batches")
            break

        mel = batch["mel"].to(device, non_blocking=True)
        if mel.dim() == 4:
            mel = mel.squeeze(1)

        _log(f"[VAL] step={step} batch {batch_idx+1} — running ASV forward")
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            s = model.asv(mel)
            _log(f"[VAL] step={step} batch {batch_idx+1} — running uncompiled anonymizer forward")
            x_hat, _ = uncompiled_anonymizer(mel, s)
            val_cons = F.mse_loss(x_hat, mel)

        totals["consistency"] += val_cons.item()
        count += 1
        _log(f"[VAL] step={step} batch {batch_idx+1} — cons={val_cons.item():.6f}")

    _log(f"[VAL] step={step} — validation done ({count} batches)")
    if count == 0:
        return {"consistency": 0.0}
    return {k: v / count for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_rano(args) -> None:
    _log("train_rano() started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device = {device}")

    # --- CUDA performance flags ---
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 'high' enables TF32 on A100 matmuls (3× faster than FP32, same accuracy as BF16)
        torch.set_float32_matmul_precision('high')
        _log(f"CUDA device: {torch.cuda.get_device_name(0)}  "
             f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    processor = MelProcessor()
    _log("MelProcessor created")

    # --- Build datasets ---
    _log("Building training dataset...")
    dataset = build_dataset(
        vctk_root=args.vctk_root,
        libritts_root=args.libritts_root,
        split="train",
        libritts_subsets=args.librispeech_subsets,
        processor=processor,
        validate_librispeech=args.validate_dataset,
        fail_on_validation_error=not args.allow_invalid_dataset,
    )
    _log(f"Training dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        # prefetch_factor=4: keep 4 batches pre-loaded per worker so the GPU
        # never stalls waiting for CPU→GPU transfers. The A100 can process
        # batches fast enough that prefetch_factor=2 occasionally starves it.
        prefetch_factor=(4 if args.num_workers > 0 else None),
        drop_last=True,  # constant batch size B so CUDA-graph shapes never change
    )

    _log("Building validation dataset...")
    val_dataset = build_dataset(
        vctk_root=args.vctk_root,
        libritts_root=args.libritts_root,
        split="test",
        libritts_subsets=args.librispeech_subsets,
        processor=processor,
    )
    _log(f"Validation dataset: {len(val_dataset)} samples")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # FIX: val_loader only runs for 10 batches every 500 steps.
        # Spawning new worker processes (num_workers > 0) every 500 steps without
        # exhausting the iterator leaks shared memory (/dev/shm) and file descriptors.
        # At step 6000 (12th validation), /dev/shm fills up and the DataLoader deadlocks.
        # Setting num_workers=0 runs it safely in the main thread (plenty fast for 10 batches).
        num_workers=0,
        # FIX 2: PyTorch's pin_memory=True spawns a background PinMemoryThread even with num_workers=0.
        # Dropping the iterator early after 10 batches leaks these threads or deadlocks their queue.
        # We disable it for validation to make it 100% synchronous and safe to drop.
        pin_memory=False,
    )

    # --- Build model ---
    _log("Building Rano model...")
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
    _log("Rano model created and moved to device")

    # Load ACG checkpoint
    acg_ckpt = Path(args.acg_checkpoint)
    if not acg_ckpt.exists():
        raise FileNotFoundError(
            f"ACG checkpoint not found: {acg_ckpt}\n"
            "Train ACG first with: python train_stage1.py\n"
            "Or specify correct path: --acg_checkpoint <path>"
        )
    _log(f"Loading ACG checkpoint: {acg_ckpt}")
    model.acg.load_state_dict(torch.load(acg_ckpt, map_location=device, weights_only=True))
    for p in model.acg.parameters():
        p.requires_grad_(False)
    _log("ACG loaded and frozen")

    # Load ASV checkpoint
    asv_ckpt = Path(args.asv_checkpoint)
    if not asv_ckpt.exists():
        raise FileNotFoundError(
            f"ASV checkpoint not found: {asv_ckpt}\n"
            "Train ASV first with: python train_asv.py\n"
            "Or specify correct path: --asv_checkpoint <path>"
        )
    _log(f"Loading ASV checkpoint: {asv_ckpt}")
    model.asv.load_state_dict(torch.load(asv_ckpt, map_location=device, weights_only=True))
    for p in model.asv.parameters():
        p.requires_grad_(False)
    _log("ASV loaded and frozen")

    # --- Mixed precision ---
    use_amp = args.amp and device.type == "cuda"
    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
        _log("AMP: BF16 (no GradScaler needed)")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler()
        _log("AMP: FP16 + GradScaler")
    else:
        amp_dtype = torch.float32
        scaler = None
        _log("AMP: disabled (FP32)")

    # Pass AMP settings to model so training_step applies autocast internally
    model._amp_enabled = use_amp
    model._amp_dtype = amp_dtype
    model._device_type = device.type

    # --- CRITICAL: save uncompiled anonymizer reference BEFORE torch.compile ---
    # _validate() will use this reference exclusively.  Calling the compiled
    # module during validation (in eval mode, or with val-set shaped inputs that
    # differ from training shapes) triggers CUDA-graph recapture which deadlocks.
    _uncompiled_anonymizer = model.anonymizer
    _log(f"Uncompiled anonymizer reference saved: {type(_uncompiled_anonymizer).__name__}")

    # --- torch.compile (optional) ---
    if args.compile and hasattr(torch, "compile"):
        # ── Anonymizer ─────────────────────────────────────────────────────────
        # mode="reduce-overhead" → CUDA Graphs. Safe because _validate() uses
        # _uncompiled_anonymizer exclusively — the compiled wrapper is NEVER
        # called during validation, so there is no eval-mode graph recapture.
        # dynamic=False: drop_last=True guarantees constant batch size B.
        _log("Compiling anonymizer with torch.compile(mode='reduce-overhead', dynamic=False) ...")
        _log("  (first forward pass will be slow — 3-8 min for 12 RRDB blocks)")
        model.anonymizer = torch.compile(
            model.anonymizer,
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=False,
        )

        # ── ASV speaker encoder ─────────────────────────────────────────────────
        # Called TWICE per training step (once no_grad for embedding x, once
        # with grad for embedding xa for triplet loss). Compiling with Inductor
        # fuses Conv1d + BatchNorm1d + ReLU chains into efficient CUDA kernels.
        # Must use mode="default" (Inductor), NOT "reduce-overhead": BatchNorm1d
        # in train mode updates running_mean/running_var — a module state
        # side-effect that CUDA Graphs cannot safely handle.
        _log("Compiling ASV with torch.compile(mode='default') ...")
        model.asv = torch.compile(model.asv, mode="default", fullgraph=False)

        # ── ACG generate (inverse INN pass) ────────────────────────────────────
        # Called once per step with B*32 = 4096 samples through 8 INN blocks.
        # We compile the generate method directly (it is @torch.no_grad() at
        # class definition time, which torch.compile handles transparently).
        _log("Compiling ACG.generate with torch.compile(mode='default') ...")
        model.acg.generate = torch.compile(
            model.acg.generate, mode="default", fullgraph=False
        )

        # ── Loss function ───────────────────────────────────────────────────────
        # MSE + cosine_similarity + relu + log_det normalisation — trivially
        # fusible by Inductor. Small gain, completely free.
        _log("Compiling loss_fn with torch.compile(mode='default') ...")
        model.loss_fn = torch.compile(model.loss_fn, mode="default", fullgraph=False)

        _log("All torch.compile() registrations done (compilation is lazy — first forward)")
    else:
        _log("torch.compile: disabled")



    # --- Optimizer / scheduler ---
    _log("Creating Adam optimizer and StepLR scheduler")
    # fused=True: runs the Adam update as a single fused CUDA kernel instead of
    # one kernel per parameter tensor. For the anonymizer with ~12×4 subnets
    # worth of parameters, this avoids hundreds of small kernel launches per
    # optimizer step. Requires PyTorch ≥ 2.0 and CUDA device.
    _use_fused_adam = (device.type == "cuda")
    try:
        optimizer = Adam(
            model.anonymizer.parameters(),
            lr=args.lr, betas=(0.9, 0.99), eps=1e-8,
            fused=_use_fused_adam,
        )
        _log(f"Adam optimizer created (fused={_use_fused_adam})")
    except TypeError:
        # fused= not available in older PyTorch versions — fall back gracefully
        optimizer = Adam(model.anonymizer.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
        _log("Adam optimizer created (fused not available — PyTorch < 2.0?)")
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    writer = SummaryWriter(args.log_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume from checkpoint ---
    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            _log(f"Resuming from explicit checkpoint: {resume_path}")
            start_step, best_val_loss = _load_training_state(
                resume_path, model, optimizer, scheduler, scaler, device
            )
            _log(f"Resumed at step={start_step}  best_val_loss={best_val_loss:.6f}")
        else:
            _log(f"[WARN] Explicit resume checkpoint not found: {resume_path} — starting from scratch")
    elif (out_dir / "training_state.pt").exists():
        _log(f"Auto-resuming from {out_dir / 'training_state.pt'}")
        start_step, best_val_loss = _load_training_state(
            out_dir / "training_state.pt", model, optimizer, scheduler, scaler, device
        )
        _log(f"Resumed at step={start_step}  best_val_loss={best_val_loss:.6f}")

    # --- Training loop ---
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    step_times = []

    print(f"\n{'='*60}", flush=True)
    print(f"Training config:", flush=True)
    print(f"  Device:           {device}", flush=True)
    print(f"  AMP:              {use_amp} ({amp_dtype})", flush=True)
    print(f"  Batch size:       {args.batch_size} x {args.accumulate_steps} accum = {args.batch_size * args.accumulate_steps} effective", flush=True)
    print(f"  Steps:            {start_step} -> {args.iterations}", flush=True)
    print(f"  LR:               {args.lr} (StepLR step_size={args.lr_step})", flush=True)
    print(f"  lambda_logdet:    {args.lambda_logdet}", flush=True)
    print(f"  torch.compile:    {args.compile}", flush=True)
    print(f"  val_every:        {args.val_every}", flush=True)
    print(f"  Checkpoint dir:   {out_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    _log("Entering training loop")

    for step in tqdm(range(start_step + 1, args.iterations + 1), desc="Rano training", initial=start_step, total=args.iterations):
        step_start = time.time()

        # --- Data loading ---
        _log(f"[STEP {step}] fetching batch")
        try:
            batch = next(data_iter)
        except StopIteration:
            _log(f"[STEP {step}] DataLoader exhausted — resetting iterator")
            data_iter = iter(loader)
            batch = next(data_iter)

        mel = batch["mel"].to(device, non_blocking=True)
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        _log(f"[STEP {step}] batch loaded  shape={tuple(mel.shape)}  dtype={mel.dtype}")

        # --- Forward pass ---
        # autocast is applied INSIDE training_step via model._amp_* attributes
        # so the compiled graph sees no external context-manager boundary
        _log(f"[STEP {step}] calling training_step (forward pass)")
        losses = model.training_step(mel, distance_threshold=args.distance_threshold)
        _log(f"[STEP {step}] training_step done  total={losses['total'].item():.4f}")

        # --- NaN guard ---
        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            _log(f"[STEP {step}] NaN/Inf loss — skipping optimizer step")
            tqdm.write(f"  [WARN] step={step}  NaN/Inf loss — skipping")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        # --- Backward ---
        _log(f"[STEP {step}] backward pass")
        loss_to_backprop = losses["total"] / args.accumulate_steps
        if scaler is not None:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()
        _log(f"[STEP {step}] backward done")

        # --- Optimizer step ---
        if step % args.accumulate_steps == 0:
            _log(f"[STEP {step}] optimizer step")
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.anonymizer.parameters(), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            _log(f"[STEP {step}] optimizer step done  lr={scheduler.get_last_lr()[0]:.2e}")

        step_time = time.time() - step_start
        step_times.append(step_time)

        # --- TensorBoard logging (every 100 steps) ---
        if step % 100 == 0:
            _log(f"[STEP {step}] writing TensorBoard scalars")
            writer.add_scalar("loss/total", losses["total"].item(), step)
            writer.add_scalar("loss/consistency", losses["consistency"].item(), step)
            writer.add_scalar("loss/triplet", losses["triplet"].item(), step)
            if "logdet" in losses:
                writer.add_scalar("loss/logdet", losses["logdet"].item(), step)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
            avg_step_time = sum(step_times[-100:]) / len(step_times[-100:])
            writer.add_scalar("perf/step_time", avg_step_time, step)

        # --- Console logging (every 1000 steps) ---
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

        # --- Validation (uses UNCOMPILED anonymizer — no deadlock) ---
        if step % args.val_every == 0:
            _log(f"[STEP {step}] === VALIDATION START ===")
            # Explicit CUDA sync BEFORE validation so any in-flight training
            # kernels are fully retired before we touch the anonymizer.
            if device.type == "cuda":
                _log(f"[STEP {step}] cuda.synchronize() before validation")
                torch.cuda.synchronize()
                _log(f"[STEP {step}] cuda.synchronize() done")

            val_losses = _validate(
                model, _uncompiled_anonymizer, val_loader,
                device, use_amp, amp_dtype, step,
            )
            writer.add_scalar("val/consistency", val_losses["consistency"], step)
            tqdm.write(
                f"  [VAL] step={step}  consistency={val_losses['consistency']:.6f}"
            )
            if val_losses["consistency"] < best_val_loss:
                best_val_loss = val_losses["consistency"]
                torch.save(model.anonymizer.state_dict(), out_dir / "anonymizer_best.pt")
                tqdm.write(f"  [VAL] New best! Saved anonymizer_best.pt (val_cons={best_val_loss:.6f})")
            _log(f"[STEP {step}] === VALIDATION END ===")

        # --- Checkpointing (every 4000 steps) ---
        if step % 4000 == 0:
            _log(f"[STEP {step}] saving checkpoint")
            torch.save(model.anonymizer.state_dict(), out_dir / f"anonymizer_step{step}.pt")
            _save_training_state(
                out_dir / "training_state.pt",
                model, optimizer, scheduler, scaler, step, best_val_loss,
            )
            tqdm.write(f"  [CKPT] Saved step {step} checkpoint + training state")
            _log(f"[STEP {step}] checkpoint saved")

    # --- Final save ---
    _log("Training loop complete — saving final checkpoints")
    torch.save(model.anonymizer.state_dict(), out_dir / "anonymizer_final.pt")
    _save_training_state(
        out_dir / "training_state.pt",
        model, optimizer, scheduler, scaler, args.iterations, best_val_loss,
    )
    torch.save(model.state_dict(), out_dir / "rano_final.pt")
    _log(f"Saved rano_final.pt  best_val_loss={best_val_loss:.6f}")
    print(f"Rano saved to {out_dir / 'rano_final.pt'}", flush=True)
    print(f"Best validation consistency loss: {best_val_loss:.6f}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 2: Train Rano Anonymizer cINN")
    p.add_argument("--vctk_root", type=str, default=None)
    p.add_argument("--libritts_root", type=str, default=None)
    p.add_argument("--librispeech_root", type=str, default=None,
                   help="Alias for --libritts_root")
    p.add_argument("--librispeech_subsets", nargs="+", default=["train-clean-100"])
    p.add_argument("--validate_dataset", action="store_true")
    p.add_argument("--allow_invalid_dataset", action="store_true")
    p.add_argument("--acg_checkpoint", type=str, default="checkpoints/acg/acg_final.pt")
    p.add_argument("--asv_checkpoint", type=str, default="checkpoints/asv.pt")
    p.add_argument("--output_dir", type=str, default="checkpoints/rano")
    p.add_argument("--log_dir", type=str, default="logs/rano")
    p.add_argument("--mel_channels", type=int, default=80)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_cinn_blocks", type=int, default=12)
    p.add_argument("--num_acg_blocks", type=int, default=8)
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=5.0)
    p.add_argument("--lambda_logdet", type=float, default=0.01)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Physical batch size. A100 80GB fits bs=128 with AMP.")
    p.add_argument("--accumulate_steps", type=int, default=1,
                   help="Gradient accumulation steps. Effective bs = batch_size * accumulate_steps.")
    p.add_argument("--iterations", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_step", type=int, default=50_000)
    p.add_argument("--distance_threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_every", type=int, default=500,
                   help="Run validation every N steps.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to training_state.pt. Auto-resumes from output_dir if not set.")
    p.add_argument("--amp", action="store_true", default=True,
                   help="Enable AMP (default: True).")
    p.add_argument("--no_amp", action="store_true",
                   help="Disable AMP.")
    p.add_argument("--compile", action="store_true",
                   help="Compile anonymizer with torch.compile mode=default (Inductor). "
                        "First forward pass is slow (~3-8 min), all subsequent steps faster.")
    args = p.parse_args()

    if args.no_amp:
        args.amp = False
    if args.librispeech_root:
        args.libritts_root = args.librispeech_root
    if not args.libritts_root:
        raise ValueError("Either --libritts_root or --librispeech_root must be provided.")

    train_rano(args)