"""Train AdaIN-VC Speaker Encoder (ASV) on LibriSpeech speaker classification.

This produces the pre-trained ASV checkpoint required for Stage 2 (Anonymizer)
training.  The encoder is trained as a speaker classifier using cross-entropy
loss, then the classification head is discarded and only the encoder weights
are saved as ``checkpoints/asv.pt``.

Why is this needed?
  The ASV provides meaningful speaker-identity embeddings that the triplet loss
  (Eq. 6) relies on.  Without a trained ASV, the anonymizer cannot learn to
  actually change speaker identity — it would just be optimising against random
  projections.

Usage:
  python train_asv.py
      --librispeech_root D:/thesis/data/LibriSpeech
      --output checkpoints/asv.pt
      --epochs 50 --batch_size 32 --amp
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from speaker_encoder import AdaINVCSpeakerEncoder
from audio import MelProcessor
from data import build_dataset


# ---------------------------------------------------------------------------
# Thin wrapper: encoder + linear classifier
# ---------------------------------------------------------------------------

class ASVClassifier(nn.Module):
    """Speaker encoder + classification head (discarded after training)."""

    def __init__(self, mel_channels: int = 80, embed_dim: int = 256,
                 num_speakers: int = 251):
        super().__init__()
        self.encoder = AdaINVCSpeakerEncoder(mel_channels, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_speakers)

    def forward(self, mel: torch.Tensor):
        emb = self.encoder(mel)                # (B, embed_dim) L2-normalised
        logits = self.classifier(emb)          # (B, num_speakers)
        return logits, emb


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(model, val_loader, device, use_amp, speaker_id_remap=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in val_loader:
        mel = batch["mel"].to(device, non_blocking=True)
        speaker_id = batch["speaker_id"].to(device, non_blocking=True)
        if mel.dim() == 4:
            mel = mel.squeeze(1)

        # Remap val speaker IDs to train speaker IDs if needed
        if speaker_id_remap is not None:
            speaker_id = speaker_id_remap[speaker_id]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(mel)
            loss = F.cross_entropy(logits, speaker_id)

        total_loss += loss.item() * mel.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == speaker_id).sum().item()
        total += mel.size(0)
    model.train()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_asv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor()

    # ---- Build datasets ----
    print("Building training dataset ...")
    train_dataset = build_dataset(
        libritts_root=args.librispeech_root,
        split="train",
        libritts_subsets=args.librispeech_subsets,
        processor=processor,
    )
    print("Building validation dataset ...")
    val_dataset = build_dataset(
        libritts_root=args.librispeech_root,
        split="test",
        libritts_subsets=args.librispeech_subsets,
        processor=processor,
    )

    num_speakers = len(train_dataset.speaker_ids)

    # Build a remapping tensor so val speaker IDs align with train IDs.
    # Both splits sort speaker names independently; if a speaker is absent
    # from one split the indices will diverge.
    train_spk_map = train_dataset.speaker_ids   # {name: idx}
    val_spk_map = val_dataset.speaker_ids        # {name: idx}
    remap_tensor = None
    if train_spk_map != val_spk_map:
        # Create val_idx → train_idx mapping
        remap = torch.zeros(len(val_spk_map), dtype=torch.long)
        for name, val_idx in val_spk_map.items():
            if name in train_spk_map:
                remap[val_idx] = train_spk_map[name]
            else:
                remap[val_idx] = 0  # fallback (won't affect training)
                print(f"  [WARN] Val speaker '{name}' not in train set — ID mapped to 0")
        remap_tensor = remap.to(device)
        print(f"  Speaker ID remapping applied ({len(val_spk_map)} val → {num_speakers} train)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )

    # ---- Build model ----
    model = ASVClassifier(
        mel_channels=80, embed_dim=args.embed_dim, num_speakers=num_speakers,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"ASV Speaker Encoder Training")
    print(f"{'='*60}")
    print(f"  Device:          {device}")
    print(f"  AMP:             {use_amp}")
    print(f"  Speakers:        {num_speakers}")
    print(f"  Train samples:   {len(train_dataset)}")
    print(f"  Val samples:     {len(val_dataset)}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Early stopping:  {args.patience} epochs")
    print(f"  Output:          {output_path}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}")
        for batch in pbar:
            mel = batch["mel"].to(device, non_blocking=True)
            speaker_id = batch["speaker_id"].to(device, non_blocking=True)
            if mel.dim() == 4:
                mel = mel.squeeze(1)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(mel)
                loss = F.cross_entropy(logits, speaker_id)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item() * mel.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == speaker_id).sum().item()
            total += mel.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.3f}",
            )

        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total
        epoch_time = time.time() - epoch_start

        # ---- Validation ----
        val_loss, val_acc = _validate(
            model, val_loader, device, use_amp, speaker_id_remap=remap_tensor,
        )

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} | "
            f"time={epoch_time:.1f}s  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # ---- Save best encoder (no classification head) ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.encoder.state_dict(), output_path)
            print(
                f"  * New best!  Saved encoder -> {output_path}  "
                f"(val_acc={val_acc:.3f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping after {args.patience} epochs without "
                    f"improvement.  Best val_acc = {best_val_acc:.3f}"
                )
                break

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best validation accuracy:  {best_val_acc:.3f}")
    print(f"  ASV encoder saved to:      {output_path}")
    print(f"{'='*60}")
    print(
        f"\nYou can now run Stage 2 training with:\n"
        f"  python train_stage2.py\n"
        f"    --asv_checkpoint {output_path}\n"
        f"    --librispeech_root D:/thesis/data/LibriSpeech\n"
        f"    --librispeech_subsets train-clean-100\n"
        f"    --acg_checkpoint checkpoints/acg/acg_best.pt\n"
        f"    --output_dir checkpoints/rano\n"
        f"    --amp"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train AdaIN-VC speaker encoder (ASV) on LibriSpeech",
    )
    p.add_argument("--librispeech_root", type=str, required=True,
                    help="Path to LibriSpeech root (containing train-clean-100/)")
    p.add_argument("--librispeech_subsets", nargs="+", default=["train-clean-100"])
    p.add_argument("--output", type=str, default="checkpoints/asv.pt",
                    help="Where to save the trained encoder weights")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10,
                    help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True,
                    help="Enable automatic mixed precision (default: True)")
    p.add_argument("--no_amp", action="store_true",
                    help="Disable automatic mixed precision")
    args = p.parse_args()

    if args.no_amp:
        args.amp = False

    train_asv(args)
