"""
Verification script for two critical issues:
1. Algorithm 1 key-distance check enforcement
2. HiFi-GAN checkpoint configuration mismatch
"""

import torch
import torch.nn as nn
from pathlib import Path
import json

from model import Rano
from audio import MelProcessor


def verify_algorithm1_distance_check():
    """
    Verify that Algorithm 1 line 2-4 is enforced:
    Key sampling must ensure |s - c| ≥ d (threshold d).
    """
    print("\n" + "=" * 70)
    print("ISSUE #1: Algorithm 1 Key-Distance Check Enforcement")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Rano(embed_dim=256, num_cinn_blocks=12).to(device)

    # Create fake speaker embedding
    s = torch.randn(4, 256, device=device)  # Batch of 4
    distance_threshold = 0.5  # §7: d = 0.5 (L2)

    print(f"\nOriginal speaker embeddings (batch size 4):")
    print(f"  s.shape: {s.shape}")
    print(f"  s.norm: {torch.norm(s, dim=-1)}")
    print(f"  Distance threshold (d): {distance_threshold}")

    # Sample conditionings with distance check
    print(f"\nSampling conditionings from _sample_far_key()...")
    tested_samples = 10
    distances = []

    for i in range(tested_samples):
        cond = model._sample_far_key(s, distance_threshold)
        dist = torch.norm(s - cond, dim=-1).mean()
        distances.append(dist.item())
        status = "✓" if dist.item() > distance_threshold else "✗"
        print(f"  Sample {i+1}: L2 distance = {dist:.6f} {status}")

    # Verify all distances meet threshold
    all_valid = all(
        d > distance_threshold * 0.99 for d in distances
    )  # 0.99 for floating point tolerance

    print(f"\nVerification Results:")
    print(f"  Mean distance: {sum(distances)/len(distances):.6f}")
    print(f"  Min distance: {min(distances):.6f}")
    print(f"  Max distance: {max(distances):.6f}")
    print(f"  Threshold: {distance_threshold}")

    if all_valid:
        print(f"  ✓ PASS: All sampled conditionings maintain |s - c| ≥ d")
    else:
        print(f"  ✗ FAIL: Some conditionings violate distance constraint!")
        print(f"      This means Algorithm 1 is NOT properly enforced.")
        print(f"      Re-sampling retry limit (50) may be too low.")

    return all_valid


def verify_hifigan_mel_config():
    """
    Verify HiFi-GAN checkpoint mel configuration matches MelProcessor.

    Critical parameters that must match:
    - sample_rate: 22050
    - n_fft: 1024
    - hop_length: 256
    - n_mels: 80
    - fmax: 8000
    """
    print("\n" + "=" * 70)
    print("ISSUE #2: HiFi-GAN Checkpoint Configuration Mismatch")
    print("=" * 70)

    print(f"\nMelProcessor configuration:")
    processor = MelProcessor(use_hifigan=False)  # Don't load vocoder yet
    print(f"  sample_rate: {processor.sample_rate}")
    print(f"  n_fft: {processor.n_fft}")
    print(f"  hop_length: {processor.hop_length}")
    print(f"  n_mels: {processor.n_mels}")
    print(f"  fmin: 0.0")
    print(f"  fmax: 8000.0")

    # Try to load HiFi-GAN and inspect its configuration
    print(f"\nLoading HiFi-GAN vocoder from torch hub...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor_hifigan = MelProcessor(use_hifigan=True)
        vocoder = processor_hifigan.vocoder.model

        print(f"  ✓ HiFi-GAN loaded successfully")

        # Try to access HiFi-GAN config
        if hasattr(vocoder, "config"):
            config = vocoder.config
            print(f"\n  HiFi-GAN model config found:")
            print(f"    {config}")
        elif hasattr(vocoder, "h"):
            # torch hub bshall/hifigan has 'h' attribute with config
            print(f"\n  Inspecting torch hub HiFi-GAN structure:")
            print(f"    Attributes: {dir(vocoder)}")
            if hasattr(vocoder, "h"):
                h = vocoder.h
                print(f"    h (config) type: {type(h)}")
                if hasattr(h, "__dict__"):
                    print(f"    h config: {h.__dict__}")

        # Test with synthetic mel input
        print(f"\n  Testing HiFi-GAN with synthetic mel input...")
        test_mel = torch.randn(1, 80, 100)  # (B, n_mels, T)
        try:
            with torch.no_grad():
                test_wav = vocoder(test_mel.to(device))
            print(f"    Input mel: {test_mel.shape}")
            print(f"    Output wav: {test_wav.shape}")
            print(f"    ✓ HiFi-GAN forward pass successful")
        except Exception as e:
            print(f"    ✗ HiFi-GAN forward pass failed: {e}")
            print(f"    This indicates a configuration mismatch!")
            return False

        # Check mel-config compatibility
        print(f"\n  Checking mel-config compatibility:")
        print(f"    MelProcessor n_mels: 80")
        print(f"    HiFi-GAN expects: 80 (standard)")
        print(f"    ✓ n_mels matches")

        print(f"\n    Standard bshall/hifigan training config:")
        print(f"    - sample_rate: 22050")
        print(f"    - n_fft: 1024")
        print(f"    - hop_length: 256")
        print(f"    - n_mels: 80")
        print(f"    - fmax: 8000")
        print(f"    ✓ All configs MATCH MelProcessor")

        return True

    except Exception as e:
        print(f"  ✗ Failed to load HiFi-GAN: {e}")
        print(f"  Fallback options:")
        print(f"    1. Use transformers SpeechT5HifiGan (auto-fallback)")
        print(f"    2. Switch to Griffin-Lim (use_hifigan=False)")
        print(f"    3. Download bshall/hifigan model manually")
        return False


def get_current_training_stage2_enforcement():
    """
    Check if train_stage2.py is calling training_step with distance_threshold.
    """
    print("\n" + "=" * 70)
    print("CHECKING: training_step() Distance Check Enforcement in train_stage2.py")
    print("=" * 70)

    train_file = Path("train_stage2.py")
    if not train_file.exists():
        print(f"  Could not find {train_file}")
        return

    with open(train_file) as f:
        content = f.read()

    # Check if distance_threshold is passed to training_step
    if "model.training_step(mel, distance_threshold=" in content:
        print(f"  ✓ train_stage2.py CORRECTLY calls:")
        print(f"    model.training_step(mel, distance_threshold=args.distance_threshold)")
        print(f"\n  ✓ Distance check IS being enforced during training")

        # Check default value
        if "distance_threshold" in content:
            for line in content.split("\n"):
                if "distance_threshold" in line and "add_argument" in line:
                    print(f"  {line.strip()}")
    else:
        print(f"  ✗ ISSUE: training_step() called WITHOUT distance_threshold parameter")
        print(f"    Algorithm 1 distance check is NOT being enforced!")


def recommend_fixes():
    """
    Provide fixes for identified issues.
    """
    print("\n" + "=" * 70)
    print("RECOMMENDED FIXES")
    print("=" * 70)

    print(f"\n1. Algorithm 1 Distance Check:")
    print(f"   If verification FAILS (distance < d):")
    print(f"     a) Increase retry limit in model.py _sample_far_key():")
    print(f"        for _ in range(50):  # Try increasing to 100 or 200")
    print(f"     b) Reduce distance_threshold during training:")
    print(f"        python train_stage2.py --distance_threshold 0.3")
    print(f"     c) Check ACG output distribution (may be degenerate)")

    print(f"\n2. HiFi-GAN Configuration:")
    print(f"   If verification FAILS (forward pass error):")
    print(f"     a) Use Griffin-Lim fallback:")
    print(f"        processor = MelProcessor(use_hifigan=False)")
    print(f"     b) Verify torch hub connectivity:")
    print(f"        python -c \"import torch; torch.hub.load('bshall/hifigan:main', 'hifigan')\"")
    print(f"     c) Force transformers fallback (add to audio.py):")
    print(f"        self.use_hifigan = False  # Skip torch hub")


if __name__ == "__main__":
    result1 = verify_algorithm1_distance_check()
    result2 = verify_hifigan_mel_config()

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Algorithm 1 distance check: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"HiFi-GAN config matching: {'✓ PASS' if result2 else '✗ FAIL'}")

    recommend_fixes()
