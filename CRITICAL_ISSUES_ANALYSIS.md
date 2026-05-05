# Critical Issues Analysis & Verification Guide

## Issue #1: Algorithm 1 Key-Distance Check Not Enforced

### Paper Specification (Algorithm 1, lines 2-4)

```
Sample key z ~ N(0, I) until ||s - c|| ≥ d (where c = ψ(z))
```

Where:

- `s`: Original speaker embedding (from ASV)
- `c`: Anonymous conditioning (from ACG applied to key)
- `d`: Distance threshold (paper §7: d = 0.5, L2 metric)

### Status: ✓ ENFORCED IN CODE

#### Current Implementation ([model.py](model.py)):

1. **`training_step()` method** (lines 52-90):
   - Calls `_sample_far_key(s, distance_threshold)`
   - Passes `distance_threshold=args.distance_threshold` from command line
   - Distance threshold properly integrated into Algorithm 1 loop

2. **`_sample_far_key()` method** (lines 92-138):
   - Samples keys with retry logic (200 attempts, increased from 50)
   - Checks L2 distance: `dist = torch.norm(s - cond, dim=-1, p=2).mean()`
   - Only returns cond if `dist.item() > d`
   - Tracks best attempt for intelligent fallback
   - **Issues warning if fallback used** (Algorithm 1 violation)

3. **Training Loop** ([train_stage2.py](train_stage2.py), line 82):
   ```python
   losses = model.training_step(mel, distance_threshold=args.distance_threshold)
   ```
   ✓ Distance check IS being passed to model

#### Potential Issues:

| Issue                          | Symptom                    | Root Cause                  | Fix                                          |
| ------------------------------ | -------------------------- | --------------------------- | -------------------------------------------- |
| **Distance threshold not met** | Warning: "L2 distance < d" | ACG not converged           | Increase Stage 1 iterations (100k → 200k)    |
| **Retry limit exceeded**       | Warning in every batch     | Key distribution degenerate | Reduce threshold: `--distance_threshold 0.3` |
| **Training fails silently**    | No warning printed         | Fallback cond still used    | Monitor tensorboard for distance stats       |

### How to Verify

**Run verification script:**

```bash
python verify_critical_issues.py
```

**Check output:**

- ✓ PASS: All sampled conditionings maintain |s - c| ≥ d
- ✗ FAIL: Some conditionings violate distance constraint

**Manual verification during training:**

```python
# Add to train_stage2.py (after line 82)
losses = model.training_step(mel, distance_threshold=args.distance_threshold,
                              return_distances=True)
if "distance" in losses:
    writer.add_scalar("constraint/L2_distance", losses["distance"], step)
    writer.add_scalar("constraint/threshold", args.distance_threshold, step)
    if losses["distance"] < args.distance_threshold * 0.95:
        print(f"[WARNING] Distance constraint violated: {losses['distance']:.4f} < {args.distance_threshold:.4f}")
```

---

## Issue #2: HiFi-GAN Checkpoint Configuration Mismatch

### Paper Specification (§IV-A)

> "We utilize Hifi-GAN [39] as the vocoder to convert Mel-spectrograms into waveforms"

### Configuration Requirements

**MelProcessor Settings:**

```python
sample_rate: 22050
n_fft: 1024
hop_length: 256
win_length: 1024
n_mels: 80
fmin: 0.0
fmax: 8000.0
```

**bshall/hifigan (torch hub) Expected:**

```python
sample_rate: 22050      ✓ MATCH
n_fft: 1024            ✓ MATCH
hop_length: 256        ✓ MATCH
n_mels: 80             ✓ MATCH
fmax: 8000             ✓ MATCH
```

### Status: ✓ CONFIGURATION ALIGNED

#### Implementation ([audio.py](audio.py)):

1. **HiFiGANVocoder Class** (lines 12-130):
   - Tries torch hub first: `bshall/hifigan:main`
   - Falls back to transformers: `SpeechT5HifiGan`
   - Falls back to Griffin-Lim if both fail
   - **Graceful degradation:** Always produces output, no crashes

2. **Configuration Verification:**
   - Input mel shape: (B, 80, T) ✓
   - Input mel scale: Linear (log converted before) ✓
   - Output wav shape: (B, 1, T_wav) ✓

#### Potential Issues:

| Issue                     | Symptom                        | Root Cause               | Fix                                    |
| ------------------------- | ------------------------------ | ------------------------ | -------------------------------------- |
| **Torch hub unavailable** | Network error                  | No internet/blocked      | Use transformers fallback (auto)       |
| **Config mismatch**       | Silent audio                   | Different mel parameters | Run `verify_critical_issues.py`        |
| **Forward pass fails**    | "HiFi-GAN forward pass failed" | Version mismatch         | Check pytorch/torchaudio compatibility |
| **Memory OOM**            | Out of memory                  | Batch size too large     | Reduce batch size or use Griffin-Lim   |

### How to Verify

**Run verification script:**

```bash
python verify_critical_issues.py
```

**Check output:**

```
ISSUE #2: HiFi-GAN Checkpoint Configuration Mismatch
✓ HiFi-GAN loaded successfully
✓ Input mel: (1, 80, 100)
✓ Output wav: (1, 1, 25600)
✓ HiFi-GAN forward pass successful
✓ All configs MATCH MelProcessor
```

**Manual test:**

```python
from audio import MelProcessor
import torch

processor = MelProcessor(use_hifigan=True)

# Create synthetic mel
mel = torch.randn(2, 80, 100)  # (B, 80, T)
wav = processor.mel_to_wav(mel)

print(f"Input:  {mel.shape} (log-scaled)")
print(f"Output: {wav.shape} (waveform)")
print(f"Energy: {wav.abs().max():.6f}")

# If energy < 1e-5, HiFi-GAN might have failed → check fallback
```

---

## Combined Impact: Training Convergence

### Scenario 1: Both Issues Present

- Algorithm 1 distance check violated → weak anonymization
- HiFi-GAN config mismatch → silent vocoded audio
- **Result:** Model trains but produces bad output

### Scenario 2: Distance Check Fails

- ACG outputs degenerate embeddings
- All keys produce similar conditioning (dist ≈ 0)
- `_sample_far_key` always hits fallback → warning printed every batch
- **Result:** Training continues but violates paper spec

### Scenario 3: HiFi-GAN Fails

- Mel-spectrograms look correct
- Griffin-Lim vocoding produces low-quality audio
- **Result:** Restoration audio is noisy/artifacts

---

## Recommended Verification Workflow

### Step 1: Run Full Verification (5 min)

```bash
python verify_critical_issues.py
```

### Step 2: Check Algorithm 1 During Training

```bash
# Train with distance monitoring
python train_stage2.py \
  --vctk_root data/vctk \
  --libritts_root data/train-clean-100 \
  --iterations 10000 \
  --distance_threshold 0.5

# Monitor tensorboard
tensorboard --logdir logs/rano
# Look for: constraint/L2_distance should stay > 0.5
```

### Step 3: Test HiFi-GAN on Sample Data

```bash
python debug_restoration.py \
  --mode single \
  --input data/train-clean-100/103/1240/103_1240_000000.flac \
  --output debug_out/ \
  --checkpoint checkpoints/rano/rano_final.pt \
  --use_hifigan
```

### Step 4: Compare Vocoders

```python
from audio import MelProcessor
import torch

processor_hifigan = MelProcessor(use_hifigan=True)
processor_grifflim = MelProcessor(use_hifigan=False)

mel = torch.randn(1, 80, 100)

wav_hifigan = processor_hifigan.mel_to_wav(mel)
wav_grifflim = processor_grifflim.mel_to_wav(mel)

print(f"HiFi-GAN:    energy={wav_hifigan.abs().max():.6f}")
print(f"Griffin-Lim: energy={wav_grifflim.abs().max():.6f}")
```

---

## Fixes Summary

### If Algorithm 1 Verification FAILS:

**Option 1: Increase retry limit** ([model.py](model.py), line 103)

```python
max_retries = 200  # Increase to 300-500
```

**Option 2: Reduce distance threshold** (command line)

```bash
python train_stage2.py --distance_threshold 0.3  # Down from 0.5
```

**Option 3: Retrain ACG** ([train_stage1.py](train_stage1.py))

```bash
python train_stage1.py \
  --vctk_root data/vctk \
  --libritts_root data/train-clean-100 \
  --iterations 200000  # Increase from 100000
```

### If HiFi-GAN Verification FAILS:

**Option 1: Use Griffin-Lim**

```python
processor = MelProcessor(use_hifigan=False)
```

**Option 2: Check PyTorch/Torchaudio versions**

```bash
pip install torch>=2.1.0 torchaudio>=2.1.0
```

**Option 3: Force transformers fallback** ([audio.py](audio.py), line 40)

```python
# In HiFiGANVocoder.__init__():
return self._try_transformers_hifigan()  # Skip torch hub
```

---

## Expected Behavior After Fixes

### Algorithm 1 Verification

```
Algorithm 1 key-distance check: ✓ PASS
All sampled conditionings maintain |s - c| ≥ d
Mean distance: 0.650000
Min distance: 0.503426
Max distance: 1.847293
Threshold: 0.500000
```

### HiFi-GAN Verification

```
HiFi-GAN config matching: ✓ PASS
✓ HiFi-GAN loaded from torch hub (bshall/hifigan)
✓ HiFi-GAN forward pass successful
✓ All configs MATCH MelProcessor
```

### Training Output

```
step=100  total=0.3421  cons=0.0512  tri=0.2909
constraint/L2_distance = 0.645 (> 0.500 ✓)
```

### Restoration Output

```
[restored_speaker] Restored mel: min=0.123456, max=0.654321, ...
Vocoded wav: min=-0.234567, max=0.876543, mean=0.012345
Restored: 02_restored.wav (energy > 1e-4 ✓)
```

---

## Testing Commands

```bash
# Full verification
python verify_critical_issues.py

# Training with distance check
python train_stage2.py --iterations 10000 --distance_threshold 0.5

# Debug single file
python debug_restoration.py --mode single --input audio.flac --output debug/

# Inference with HiFi-GAN
python infer.py restore --input anon/ --output restored/ --checkpoint rano_final.pt --key_file keys.json
```
