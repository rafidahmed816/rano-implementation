# HiFi-GAN Vocoder Integration & Blank Audio Debugging Guide

## Changes Made

### 1. **Replaced Griffin-Lim with HiFi-GAN Vocoder** ([audio.py](audio.py))

- ✅ Added `HiFiGANVocoder` class that loads pre-trained model from torch hub
- ✅ Implements high-quality mel→wav conversion (§IV-A: "We utilize Hifi-GAN [39] as the vocoder")
- ✅ Supports fallback to transformers library if torch hub fails
- ✅ Replaced `mel_to_wav_grifflim()` with `mel_to_wav()` that uses HiFi-GAN by default

**Why HiFi-GAN?**

- Griffin-Lim is phase recovery via iteration (slow, low-quality)
- HiFi-GAN is a neural vocoder trained end-to-end (fast, high-quality, follows paper)
- Can handle mel-spectrograms with numerical issues better

### 2. **Enhanced Restoration Debugging** ([infer.py](infer.py))

- Added comprehensive mel-spectrogram statistics before/after restoration
- Added NaN/Inf detection
- Added vocoded waveform energy monitoring
- Clear warnings for numerical issues

### 3. **Created Comprehensive Debug Script** ([debug_restoration.py](debug_restoration.py))

- Traces full pipeline: wav → mel → anon → restore → wav
- Checks invertibility (L1 error between original and restored mel)
- Validates numerical stability at each step
- Generates debug output files with statistics

---

## Root Causes of Blank Audio (in order of likelihood)

### **1. Mel-Spectrogram Collapse (Most Common)**

When `mel_restored` has very small values or becomes all zeros/near-zeros:

- **Symptom:** `mel_restored.min()` and `max()` both very close to zero
- **Cause:** cINN blocks might have exploding/vanishing gradients during training
- **Fix:**
  ```python
  # In blocks.py, increase clamping range or add gradient monitoring
  _EXP_CLAMP = 4.0  # Current: [-4, 4] → try 5.0 or 6.0
  ```

### **2. Inverse Operation Not Exact**

The inverse should be mathematically exact, but numerical precision can fail:

- **Symptom:** `diff.max()` > 1e-3 in debug output
- **Cause:** Floating-point accumulation over 12 blocks × permutations
- **Check:** Run `debug_restoration.py --mode single` to see L1/L2 error
- **Fix:**
  ```python
  # Use float64 for better precision during restoration
  xr = model.restore(mel_anon.double().to(device), key.double()).float()
  ```

### **3. Key Not Matching**

- **Symptom:** Restoration works fine for anonymization but fails for restore
- **Cause:** JSON stores keys with limited float precision
- **Check:** Verify key values in `keys.json` have enough decimal places
- **Fix:**
  ```python
  # In infer.py anonymize_dir():
  json.dump(key_store, f, indent=2)  # Add indent for readability
  # Add precision option:
  class FloatPrecisionEncoder(json.JSONEncoder):
      def encode(self, o):
          if isinstance(o, float):
              return f"{o:.17e}"  # Full float64 precision
          return super().encode(o)
  ```

### **4. Griffin-Lim Failure** (Now fixed with HiFi-GAN)

- **Symptom:** Mel-spectrogram looks fine, but waveform is blank
- **Cause:** Griffin-Lim requires carefully tuned initialization
- **Why fixed:** HiFi-GAN handles edge cases better

### **5. Model Not Converged**

- **Symptom:** Uniform pattern in mel-restoration error
- **Cause:** Training didn't reach convergence
- **Check:** Review training loss in tensorboard `logs/rano/`
- **Look for:**
  - Total loss < 0.1
  - Consistency loss decaying
  - Triplet loss stabilizing

---

## How to Debug

### **Step 1: Quick Test (Single File)**

```bash
python debug_restoration.py \
  --mode single \
  --input data/train-clean-100/103/1240/103_1240_000000.flac \
  --output debug_output/ \
  --checkpoint checkpoints/rano/rano_final.pt \
  --use_hifigan
```

**Look for in output:**

- Is `mel_restored` nearly identical to `mel_orig` (L1 diff < 1e-3)?
- Is `wav_restored` silent (energy < 1e-5)?
- Where does the signal collapse? (at which step?)

### **Step 2: Check Invertibility**

```python
# In debug_restoration.py output, look at step [7]:
# If "L1 diff: > 1e-3" → cINN is not inverting properly
# If "L1 diff: < 1e-4" → cINN is perfect, issue is vocoding
```

### **Step 3: Vocoder Test**

```python
from audio import MelProcessor
import torch

processor = MelProcessor(use_hifigan=True)

# Create synthetic mel
mel = torch.randn(1, 80, 100)  # (B, n_mels, T)
wav = processor.mel_to_wav(mel)
print(f"Synthetic mel → wav: {wav.shape}, energy={wav.abs().max()}")

# If energy < 1e-5, HiFi-GAN is broken → try Griffin-Lim fallback
```

### **Step 4: Batch Test (Multiple Files)**

```bash
python debug_restoration.py \
  --mode batch \
  --input data/train-clean-100/ \
  --output debug_output/ \
  --checkpoint checkpoints/rano/rano_final.pt
```

---

## Quick Fixes

### **If vocoding produces silence:**

```python
# In audio.py, temporarily switch to Griffin-Lim:
processor = MelProcessor(use_hifigan=False)
wav = processor.mel_to_wav_grifflim(mel)
```

### **If invertibility error is high (L1 > 1e-3):**

```python
# In model.py, use double precision during restoration:
@torch.no_grad()
def restore(self, xa: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Use double precision for better numerical stability."""
    xa_double = xa.double()
    key_double = key.double()
    cond = self.acg.generate(key_double)
    xr = self.anonymizer.inverse(xa_double, cond)
    return xr.float()  # Convert back to float32
```

### **If training didn't converge:**

```bash
# Retrain with longer iterations:
python train_stage2.py \
  --vctk_root data/vctk \
  --libritts_root data/train-clean-100 \
  --iterations 300000 \  # Increased from 200000
  --batch_size 8 \       # Larger batch
  --accumulate_steps 2
```

---

## Expected Behavior After Fix

**Before fix (Griffin-Lim):**

- Restored waveform: Silent or very low energy
- Griffin-Lim struggles with processed mel-spectrograms

**After fix (HiFi-GAN):**

- Restored waveform: Natural speech with good energy
- HiFi-GAN handles restoration mel-spectrograms better
- Full restoration cycle should work end-to-end

---

## Paper Reference

From paper §IV-A (Experimental Setup):

> "We utilize Hifi-GAN [39] as the vocoder to convert Mel-spectrograms into waveforms to obtain anonymized speech and restored speech."

This ensures consistency with the paper's methodology. ✅

---

## Next Steps

1. **Run debug script on your data:**

   ```bash
   python debug_restoration.py --mode single --input <audio_file> ...
   ```

2. **Review debug output for which step fails**

3. **Apply appropriate fix from "Quick Fixes" section**

4. **Re-run inference:**

   ```bash
   python infer.py restore --input anon/ --output restored/ ...
   ```

5. **Listen to restored audio** — should match original speaker characteristics ✓
