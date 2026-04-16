# Rano — Restorable Speaker Anonymization via Conditional Invertible Neural Network

Implementation of the IJCNN 2025 paper:  
**"Rano: Restorable Speaker Anonymization via Conditional Invertible Neural Network"**  
Jianzong Wang, Xulong Zhang, Xiaoyang Qu — Ping An Technology (Shenzhen)

---

## Overview

Rano anonymizes a speaker's identity in speech while:
- Preserving speech content and prosody (speaker-independent information)
- **Enabling lossless restoration** from anonymized → original speech using a secret key
- Following Kerckhoffs's principle: security comes from key secrecy, not algorithm secrecy

Unlike prior SRD-based methods, Rano performs **audio-to-audio** transformation with a cINN — no explicit content disentanglement, no information loss.

```
original speech ──[ACG(key)]──► Anonymizer (cINN forward) ──► anonymized speech
                                                                        │
                                         ◄──────────── Restorer (cINN inverse, same weights)
                                         ▲
                                      same key
```

---

## Project Structure

```
rano/
├── rano/
│   ├── __init__.py          # Package exports
│   ├── blocks.py            # CINNBlock, INNBlock, RRDB, SubNet
│   ├── acg.py               # AnonymizationConditionGenerator
│   ├── anonymizer.py        # Anonymizer (cINN forward) + inverse (Restorer)
│   ├── speaker_encoder.py   # Lightweight TDNN speaker encoder (ASV)
│   ├── model.py             # Rano: top-level model integrating all components
│   ├── loss.py              # ConsistencyLoss, TripletLoss, RanoLoss
│   ├── audio.py             # MelProcessor: wav ↔ mel-spectrogram
│   ├── data.py              # VCTKDataset, LibriTTSDataset, build_dataset
│   └── metrics.py           # EER, GVD, pitch correlation, WER
├── scripts/
│   ├── train_stage1.py      # Pre-train ACG
│   ├── train_stage2.py      # Train full Rano model
│   ├── evaluate.py          # Compute GVD, pitch correlation
│   ├── security_eval.py     # Simulate key-attack restoration (Table III)
│   └── infer.py             # CLI: anonymize / restore audio files
├── configs/
│   └── train.yaml           # Hydra config for all hyperparameters
├── tests/
│   └── test_rano.py         # Unit tests for all modules
└── pyproject.toml           # uv/pip dependencies
```

---

## Module Reference

### `rano/blocks.py`

| Class | Description |
|---|---|
| `RRDB` | Residual-in-Residual Dense Block — used as sub-net backbone inside coupling layers |
| `SubNet` | Coupling sub-net: maps `(x, condition) → transform`; implements ψ, φ, ρ, η from Eq. 1 |
| `CINNBlock` | Full coupling block: `forward()` = anonymizer pass (Eq. 1), `inverse()` = restorer pass (Eq. 2) |
| `INNBlock` | Unconditional INN coupling block used inside ACG |

**Key equations implemented:**

```
# Forward (Eq. 1)
u^{i+1} = v^i ⊙ exp(ψ(v^i, c)) + φ(v^i, c)
v^{i+1} = v^i ⊙ exp(ρ(u^{i+1}, c)) + η(u^{i+1}, c)

# Inverse (Eq. 2)
v^i = (v^{i+1} − η(u^{i+1}, c)) ⊘ exp(ρ(u^{i+1}, c))
u^i = (u^{i+1} − φ(v^i, c)) ⊘ exp(ψ(v^i, c))
```

---

### `rano/acg.py` — `AnonymizationConditionGenerator`

INN-based generative model mapping speaker embedding space ↔ standard normal latent space.

- **Training**: maximise log-likelihood of real speaker embeddings via `loss()` (Eq. 4)
- **Inference**: `generate(key)` — sample `key ~ N(0,1)` → run INN in reverse → anonymous speaker embedding `sa`

```python
acg = AnonymizationConditionGenerator(embed_dim=256, num_blocks=8)
key = torch.randn(1, 256)
sa = acg.generate(key)  # (1, 256) anonymous speaker embedding
```

---

### `rano/anonymizer.py` — `Anonymizer`

Stack of `CINNBlock`s: performs the speaker-identity transformation.

- `forward(x, cond)` → anonymized mel `xa`
- `inverse(xa, cond)` → restored mel `xr` ≈ `x` (exact, lossless)
- **Restorer reuses Anonymizer's weights** — no extra parameters needed

```python
anonymizer = Anonymizer(mel_channels=80, cond_dim=256, num_blocks=8)
xa = anonymizer(x, cond)       # anonymize
xr = anonymizer.inverse(xa, cond)  # restore — same weights, reverse order
```

---

### `rano/speaker_encoder.py` — `SpeakerEncoder`

Lightweight TDNN-based speaker encoder producing L2-normalised embeddings from mel-spectrograms.

- Used as **ASV** for training (contrastive + consistency losses)
- **Different ASV** (SpeechBrain ECAPA-TDNN) used at evaluation to avoid train/eval leakage
- Differentiable: gradients flow through during stage 2 training

---

### `rano/model.py` — `Rano`

Top-level model integrating all components.

| Method | Description |
|---|---|
| `acg_loss(s)` | NLL loss for stage 1 ACG pre-training |
| `training_step(x)` | Full Algorithm 1: sample key, anonymize, compute Lcons + Ltri |
| `anonymize(x, key)` | Inference: returns `(xa, cond)` |
| `restore(xa, key)` | Inference: lossless restoration if correct key provided |

---

### `rano/loss.py`

| Class | Equation | Purpose |
|---|---|---|
| `ConsistencyLoss` | Eq. 5: `||x − f(x; s, θ)||²` | Preserve speaker-independent info (SII) |
| `TripletLoss` | Eq. 6: `max(0, κ + d(asv(xa), c) − d(asv(xa), s))` | Differentiate anonymized SDI from original |
| `RanoLoss` | Eq. 7: `λ1·Lcons + λ2·Ltri` (default: λ1=1, λ2=5) | Combined training objective |

---

### `rano/audio.py` — `MelProcessor`

- `wav_to_mel(wav)` → `(B, 1, F, T)` log-mel spectrogram at 22050 Hz
- `mel_to_wav_grifflim(mel)` → approximate waveform (Griffin-Lim baseline)
- `resample(wav, orig_sr)` → resample to 22050 Hz

Use HiFi-GAN vocoder for production-quality waveforms (not bundled; load externally).

---

### `rano/data.py`

| Class | Dataset | ~Size |
|---|---|---|
| `VCTKDataset` | VCTK 100+ speakers, 100+ hours | 11 GB |
| `LibriTTSDataset` | LibriTTS audiobooks, 500+ hours | 59 GB total |
| `build_dataset` | Combine VCTK + LibriTTS into one Dataset | — |

---

### `rano/metrics.py`

| Function | Paper metric | Notes |
|---|---|---|
| `compute_eer` | EER (%) ↑ | Higher = better anonymization |
| `compute_gvd` | GVD (dB) | ≥ 0 dB ideal |
| `compute_pitch_correlation` | ρf0 ↑ | Pearson corr of F0 sequences |
| `compute_wer` | WER (%) ↓ | Use Whisper large externally |
| `extract_f0` | — | PYIN via librosa |

---

## Installation

Requires Python 3.10+. Install [uv](https://github.com/astral-sh/uv) first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone <repo>
cd rano
uv sync              # install all dependencies from pyproject.toml
uv sync --extra dev  # also install dev/test tools
```

---

## Datasets

### VCTK (~11 GB)
```bash
mkdir -p data/vctk
wget -P data/vctk \
  "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
unzip data/vctk/VCTK-Corpus-0.92.zip -d data/vctk/
```

### LibriTTS (~6–59 GB depending on subsets)
```bash
mkdir -p data/libritts
# train-clean-100 (~6.3 GB) — minimum for training
wget -P data/libritts https://us.openslr.org/resources/60/train-clean-100.tar.gz
tar -xzf data/libritts/train-clean-100.tar.gz -C data/libritts/

# Optional larger subsets
wget -P data/libritts https://us.openslr.org/resources/60/train-clean-360.tar.gz
wget -P data/libritts https://us.openslr.org/resources/60/train-other-500.tar.gz
tar -xzf data/libritts/train-clean-360.tar.gz -C data/libritts/
tar -xzf data/libritts/train-other-500.tar.gz -C data/libritts/
```

Full list of subsets: https://www.openslr.org/60/

---

## Training

### Stage 1 — Pre-train ACG (100k iterations, ~4 hrs on A100)

```bash
uv run scripts/train_stage1.py \
  --vctk_root data/vctk/VCTK-Corpus-0.92 \
  --libritts_root data/libritts \
  --asv_checkpoint checkpoints/asv.pt \
  --output_dir checkpoints/acg \
  --iterations 100000 \
  --batch_size 32
```

### Stage 2 — Train Rano (200k iterations, ~12 hrs on A100)

```bash
uv run scripts/train_stage2.py \
  --vctk_root data/vctk/VCTK-Corpus-0.92 \
  --libritts_root data/libritts \
  --acg_checkpoint checkpoints/acg/acg_final.pt \
  --asv_checkpoint checkpoints/asv.pt \
  --output_dir checkpoints/rano \
  --iterations 200000 \
  --lambda1 1.0 --lambda2 5.0
```

Monitor with TensorBoard:
```bash
uv run tensorboard --logdir logs/rano
```

---

## Inference

### Anonymize a folder of audio files
```bash
uv run scripts/infer.py anonymize \
  --input data/test_speakers/ \
  --output data/anonymized/ \
  --checkpoint checkpoints/rano/rano_final.pt
# keys.json saved to data/anonymized/ — keep this secret!
```

### Restore from anonymized speech
```bash
uv run scripts/infer.py restore \
  --input data/anonymized/ \
  --output data/restored/ \
  --checkpoint checkpoints/rano/rano_final.pt \
  --key_file data/anonymized/keys.json
```

---

## Evaluation

```bash
# GVD + pitch correlation
uv run scripts/evaluate.py \
  --checkpoint checkpoints/rano/rano_final.pt \
  --eval_asv_checkpoint checkpoints/eval_asv.pt \
  --test_dir data/test/

# Security: simulate key attacks (Table III)
uv run scripts/security_eval.py \
  --checkpoint checkpoints/rano/rano_final.pt \
  --eval_asv_checkpoint checkpoints/eval_asv.pt \
  --test_dir data/test/ \
  --num_samples 100 \
  --attack_trials 10
```

For **EER** and **WER**, use external tools:
- EER: SpeechBrain ECAPA-TDNN speaker verification (`speechbrain/spkrec-ecapa-voxceleb`)
- WER: Whisper large-v3 (`openai/whisper-large-v3`)

---

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=rano --cov-report=term-missing
```

---

## Paper Results (Table I)

| Method | EER (%) ↑ | WER (%) ↓ | GVD (dB) | ρf0 ↑ | Naturalness ↑ | Restorable |
|---|---|---|---|---|---|---|
| Ground-Truth | 3.20 | 6.58 | — | — | 4.45 | — |
| VPC B1.a | 17.40 | 16.21 | −10.34 | 0.73 | 3.13 | ✗ |
| DeID-VC | 41.46 | 13.04 | 0.21 | 0.78 | 3.35 | ✗ |
| SALT | 45.32 | 10.77 | −0.03 | 0.83 | 3.71 | ✗ |
| **Rano (ours)** | **47.81** | 11.91 | **0.39** | 0.80 | 3.73 | **✓** |

---

## Citation

```bibtex
@inproceedings{wang2025rano,
  title={Rano: Restorable Speaker Anonymization via Conditional Invertible Neural Network},
  author={Wang, Jianzong and Zhang, Xulong and Qu, Xiaoyang},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2025},
  doi={10.1109/IJCNN64981.2025.11227971}
}
```
# rano-implementation
