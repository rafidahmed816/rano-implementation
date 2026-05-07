# Rano: Restoring Anonymized Speech — Knowledge Reference

## Overview

Rano is a speaker anonymization model that:
- Anonymizes speech by replacing speaker-dependent information (SDI) while preserving speaker-independent information (SII)
- Supports **lossless restoration** using a key (Kerckhoffs's principle)
- Uses a conditional Invertible Neural Network (cINN) as the core anonymizer/restorer

---

## Architecture Components

### 1. ACG — Anonymization Condition Generator (`acg.py`)
- **Type**: INN (Invertible Neural Network), N_acg = 8 blocks
- **Forward** (training): speaker embedding `s` → latent `z ~ N(0,1)`
- **Inverse** (inference): `key ~ N(0,1)` → anonymous speaker embedding `c = f⁻¹_acg(key)`
- **Purpose**: Maps a random key to a realistic anonymous speaker embedding
- **Stage 1 training loss (Eq. 4)**:
  ```
  L_ACG = E[ ||f(s)||² / 2  −  log|J| ] + τ||ω||²
  ```
  (τ||ω||² implemented as Adam weight_decay, not explicit loss term)

### 2. Anonymizer / Restorer (`anonymizer.py`)
- **Type**: cINN (conditional INN), N_inn = 12 blocks
- **Forward**: `(x, c) → xa` — anonymizes mel-spectrogram using condition `c`
- **Inverse**: `(xa, c) → x` — restores original mel using same condition (lossless)
- **Restorer shares all weights with Anonymizer** — no separate parameters
- Architecture: CINNBlock + FixedPermutation (seed 42+i) per block

### 3. ASV — Speaker Encoder (`speaker_encoder.py`)
- Pre-trained AdaIN-VC speaker encoder (frozen in Stage 2)
- Extracts speaker embeddings from mel-spectrograms differentially
- Used for: Stage 1 training data, consistency loss, triplet loss

---

## Training Strategy (Two Stages)

### Stage 1: Pre-train ACG
- Train ACG with NLL loss (Eq. 4) to learn the bijection between speaker embedding space and N(0,1)
- 100,000 iterations
- ASV is frozen throughout

### Stage 2: Train Anonymizer (Algorithm 1)

```
Input: x (original mel-spectrogram)

1. Extract speaker embedding:  s = ASV(x)   [ASV frozen]
2. Sample key ~ N(0,1), compute c = ACG(key)  [ACG frozen]
3. Repeat until ||s - c|| >= d:              [d = 0.5, L2]
     resample key, recompute c
4. Anonymize:    xa   = cINN(x, c)           [Forward pass 1]
5. Consistency:  x̂   = cINN(x, s)           [Forward pass 2]
6. L_cons = ||x − x̂||²                      [Eq. 5]
7. L_tri  = max(0, κ + d(ASV(xa), c) − d(ASV(xa), s))  [Eq. 6]
8. L_total = λ1·L_cons + λ2·L_tri           [Eq. 7]
```

- λ1 = 1, λ2 = 5, d = 0.5 (L2), κ = 0.3 (margin)
- 200,000 iterations
- Optimizer: Adam (β1=0.9, β2=0.99, ε=1e-8, lr=1e-5) + StepLR scheduler

---

## Loss Functions

### Consistency Loss — L_cons (Eq. 5) — SII-Consistency
```
L_cons = ||x − f(x; s, θ)||²
```
- `x̂ = f(x; s)`: anonymizer run with the **original** speaker embedding as condition
- Forces the cINN to learn that when condition = original speaker, nothing changes
- Ensures SII (content, prosody) is preserved without requiring parallel data

### Triplet Loss — L_tri (Eq. 6) — SDI-Differentiation
```
L_tri = max(0, κ + d(ASV(xa), c) − d(ASV(xa), s))
```
- anchor  = `ASV(xa)` — speaker embedding of anonymized speech
- positive = `c`       — anonymous condition (should be close to anchor)
- negative = `s`       — original speaker embedding (should be far from anchor)
- Goal: `d(ASV(xa), c) < d(ASV(xa), s)` by at least margin κ
- Distance metric in code: cosine distance (`1 − cosine_similarity`)

### Total Loss (Eq. 7)
```
L_total = λ1·L_cons + λ2·L_tri  = 1·L_cons + 5·L_tri
```

---

## Key and Anonymization Condition

- **Key**: random vector sampled from N(0,1); determines the anonymous identity
- **Condition c**: `c = ACG⁻¹(key)` — a realistic speaker embedding
- **Speaker-level anonymization**: same key for all utterances of a speaker → same anonymous voice
- **Utterance-level anonymization**: different key per utterance → different anonymous voices
- **Restoration** requires the same key used during anonymization (Kerckhoffs's principle)
- Distance constraint: `||s − c|| >= d` ensures the anonymous condition is sufficiently different from the original speaker

---

## Inference

### Anonymize (§4.1)
```python
c = ACG⁻¹(key)          # key ~ N(0,1)
xa = cINN_forward(x, c)  # anonymized mel
# Store key for restoration
```

### Restore (§4.2)
```python
c = ACG⁻¹(key)           # same key as anonymization
x = cINN_inverse(xa, c)  # lossless restoration
```

---

## Data & Hyperparameters

| Parameter | Value |
|---|---|
| Sample rate | 22,050 Hz |
| Mel channels | 80 |
| Embed dim | 256 |
| N_acg (ACG blocks) | 8 |
| N_inn (cINN blocks) | 12 |
| λ1 | 1.0 |
| λ2 | 5.0 |
| distance threshold d | 0.5 (L2) |
| triplet margin κ | 0.3 |
| ACG τ (weight decay) | 0.5 |
| Stage 1 iters | 100,000 |
| Stage 2 iters | 200,000 |
| Learning rate | 1e-5 |
| Vocoder | HiFi-GAN |

**Training data**: VCTK (~100h, 100+ speakers) + LibriTTS (~500h audiobooks); 15% test split

---

## Evaluation Metrics

- **EER** (Equal Error Rate): higher = better anonymization (harder for ASV to identify speaker)
- **GVD** (Gain of Voice Distinctiveness): speaker diversity before vs. after anonymization
- Evaluation ASV: ECAPA-TDNN (SpeechBrain) — **different** from training ASV (fairness)

---

## Code Implementation Notes

- `_sample_far_key()`: vectorized — samples N=32 candidates per batch element in one ACG pass, picks the one with max distance from `s` (deviation from Algorithm 1's sequential resampling, but equivalent in expectation and ~32× faster)
- Two cINN forward passes are batched by concatenating along the batch dim (`2B` forward call, then split) — avoids running 12 cINN blocks twice
- Gradients flow through `ASV(xa)` into `xa` and into the anonymizer for L_tri — intentionally not wrapped in `no_grad`
- ACG's `generate()` is `@torch.no_grad()` — ACG is frozen in Stage 2