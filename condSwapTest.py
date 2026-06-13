"""
COND-SWAP DIAGNOSTIC

Hypothesis: rano_model.anonymize(mel, key) internally calls
            cond = acg.generate(key)
and the cINN was only trained (via Lcons) with cond = real ASV
embeddings (s = uncompiled_asv(mel)). If acg.generate(key) produces
embeddings that are out-of-distribution relative to real ASV embeddings,
the cINN may not preserve content well when conditioned on them —
even though the triplet loss (EER) is satisfied.

This script tests that by swapping the conditioning vector:

  Path A (normal):     xa_normal = anonymizer.forward(mel, acg.generate(key))
  Path B (cond-swap):  xa_swapped = anonymizer.forward(mel, real_emb_from_other_speaker)

If WER(Path B) << WER(Path A), the ACG output distribution is the
likely root cause of the content-preservation gap.

Also prints embedding-space statistics comparing ACG-generated
embeddings vs real ASV embeddings.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from audio import MelProcessor
from model import Rano
from evaluate3 import (
    pseudo_inverse_vocoder, load_librispeech_transcripts, load_rano, load_asv,
    to_mono, peak_normalize, resample_to
)

# Normalized WER (Whisper's official normalizer)
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()

def calculate_wer(ref: str, hyp: str) -> float:
    ref_n = normalizer(ref)
    hyp_n = normalizer(hyp)
    r, h = ref_n.split(), hyp_n.split()
    if len(r) == 0:
        return 0.0
    dp = np.zeros((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1): dp[i][0] = i
    for j in range(len(h) + 1): dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return float(dp[-1][-1] / max(1, len(r)))


# ============================================================
# SETUP — EDIT THESE PATHS TO MATCH YOUR SETUP
# ============================================================
ACG_CHECKPOINT  = "checkpoints/acg/acg_final.pt"        # adjust path
ANON_CHECKPOINT = "checkpoints/rano/rano_final.pt"       # adjust path
EMBED_DIM       = 256
NUM_CINN_BLOCKS = 12
TEST_DIR        = "test_audio"
N_FILES         = 20   # how many utterances to test (keep small, this is a diagnostic)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

proc_sr = 16000
processor = MelProcessor(device=device, use_hifigan=False, sample_rate=proc_sr)

import whisper
print("Loading Whisper large-v2...")
whisper_model = whisper.load_model("large-v2").to(device)

def transcribe(wav, sr):
    wav = np.asarray(wav, dtype=np.float32)
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000).numpy()
    wav = peak_normalize(wav)
    audio = whisper.pad_or_trim(wav)
    n_mels = whisper_model.dims.n_mels
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(device)
    opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
    return whisper.decode(whisper_model, mel, opts).text


# ============================================================
# LOAD MODEL
# ============================================================
class Args:
    acg_checkpoint = ACG_CHECKPOINT
    anonymizer_ckpt = ANON_CHECKPOINT
    embed_dim = EMBED_DIM
    num_cinn_blocks = NUM_CINN_BLOCKS

args = Args()
rano_model = load_rano(args, device)
extract_asv_emb = load_asv(device)  # SpeechBrain ECAPA — eval ASV, NOT training ASV

# NOTE: the model's *internal* ASV (uncompiled_asv used during training, the
# lightweight TDNN SpeakerEncoder) is what Lcons was trained against — not
# SpeechBrain. If accessible, prefer rano_model.asv / rano_model.speaker_encoder
# for the "real embedding" extraction below, since that matches training
# distribution exactly. Fallback to SpeechBrain if not exposed.
internal_asv = getattr(rano_model, "asv", None) or getattr(rano_model, "speaker_encoder", None)
if internal_asv is not None:
    print("Using model's internal SpeakerEncoder for real-embedding extraction (matches training).")
else:
    print("WARNING: internal SpeakerEncoder not found on Rano model — "
          "falling back to SpeechBrain ECAPA embeddings for cond-swap. "
          "This is still a valid 'real speech embedding' but wasn't the exact "
          "embedding space Lcons trained against, so treat results as approximate.")


# ============================================================
# PARAMETER COLLAPSE CHECK
# ============================================================
print("\n" + "=" * 60)
print("PARAMETER COLLAPSE CHECK")
print("=" * 60)

for i, block in enumerate(rano_model.anonymizer.blocks):
    for name, subnet in [("psi", block.psi), ("phi", block.phi),
                          ("rho", block.rho), ("eta", block.eta)]:
        w = subnet.out_proj.weight.abs().sum().item()
        b = subnet.out_proj.bias.abs().sum().item()
        scales = [r.scale.item() for r in subnet.rrdb_blocks]
        print(f"  Block {i:2d} {name}: out_proj |w|sum={w:.6f}  |b|sum={b:.6f}  "
              f"rrdb_scales={[f'{s:.6f}' for s in scales]}")


# ============================================================
# LOAD DATA
# ============================================================
transcripts = load_librispeech_transcripts(TEST_DIR)
files = (
    list(Path(TEST_DIR).rglob("*.flac")) + list(Path(TEST_DIR).rglob("*.wav"))
)
files = [f for f in files if f.stem in transcripts]
# Sort by file size (largest first) and take top N_FILES
files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)[:N_FILES]
print(f"Testing on {len(files)} utterances (sorted by file size, largest first).\n")


# ============================================================
# PRECOMPUTE MELS + REAL EMBEDDINGS FOR ALL FILES
# ============================================================
mels = []
wavs = []
refs = []
real_embs = []

for path in files:
    wav_np, sr = sf.read(str(path))
    wav_np = to_mono(wav_np)
    wav_16k = resample_to(wav_np, sr, proc_sr)
    
    wav_t = torch.tensor(wav_16k).float()
    mel = processor.wav_to_mel(wav_t).to(device)

    if internal_asv is not None:
        with torch.no_grad():
            emb = internal_asv(mel)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
    else:
        emb_np = extract_asv_emb(torch.tensor(wav_16k).float().to(device))
        emb = torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0)
        # Project/pad to EMBED_DIM if shapes mismatch
        if emb.shape[-1] != EMBED_DIM:
            print(f"  NOTE: real-embedding dim {emb.shape[-1]} != model embed_dim {EMBED_DIM}. "
                  f"Cond-swap may fail or be invalid for this file.")

    mels.append(mel)
    wavs.append(wav_16k)
    refs.append(transcripts[path.stem])
    real_embs.append(emb)


# ============================================================
# EMBEDDING-SPACE STATS: ACG-generated vs real
# ============================================================
print("=" * 60)
print("EMBEDDING-SPACE STATISTICS")
print("=" * 60)

with torch.no_grad():
    gen_embs = []
    for _ in range(50):
        key = torch.randn(1, EMBED_DIM, device=device)
        sa = rano_model.acg.generate(key)
        gen_embs.append(sa)
    gen_embs = torch.cat(gen_embs, dim=0)        # (50, 256)

real_embs_stacked = torch.cat(real_embs, dim=0)  # (N, D_real)

print(f"  ACG-generated embeddings:")
print(f"    shape: {gen_embs.shape}")
print(f"    mean norm: {gen_embs.norm(dim=-1).mean().item():.4f}")
print(f"    per-dim std (avg): {gen_embs.std(dim=0).mean().item():.4f}")
print(f"    min/max: {gen_embs.min().item():.4f} / {gen_embs.max().item():.4f}")

print(f"\n  Real speaker embeddings ({'internal SpeakerEncoder' if internal_asv is not None else 'SpeechBrain ECAPA'}):")
print(f"    shape: {real_embs_stacked.shape}")
print(f"    mean norm: {real_embs_stacked.norm(dim=-1).mean().item():.4f}")
print(f"    per-dim std (avg): {real_embs_stacked.std(dim=0).mean().item():.4f}")
print(f"    min/max: {real_embs_stacked.min().item():.4f} / {real_embs_stacked.max().item():.4f}")

if gen_embs.shape[-1] != real_embs_stacked.shape[-1]:
    print(f"\n  *** DIMENSION MISMATCH: ACG generates dim={gen_embs.shape[-1]}, "
          f"real embeddings dim={real_embs_stacked.shape[-1]}. ***")
    print("  Cond-swap test below may not be directly comparable — "
          "results should be interpreted with this caveat.")


# ============================================================
# MEL-SPACE CONTENT PRESERVATION (Lcons-style MSE, no audio/Whisper)
# ============================================================
print("\n" + "=" * 60)
print("MEL-SPACE CONTENT PRESERVATION (Lcons-style MSE, no audio/Whisper)")
print("=" * 60)

mse_a, mse_b, mse_c = [], [], []

with torch.no_grad():
    for i in range(len(mels)):
        mel = mels[i]

        # Path A: ACG-generated cond (norm ~11.3)
        key = torch.randn(1, EMBED_DIM, device=device)
        xa_a, _ = rano_model.anonymizer(mel, rano_model.acg.generate(key))

        # Path B: real embedding from a different utterance (norm = 1.0)
        other_idx = (i + 1) % len(mels)
        xa_b, _ = rano_model.anonymizer(mel, real_embs[other_idx])

        # Path C: ACG-generated cond, L2-normalized to norm 1.0
        sa_norm = F.normalize(rano_model.acg.generate(key), dim=-1)
        xa_c, _ = rano_model.anonymizer(mel, sa_norm)

        mse_a.append(F.mse_loss(xa_a, mel).item())
        mse_b.append(F.mse_loss(xa_b, mel).item())
        mse_c.append(F.mse_loss(xa_c, mel).item())

        print(f"  {files[i].stem}: MSE_A={mse_a[-1]:.4f}  MSE_B={mse_b[-1]:.4f}  MSE_C={mse_c[-1]:.4f}")

print(f"\n  Mean MSE — Path A (ACG raw, norm~11.3):     {np.mean(mse_a):.4f}")
print(f"  Mean MSE — Path B (real emb, norm=1.0):     {np.mean(mse_b):.4f}")
print(f"  Mean MSE — Path C (ACG normalized, norm=1): {np.mean(mse_c):.4f}")
print("\n  Lower MSE = xa closer to original mel = better content preservation")
print("  If A >> B and A >> C: norm mismatch is the driver")
print("  If A ≈ B ≈ C: the issue is elsewhere (direction/distribution, not scale)")


# ============================================================
# CONDITIONING IMPACT TEST: Do different conditions produce different outputs?
# ============================================================
print("\n" + "=" * 60)
print("CONDITIONING IMPACT TEST")
print("=" * 60)

with torch.no_grad():
    # Test on first utterance
    mel_test = mels[0]
    
    # Generate two different conditioning vectors
    key1 = torch.randn(1, EMBED_DIM, device=device)
    key2 = torch.randn(1, EMBED_DIM, device=device)
    
    xa_1, _ = rano_model.anonymizer(mel_test, rano_model.acg.generate(key1))
    xa_2, _ = rano_model.anonymizer(mel_test, rano_model.acg.generate(key2))
    
    print(f"  Different ACG-generated conditions produce identical output? {torch.equal(xa_1, xa_2)}")
    print(f"  Max absolute difference: {(xa_1 - xa_2).abs().max().item():.6f}")
    print(f"  Mean absolute difference: {(xa_1 - xa_2).abs().mean().item():.6f}")


# ============================================================
# ROUND-TRIP TEST: Anonymize with key, restore with correct vs wrong key
# ============================================================
print("\n" + "=" * 60)
print("ROUND-TRIP TEST: Key-based Restoration")
print("=" * 60)

with torch.no_grad():
    mel_test = mels[0]
    
    # Anonymize with correct key
    key_correct = torch.randn(1, EMBED_DIM, device=device)
    xa_test, _ = rano_model.anonymize(mel_test, key_correct)
    
    # Restore with correct key
    xr_correct = rano_model.restore(xa_test, key_correct)
    
    # Restore with wrong key
    key_wrong = torch.randn(1, EMBED_DIM, device=device)
    xr_wrong = rano_model.restore(xa_test, key_wrong)
    
    mse_correct = F.mse_loss(xr_correct, mel_test).item()
    mse_wrong = F.mse_loss(xr_wrong, mel_test).item()
    
    print(f"  Correct-key restoration MSE: {mse_correct:.6f}")
    print(f"  Wrong-key restoration MSE:   {mse_wrong:.6f}")
    print(f"  Ratio (wrong/correct):       {mse_wrong / max(mse_correct, 1e-8):.2f}x")
    
    if mse_wrong > mse_correct:
        print(f"  ✓ Correct key restores better than wrong key (difference: {mse_wrong - mse_correct:.6f})")
    else:
        print(f"  ✗ Key doesn't matter for restoration (suspicious!)")


# ============================================================
# COND-SWAP TEST
# Path A: normal anonymize (cond = acg.generate(key))
# Path B: cond-swap (cond = real embedding from a DIFFERENT utterance)
# ============================================================
print("\n" + "=" * 60)
print("COND-SWAP TEST")
print("=" * 60)

wers_normal = []
wers_swapped = []
wers_path_c = []

with torch.no_grad():
    for i in range(len(mels)):
        mel = mels[i]
        wav = wavs[i]
        ref_text = refs[i]

        # ---- Path A: normal anonymization ----
        key = torch.randn(1, EMBED_DIM, device=device)
        xa_normal, _ = rano_model.anonymize(mel, key)

        anon_wav_normal = pseudo_inverse_vocoder(
            xa_normal, orig_wav=wav, sr=proc_sr,
            n_fft=1024, hop_length=256, n_mels=80,
            mode="griffinlim", griffinlim_iters=128, apply_post_filter=False,
        )
        anon_wav_normal = peak_normalize(anon_wav_normal)
        hyp_normal = transcribe(anon_wav_normal, proc_sr)
        w_normal = calculate_wer(ref_text, hyp_normal) * 100
        wers_normal.append(w_normal)

        # ---- Path B: cond-swap — use a REAL embedding from a different utterance ----
        other_idx = (i + 1) % len(mels)  # rotate to a different utterance
        real_cond = real_embs[other_idx]

        if real_cond.shape[-1] == EMBED_DIM:
            orig_dtype = next(rano_model.anonymizer.parameters()).dtype
            rano_model.anonymizer.double()
            xa_swapped, _ = rano_model.anonymizer(mel.double(), real_cond.double())
            xa_swapped = xa_swapped.float()
            rano_model.anonymizer.to(orig_dtype)

            anon_wav_swapped = pseudo_inverse_vocoder(
                xa_swapped, orig_wav=wav, sr=proc_sr,
                n_fft=1024, hop_length=256, n_mels=80,
                mode="griffinlim", griffinlim_iters=128, apply_post_filter=False,
            )
            anon_wav_swapped = peak_normalize(anon_wav_swapped)
            hyp_swapped = transcribe(anon_wav_swapped, proc_sr)
            w_swapped = calculate_wer(ref_text, hyp_swapped) * 100
            wers_swapped.append(w_swapped)
            
            # Save audios
            import os
            os.makedirs("cond_swap_output", exist_ok=True)
            sf.write(f"cond_swap_output/{files[i].stem}_normal.wav", anon_wav_normal, proc_sr)
            sf.write(f"cond_swap_output/{files[i].stem}_swapped.wav", anon_wav_swapped, proc_sr)
            sf.write(f"cond_swap_output/{files[i].stem}_orig.wav", wav, proc_sr)
        else:
            w_swapped = float("nan")

        # ---- Path C: Normalized generated embedding ----
        sa_unnormalized = rano_model.acg.generate(key)
        sa_normalized = F.normalize(sa_unnormalized, dim=-1)

        orig_dtype = next(rano_model.anonymizer.parameters()).dtype
        rano_model.anonymizer.double()
        xa_path_c, _ = rano_model.anonymizer(mel.double(), sa_normalized.double())
        xa_path_c = xa_path_c.float()
        rano_model.anonymizer.to(orig_dtype)

        anon_wav_path_c = pseudo_inverse_vocoder(
            xa_path_c, orig_wav=wav, sr=proc_sr,
            n_fft=1024, hop_length=256, n_mels=80,
            mode="griffinlim", griffinlim_iters=128, apply_post_filter=False,
        )
        anon_wav_path_c = peak_normalize(anon_wav_path_c)
        hyp_path_c = transcribe(anon_wav_path_c, proc_sr)
        w_path_c = calculate_wer(ref_text, hyp_path_c) * 100
        wers_path_c.append(w_path_c)

        sf.write(f"cond_swap_output/{files[i].stem}_path_c.wav", anon_wav_path_c, proc_sr)

        print(f"  {files[i].stem}: normal_WER={w_normal:.1f}%  "
              f"swapped_WER={w_swapped:.1f}%  path_c_WER={w_path_c:.1f}%  ref: {ref_text[:40]}...")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Path A (acg.generate(key) as cond):     mean WER = {np.mean(wers_normal):.1f}%")
if wers_swapped:
    print(f"  Path B (real embedding as cond):        mean WER = {np.nanmean(wers_swapped):.1f}%")
    print(f"  Path C (normalized acg as cond):        mean WER = {np.mean(wers_path_c):.1f}%")
    print(f"\n  Delta (A - B): {np.mean(wers_normal) - np.nanmean(wers_swapped):+.1f}pp")
    print(f"  Delta (A - C): {np.mean(wers_normal) - np.mean(wers_path_c):+.1f}pp")
    print("\n  Interpretation:")
    print("  - If Path C WER drops dramatically -> Normalization is the exact fix.")
    print("  - If Path C WER is similarly high -> The cINN weights are permanently corrupted")
    print("    by training against an impossible Ltri target, requiring Stage 2 retraining.")
else:
    print("  Path B skipped due to embedding dimension mismatch — see notes above.")
print("=" * 60)