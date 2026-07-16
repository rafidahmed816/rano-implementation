import sys
import os
import shutil
import argparse
import json
import time
from pathlib import Path
import warnings

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

# ==========================================
# LIBROSA WARM-UP
# ==========================================
print("Warming up Librosa...")
_ = librosa.feature.mfcc(y=np.zeros(16000), sr=16000, n_mfcc=13)
_ = librosa.yin(np.zeros(16000), fmin=50, fmax=400, sr=16000)

from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model import Rano
from audio import MelProcessor

MCD_SR = 16000

# Mel clamping for HiFi-GAN vocoder (from quick_infer.py)
# cINN pushes some frames below -11.5 dB; HiFi-GAN was never trained below this
MEL_MIN = -11.5
MEL_MAX = 2.0

# ============================================================
# SPECTRAL POST-FILTER
# Smooths magnitude spectrogram before inversion.
# Reduces musical noise artifacts that inflate MCD.
# ============================================================

def spectral_post_filter(mag: np.ndarray, alpha: float = 1.4) -> np.ndarray:
    """
    Wiener-inspired spectral over-subtraction post-filter.
    Suppresses the low-energy inter-harmonic noise floor that Griffin-Lim
    and pseudo-inverse leave behind — the primary driver of high MCD.

    mag   : (n_fft//2+1, T) magnitude spectrogram
    alpha : over-subtraction factor. 1.4 is a safe default.
            Higher = more noise suppression, more distortion risk.
    Returns filtered magnitude, same shape.
    """
    power = mag ** 2
    # Estimate noise floor as the mean power across time (stationary noise assumption)
    noise_floor = np.mean(power, axis=1, keepdims=True)
    # Over-subtract and half-wave rectify (no negative power)
    cleaned_power = np.maximum(power - alpha * noise_floor, 0.0)
    return np.sqrt(cleaned_power)


# ============================================================
# VOCODER CORE
# ============================================================

def pseudo_inverse_vocoder(
    xa_tensor: torch.Tensor,
    orig_wav: np.ndarray = None,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    mode: str = "griffinlim",          # "griffinlim" | "phase_save" | "perturbed_phase"
    pitch_shift_semitones: float = 3.5, # only used in perturbed_phase mode
    griffinlim_iters: int = 128,        # increased from 64 for better phase convergence
    apply_post_filter: bool = True,     # spectral post-filter to reduce MCD
) -> np.ndarray:
    """
    Unified vocoder with three modes:

    griffinlim      — for anonymized output. Zero phase leakage.
                      Best EER safety. Worst MCD/WER.

    perturbed_phase — for anonymized output. Uses pitch-shifted original phase.
                      Preserves prosodic rhythm → better rho_f0 and WER.
                      Pitch shift disrupts speaker identity → EER stays near 50%.

    phase_save      — for restored output. Glues back original phase exactly.
                      Near-perfect Sim_spk and MCD for restoration path.
    """
    device = xa_tensor.device

    # Shape guard
    xa_sq = xa_tensor.squeeze().cpu().numpy()
    assert xa_sq.ndim == 2 and xa_sq.shape[0] == n_mels, \
        f"Expected ({n_mels}, T) after squeeze, got {xa_sq.shape}"

    # ── Step 1: Invert natural log ───────────────────────────────────────────
    # MelProcessor.wav_to_mel uses torch.log (natural log of power spectrogram)
    # So inverse is simply exp()
    linear_power_mel = torch.exp(xa_tensor)  # (1, 80, T) power mel

    # ── Step 2: Mel → STFT power via InverseMelScale ────────────────────────
    inv_mel_scale = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sr,
        f_min=0.0,
        f_max=8000.0,
    ).to(device)

    power_stft = inv_mel_scale(linear_power_mel)          # (1, 513, T)
    mag_stft = torch.sqrt(power_stft).squeeze().cpu().detach().numpy()  # (513, T)

    # ── Step 3: Optional spectral post-filter ───────────────────────────────
    # Reduces the musical noise floor that massively inflates MCD.
    if apply_post_filter:
        mag_stft = spectral_post_filter(mag_stft, alpha=1.4)

    # ── Step 4: Phase reconstruction ────────────────────────────────────────
    if mode == "phase_save":
        # Exact original phase — maximum fidelity for restoration
        assert orig_wav is not None, "phase_save mode requires orig_wav"
        orig_stft = librosa.stft(orig_wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        orig_phase = np.exp(1j * np.angle(orig_stft))
        min_t = min(mag_stft.shape[1], orig_phase.shape[1])
        recon_stft = mag_stft[:, :min_t] * orig_phase[:, :min_t]
        return librosa.istft(recon_stft, hop_length=hop_length, win_length=n_fft)

    elif mode == "perturbed_phase":
        # Pitch-shifted phase — preserves prosodic structure, disrupts speaker identity
        # Improves rho_f0 and WER vs pure Griffin-Lim while keeping EER near 50%
        assert orig_wav is not None, "perturbed_phase mode requires orig_wav"
        orig_shifted = librosa.effects.pitch_shift(
            orig_wav.astype(np.float32), sr=sr, n_steps=pitch_shift_semitones
        )
        shifted_stft = librosa.stft(orig_shifted, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        shifted_phase = np.exp(1j * np.angle(shifted_stft))
        min_t = min(mag_stft.shape[1], shifted_phase.shape[1])
        recon_stft = mag_stft[:, :min_t] * shifted_phase[:, :min_t]
        return librosa.istft(recon_stft, hop_length=hop_length, win_length=n_fft)

    else:
        # Pure Griffin-Lim — no phase information borrowed from original
        return librosa.griffinlim(
            mag_stft,
            n_iter=griffinlim_iters,
            hop_length=hop_length,
            win_length=n_fft,
        )


# ============================================================
# TRANSCRIPT LOADER
# ============================================================

def load_librispeech_transcripts(test_dir: str) -> dict:
    """Load ground truth transcripts from LibriSpeech .trans.txt files."""
    transcripts = {}
    for trans_file in Path(test_dir).rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1].lower()
    return transcripts


# ============================================================
# AUDIO UTILS
# ============================================================

def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x.mean(axis=1)
    return x

def peak_normalize(wav: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(wav))
    if max_val > 1e-8:
        return wav / max_val
    return wav

def resample_to(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    t = torch.tensor(wav, dtype=torch.float32)
    t = torchaudio.functional.resample(t, orig_sr, target_sr)
    return t.numpy()


# ============================================================
# METRICS
# ============================================================

def extract_f0(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    try:
        return librosa.yin(wav, fmin=50, fmax=400, sr=sr)
    except Exception:
        return np.zeros(10)

def calculate_rho_f0(f0_ref: np.ndarray, f0_deg: np.ndarray) -> float:
    n = min(len(f0_ref), len(f0_deg))
    if n < 10: return float("nan")
    a, b = f0_ref[:n], f0_deg[:n]
    valid = (a > 0) & (b > 0) & ~np.isnan(a) & ~np.isnan(b)
    if np.sum(valid) < 5: return float("nan")
    a_v, b_v = a[valid], b[valid]
    if np.std(a_v) < 1e-6 or np.std(b_v) < 1e-6: return 0.0
    return float(np.corrcoef(a_v, b_v)[0, 1])

def calculate_gvd(mel_ref: torch.Tensor, mel_deg: torch.Tensor) -> float:
    var_ref = torch.var(mel_ref.squeeze(), dim=-1)
    var_deg = torch.var(mel_deg.squeeze(), dim=-1)
    gv_ref_db = 10 * torch.log10(var_ref + 1e-8)
    gv_deg_db = 10 * torch.log10(var_deg + 1e-8)
    gvd = torch.sqrt(torch.mean((gv_ref_db - gv_deg_db) ** 2))
    return float(gvd.item())

def calculate_mcd(
    wav_ref: np.ndarray, wav_deg: np.ndarray,
    sr_ref: int, sr_deg: int, target_sr: int = MCD_SR
) -> float:
    wav_ref = resample_to(np.asarray(wav_ref, dtype=np.float32), sr_ref, target_sr)
    wav_deg = resample_to(np.asarray(wav_deg, dtype=np.float32), sr_deg, target_sr)
    wav_ref = peak_normalize(wav_ref)
    wav_deg = peak_normalize(wav_deg)

    for _, w in [("ref", wav_ref), ("deg", wav_deg)]:
        if not np.all(np.isfinite(w)): return float("nan")

    mfcc_ref = librosa.feature.mfcc(y=wav_ref, sr=target_sr, n_mfcc=13)[1:, :]
    mfcc_deg = librosa.feature.mfcc(y=wav_deg, sr=target_sr, n_mfcc=13)[1:, :]

    if mfcc_ref.shape[1] == 0 or mfcc_deg.shape[1] == 0: return float("nan")

    D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_deg, metric='euclidean')
    mean_dist = D[-1, -1] / len(wp)
    return float((10.0 * np.sqrt(2.0) / np.log(10.0)) * mean_dist)

def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
    return float(np.dot(emb1, emb2) / denom)

def calculate_eer(y_true: list, y_score: list) -> float:
    if len(np.unique(y_true)) < 2: return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer * 100.0)


# ============================================================
# ASR / WER
# ============================================================

def transcribe(whisper_model, wav: np.ndarray, sr: int) -> str:
    import whisper
    wav = np.asarray(wav, dtype=np.float32)
    if sr != 16000:
        wav = torchaudio.functional.resample(torch.tensor(wav), sr, 16000).numpy()
    wav = peak_normalize(wav)
    audio = whisper.pad_or_trim(wav)
    n_mels = whisper_model.dims.n_mels  # 80 for base/small, 128 for large/large-v2
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(whisper_model.device)
    opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
    return whisper.decode(whisper_model, mel, opts).text
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
# MODEL LOADING
# ============================================================

def load_rano(args, device: torch.device) -> Rano:
    model = Rano(embed_dim=args.embed_dim, num_cinn_blocks=args.num_cinn_blocks).to(device)
    def load_weights(path: str, module: torch.nn.Module, name: str):
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        module.load_state_dict(ckpt, strict=False)
        print(f"  Loaded {name} from {path}")
    load_weights(args.acg_checkpoint,  model.acg,       "ACG")
    load_weights(args.anonymizer_ckpt, model.anonymizer, "Anonymizer")
    model.eval()
    return model

def load_asv(device: torch.device):
    print("Loading SpeechBrain ECAPA-TDNN ASV model...")
    _orig_symlink = os.symlink
    def _patched_symlink(src, dst, target_is_directory=False, *a, **kw):
        try: _orig_symlink(src, dst, target_is_directory, *a, **kw)
        except OSError:
            if os.path.isdir(src): shutil.copytree(src, dst, dirs_exist_ok=True)
            else: shutil.copy2(src, dst)
    os.symlink = _patched_symlink
    try: from speechbrain.inference.speaker import EncoderClassifier
    except ImportError: from speechbrain.pretrained import EncoderClassifier
    asv = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_speechbrain_asv",
        run_opts={"device": str(device)},
    )
    os.symlink = _orig_symlink

    def extract_emb(wav_tensor: torch.Tensor) -> np.ndarray:
        if wav_tensor.ndim == 1: wav_tensor = wav_tensor.unsqueeze(0)
        with torch.no_grad(): return asv.encode_batch(wav_tensor).squeeze().cpu().numpy()
    return extract_emb


# ============================================================
# MAIN EVALUATION
# ============================================================

def safe_mean(arr: list) -> float:
    clean = [x for x in arr if not np.isnan(x) and not np.isinf(x)]
    return float(np.mean(clean)) if clean else float("nan")


def evaluate(args):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve anonymization vocoder mode
    vocoder_mode = args.vocoder  # "hifigan" or "pseudoinverse"
    anon_mode = args.anon_mode   # only for pseudoinverse mode
    print(f"Vocoder: {vocoder_mode}")
    if vocoder_mode == "pseudoinverse":
        print(f"  Pseudo-inverse mode: {anon_mode}")
        print(f"  Spectral post-filter: {'ON' if args.post_filter else 'OFF'}")

    save_audio_dir = Path(args.save_audio_dir)
    save_audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated audio will be saved to: {save_audio_dir}")

    # MelProcessor — 22050Hz to match training (train_stage2.py uses default=22050)
    # Using 16kHz caused cINN numerical explosions (values up to 1e18) because
    # the mel filterbank produces OOD inputs for the 22050Hz-trained model.
    processor = MelProcessor(device=device, use_hifigan=(vocoder_mode == "hifigan"), sample_rate=22050)
    proc_sr = 22050

    rano_model = load_rano(args, device)
    extract_asv_emb = load_asv(device)

    whisper_model = None
    if args.compute_wer:
        import whisper
        whisper_model = whisper.load_model(args.whisper_model).to(device)
        print(f"Whisper '{args.whisper_model}' loaded.")

    # Ground truth transcripts from LibriSpeech
    transcripts = load_librispeech_transcripts(args.test_dir)
    print(f"Loaded {len(transcripts)} ground truth transcripts.")

    files = (
        list(Path(args.test_dir).rglob("*.flac")) +
        list(Path(args.test_dir).rglob("*.wav"))
    )
    if not files: raise RuntimeError(f"No files found in: {args.test_dir}")
    print(f"Found {len(files)} utterances.")

    speaker_keys: dict[str, torch.Tensor] = {}
    embeddings = {"orig": {}, "anon": {}, "restored": {}}
    metrics: dict[str, list] = {
        "gvd": [], "rho_f0": [], "wer": [],
        "sim_spk_restored": [], "mcd_restored": [],
    }

    with torch.no_grad():
        for path in tqdm(files, desc="Processing"):
            # LibriSpeech files are SPEAKER-CHAPTER-UTT.flac, so the real speaker
            # is the first filename token — NOT the parent folder (which is the
            # chapter). Using the folder counts chapters of one speaker as many
            # speakers and makes EER degenerate. Fall back to the folder for
            # non-LibriSpeech layouts.
            _parts = path.stem.split("-")
            spk = _parts[0] if len(_parts) >= 2 and _parts[0].isdigit() else path.parent.name
            wav_np, sr = sf.read(str(path))
            wav_np = to_mono(wav_np)

            # Resample to 16kHz for consistent processing and phase alignment
            wav_16k = resample_to(wav_np, sr, proc_sr)

            # Skip utterances too short for STFT window
            if len(wav_16k) < 1024:
                print(f"Skipping {path.name} — too short ({len(wav_16k)} samples)")
                continue

            wav_t = torch.tensor(wav_16k).float()
            mel = processor.wav_to_mel(wav_t).to(device)

            if spk not in speaker_keys:
                speaker_keys[spk] = torch.randn(1, args.embed_dim, device=device)
                for k in embeddings: embeddings[k][spk] = []
            key = speaker_keys[spk]

            # ── 1. Anonymize ─────────────────────────────────────────────────
            xa, cond = rano_model.anonymize(mel, key)

            if vocoder_mode == "hifigan":
                xa_clamped = xa.clamp(MEL_MIN, MEL_MAX)
                anon_wav_t = processor.mel_to_wav(xa_clamped.cpu())
                anon_wav = anon_wav_t.squeeze(0).numpy()
            else:
                # Clamp to the valid log-mel range before the vocoder's exp():
                # an out-of-range mel overflows to Inf and griffinlim then dies with
                # "Audio buffer is not finite everywhere". Mirrors the hifigan path
                # above. restore() below still gets the RAW xa, so invertibility holds.
                anon_wav = pseudo_inverse_vocoder(
                    xa.clamp(MEL_MIN, MEL_MAX), orig_wav=wav_16k, sr=proc_sr,
                    n_fft=1024, hop_length=256, n_mels=80,
                    mode=anon_mode,
                    pitch_shift_semitones=args.pitch_shift,
                    griffinlim_iters=128,
                    apply_post_filter=args.post_filter,
                )
            anon_wav = peak_normalize(anon_wav)

            # ── 2. Restore ───────────────────────────────────────────────────
            xr = rano_model.restore(xa, key)

            if vocoder_mode == "hifigan":
                xr_clamped = xr.clamp(MEL_MIN, MEL_MAX)
                restored_wav_t = processor.mel_to_wav(xr_clamped.cpu())
                restored_wav = restored_wav_t.squeeze(0).numpy()
            else:
                # phase_save gives maximum Sim_spk/MCD quality for pseudo-inverse
                restored_wav = pseudo_inverse_vocoder(
                    xr.clamp(MEL_MIN, MEL_MAX), orig_wav=wav_16k, sr=proc_sr,
                    n_fft=1024, hop_length=256, n_mels=80,
                    mode="phase_save",
                    griffinlim_iters=128,
                    apply_post_filter=args.post_filter,
                )
            restored_wav = peak_normalize(restored_wav)

            # ── 3. Save audio ────────────────────────────────────────────────
            spk_out_dir = save_audio_dir / spk
            spk_out_dir.mkdir(parents=True, exist_ok=True)
            base_name = path.stem
            sf.write(str(spk_out_dir / f"{base_name}_orig.wav"),     peak_normalize(wav_16k), proc_sr)
            sf.write(str(spk_out_dir / f"{base_name}_anon.wav"),     anon_wav,                proc_sr)
            sf.write(str(spk_out_dir / f"{base_name}_restored.wav"), restored_wav,            proc_sr)

            # ── 4. ASV embeddings ────────────────────────────────────────────
            def to_tensor(w: np.ndarray) -> torch.Tensor:
                return torch.tensor(w).float().to(device)

            emb_o = extract_asv_emb(to_tensor(wav_16k))
            emb_a = extract_asv_emb(to_tensor(anon_wav))
            emb_r = extract_asv_emb(to_tensor(restored_wav))

            embeddings["orig"][spk].append(emb_o)
            embeddings["anon"][spk].append(emb_a)
            embeddings["restored"][spk].append(emb_r)

            # ── 5. Metrics ───────────────────────────────────────────────────
            metrics["sim_spk_restored"].append(cosine_sim(emb_o, emb_r) * 100.0)
            metrics["gvd"].append(calculate_gvd(mel, xa))
            metrics["mcd_restored"].append(
                calculate_mcd(wav_16k, restored_wav, sr_ref=proc_sr, sr_deg=proc_sr)
            )

            f0_orig = extract_f0(wav_16k, proc_sr)
            f0_anon = extract_f0(anon_wav, proc_sr)
            metrics["rho_f0"].append(calculate_rho_f0(f0_orig, f0_anon))

            # WER against LibriSpeech ground truth
            if whisper_model is not None:
                ref_text = transcripts.get(path.stem, "")
                if ref_text:
                    hyp_text = transcribe(whisper_model, anon_wav, proc_sr)
                    metrics["wer"].append(calculate_wer(ref_text, hyp_text) * 100.0)

    # ── TRUE EER ─────────────────────────────────────────────────────────────
    print("\nCalculating True ASV EER...")
    y_true, y_score = [], []
    for spk_a, embs_a in embeddings["anon"].items():
        for emb_a in embs_a:
            for spk_o, embs_o in embeddings["orig"].items():
                for emb_o in embs_o:
                    y_score.append(cosine_sim(emb_a, emb_o))
                    y_true.append(1 if spk_a == spk_o else 0)

    true_eer = calculate_eer(y_true, y_score)
    elapsed = time.time() - start_time

    result = {k: safe_mean(v) for k, v in metrics.items()}
    result["eer"] = true_eer
    result["vocoder"] = vocoder_mode
    result["anon_mode"] = anon_mode
    result["post_filter"] = args.post_filter

    wer_str = f"{result['wer']:>5.2f}" if args.compute_wer else "  N/A"

    print(f"\nRANO Evaluation Results — Table I Format (§IV-B)")
    print(f" Speakers: {len(speaker_keys)}  |  Utterances: {len(files)}  |  Time: {elapsed:.1f}s")
    print(f" Vocoder: {vocoder_mode}  |  Anon Mode: {anon_mode}")
    print("=" * 70)
    print(f" Metric                     Value   Direction  Paper Ref")
    print("-" * 65)
    print(f" True EER (%)               {result['eer']:>5.2f}    ^ better  Sec IV-B.1")
    print(f" WER (%)                    {wer_str:>5}    v better  Sec IV-B.1")
    print(f" GVD (dB)                   {result['gvd']:>5.2f}   >=0 ideal  Sec IV-B.1, [41]")
    print(f" rho_f0                     {result['rho_f0']:>5.3f}    ^ better  Sec IV-B.1")
    print("-" * 65)
    print(f" Restoration Metrics — Sec IV-D (correct key)")
    print(f" Sim_spk (%)                {result['sim_spk_restored']:>5.2f}    ^ better  Table III")
    print(f" MCD (dB)                   {result['mcd_restored']:>5.2f}    v better  Table III")
    print("=" * 70)

    out_path = f"eval_results_{vocoder_mode}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RANO evaluation (§IV-B)")
    p.add_argument("--test_dir",          required=True)
    p.add_argument("--acg_checkpoint",    required=True)
    p.add_argument("--anonymizer_ckpt",   required=True)
    p.add_argument("--embed_dim",         type=int,   default=256)
    p.add_argument("--num_cinn_blocks",   type=int,   default=12)
    p.add_argument("--compute_wer",       action="store_true")
    p.add_argument("--whisper_model",     type=str, default="large",
                   help="Whisper size for WER. Use 'medium' or 'small' on an 8GB GPU "
                        "(large needs ~6GB).")
    p.add_argument("--save_audio_dir",    type=str,   default="eval_math_vocoder_v2")

    # Vocoder selection (paper §IV-A uses HiFi-GAN)
    p.add_argument(
        "--vocoder",
        type=str,
        default="pseudoinverse",
        choices=["hifigan", "pseudoinverse"],
        help=(
            "hifigan: neural vocoder matching paper §IV-A (default). "
            "pseudoinverse: Griffin-Lim / perturbed-phase mathematical vocoder."
        ),
    )
    # Pseudo-inverse vocoder controls (only used when --vocoder=pseudoinverse)
    p.add_argument(
        "--anon_mode",
        type=str,
        default="perturbed_phase",
        choices=["griffinlim", "perturbed_phase", "phase_save"],
        help=(
            "Only for --vocoder=pseudoinverse. "
            "griffinlim: no phase leak, worst WER/rho_f0. "
            "perturbed_phase: better WER/rho_f0, but EER is inflated by the pitch "
            "shift (signal-processing anonymizer), NOT the cINN. "
            "phase_save: anonymized magnitude + TRUE original phase, no pitch shift "
            "-> isolates the MODEL's own anonymization (honest model-only EER; "
            "expect low EER if the cINN is a passthrough)."
        ),
    )
    p.add_argument(
        "--pitch_shift",
        type=float,
        default=3.5,
        help="Semitones to shift phase for perturbed_phase mode (default 3.5).",
    )
    p.add_argument(
        "--post_filter",
        action="store_true",
        default=False,
        help="Apply spectral post-filter. Default OFF — it was found to DEGRADE "
             "speaker identity badly (restoration Sim_spk 91.6%% -> 52%%) and inflate "
             "MCD. Only enable for ablation.",
    )
    p.add_argument(
        "--no_post_filter",
        dest="post_filter",
        action="store_false",
        help="Disable spectral post-filter.",
    )

    args = p.parse_args()
    evaluate(args)