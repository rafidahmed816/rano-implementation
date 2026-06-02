"""Evaluate Rano: EER, WER, GVD, pitch correlation, restoration quality (§IV-B, Table I).

Paper reference: "Rano: Restorable Speaker Anonymization via Conditional Invertible Neural Network"
All metrics follow §IV-B (Privacy Evaluation) and Table I exactly.

VALIDATION: The evaluation ASV (ECAPA-TDNN from SpeechBrain [40]) is completely
separate from the training ASV (AdaIN-VC speaker encoder [15]). This ensures
fairness of comparison per §IV-B.1: "the ASVs for training and evaluation are
completely different for the fairness of comparison."

Usage:
  python evaluate.py \
    --acg_checkpoint checkpoints/acg/acg_best.pt \
    --anonymizer_ckpt checkpoints/rano/anonymizer_final.pt \
    --test_dir data/test/

  # With WER (requires: pip install openai-whisper):
  python evaluate.py \
    --acg_checkpoint checkpoints/acg/acg_best.pt \
    --anonymizer_ckpt checkpoints/rano/anonymizer_final.pt \
    --test_dir data/test/ --compute_wer

  # With restoration quality metrics:
  python evaluate.py \
    --acg_checkpoint checkpoints/acg/acg_best.pt \
    --anonymizer_ckpt checkpoints/rano/anonymizer_final.pt \
    --test_dir data/test/ --eval_restoration

  # Full evaluation (all metrics):
  python evaluate.py \
    --acg_checkpoint checkpoints/acg/acg_best.pt \
    --anonymizer_ckpt checkpoints/rano/anonymizer_final.pt \
    --test_dir data/test/ --compute_wer --eval_restoration
    
# or .venv/bin/python3 evaluate.py ....... ( was causing in my system. This fixed)
"""

import argparse
import json
import time
import librosa
import numpy as np

# Force librosa lazy_loader to evaluate before speechbrain is loaded
_ = librosa.pyin(np.zeros(1000), fmin=50, fmax=200)

from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from tqdm import tqdm

from model import Rano
from audio import MelProcessor
from speaker_encoder import ECAPATDNNEncoder

# VALIDATION: ECAPA-TDNN operates at 16 kHz (SpeechBrain default).
# This is different from Rano's 22050 Hz — resampling is required.
_ECAPA_SR = 16000


# ---------------------------------------------------------------------------
# Metrics — §IV-B (Privacy Evaluation)
# ---------------------------------------------------------------------------

def compute_eer(scores: list[float], labels: list[int]) -> float:
    """Equal Error Rate (%) — §IV-B.1.

    VALIDATION (§IV-B.1): "an equal error rate (EER) is obtained when the
    false acceptance rate equals to false rejection rate. Higher EER indicates
    anonymized speech poses a greater challenge to the speaker verification
    model, i.e. better anonymization performance."

    Args:
        scores: Cosine similarity scores between enrollment and test embeddings.
        labels: Binary labels (1 = same speaker, 0 = different speaker).

    Returns:
        EER as percentage. Higher = better anonymization (Table I: ↑).
    """
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer) * 100


def compute_gvd(orig_mean: dict, anon_mean: dict) -> float:
    """Gain of Voice Distinctiveness in dB — §IV-B.1, Noé et al. [41].

    VALIDATION (§IV-B.1): "Gain of voice distinctiveness (GVD) measures the
    speaker distinctiveness before and after the anonymization process. It is
    obtained from two matrices that respectively contain speaker distinctiveness
    in original and anonymized spaces [41]. When GVD values 0 dB, the
    distinctiveness between original speakers is preserved in anonymous speakers.
    Increments in distinctiveness result in gains above 0, while degradation
    results in gains below 0. An ideal model owns GVD values close to 0 or above."

    GVD = 10 * log10(D_anon / D_orig), where D is mean pairwise cosine distance.

    Args:
        orig_mean: {speaker_id: mean_embedding} for original speech.
        anon_mean: {speaker_id: mean_embedding} for anonymized speech.

    Returns:
        GVD in dB. Close to 0 or positive = good (Table I).
    """
    spks = list(orig_mean.keys())
    if len(spks) < 2:
        return 0.0

    def mean_cosine_dist(embs: dict) -> float:
        """Mean pairwise cosine distance across all speaker pairs."""
        vecs = F.normalize(torch.stack(list(embs.values())), dim=-1)
        dists = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                # Cosine distance = 1 - cosine_similarity
                dists.append((1 - (vecs[i] * vecs[j]).sum().item()))
        return float(np.mean(dists)) if dists else 0.0

    d_orig = mean_cosine_dist(orig_mean)
    d_anon = mean_cosine_dist(anon_mean)

    if d_orig < 1e-9:
        return 0.0
    return float(10 * np.log10(d_anon / d_orig))


def extract_f0(wav_np: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Extract fundamental frequency using librosa pyin.

    VALIDATION: Used for pitch correlation ρf0 computation (§IV-B.1).
    """
    import librosa
    f0, _, _ = librosa.pyin(
        wav_np.astype(np.float64),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    return f0


def compute_pitch_correlation(f0_a: np.ndarray, f0_b: np.ndarray) -> float:
    """Pearson correlation between original and anonymized pitch — §IV-B.1.

    VALIDATION (§IV-B.1): "Pitch correlation measures the consistency of the
    fundamental frequency before and after the anonymization process. It is
    calculated via the Pearson correlation between two pitch sequences. A higher
    pitch correlation indicates better preservation of the pitch information."

    Returns:
        ρf0 ∈ [-1, 1]. Higher = better preservation (Table I: ↑).
    """
    n = min(len(f0_a), len(f0_b))
    f0_a, f0_b = f0_a[:n], f0_b[:n]
    # Only compare voiced frames (non-NaN)
    voiced = ~(np.isnan(f0_a) | np.isnan(f0_b))
    if voiced.sum() < 2:
        return 0.0
    return float(np.corrcoef(f0_a[voiced], f0_b[voiced])[0, 1])


def compute_wer_edit_distance(reference: str, hypothesis: str) -> float:
    """Token-level WER via edit distance — §IV-B.1.

    VALIDATION (§IV-B.1): "WER with a smaller value indicates more complete
    content retention." Lower = better (Table I: ↓).

    Returns:
        WER as percentage.
    """
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if len(ref) == 0:
        return 100.0 if len(hyp) > 0 else 0.0
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        d[i, 0] = i
    for j in range(len(hyp) + 1):
        d[0, j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(ref), len(hyp)] / len(ref)) * 100


def compute_mcd(mel_a: torch.Tensor, mel_b: torch.Tensor) -> float:
    """Mel-Cepstral Distortion between two mel-spectrograms in dB — §IV-D.

    VALIDATION (§IV-D, Table III): "Mel-cepstral distortion (MCD) which measures
    the distance between Mel frequency cepstral coefficient (MFCC)."

    Returns:
        MCD in dB. Lower = more similar.
    """
    diff = (mel_a - mel_b) ** 2
    return float((10 / np.log(10)) * torch.sqrt(2 * diff.mean()).item())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(args, device):
    """Load Rano model with separate ACG + Anonymizer checkpoints.

    VALIDATION: Supports both separate checkpoints (acg_checkpoint + anonymizer_ckpt)
    and combined checkpoint (rano_checkpoint) formats.
    """
    model = Rano(
        embed_dim=args.embed_dim,
        num_cinn_blocks=args.num_cinn_blocks,
        num_acg_blocks=args.num_acg_blocks,
    ).to(device)

    # Try combined checkpoint first
    if hasattr(args, 'rano_checkpoint') and args.rano_checkpoint:
        rano_path = Path(args.rano_checkpoint)
        if rano_path.exists():
            state_dict = torch.load(rano_path, map_location=device)
            # Handle torch.compile() wrapping
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Loaded combined Rano from {rano_path}")
            model.eval()
            return model

    # Load ACG (Stage 1) — §III-E: "ACG is pre-trained"
    acg_path = Path(args.acg_checkpoint)
    if acg_path.exists():
        acg_state = torch.load(acg_path, map_location=device)
        # Handle torch.compile() wrapping
        if any(k.startswith("_orig_mod.") for k in acg_state.keys()):
            acg_state = {k.replace("_orig_mod.", ""): v for k, v in acg_state.items()}
        model.acg.load_state_dict(acg_state)
        print(f"Loaded ACG from {acg_path}")
    else:
        print(f"[WARN] ACG not found: {acg_path} — using random weights")

    # Load Anonymizer (Stage 2)
    anon_path = Path(args.anonymizer_ckpt)
    if anon_path.exists():
        anon_state = torch.load(anon_path, map_location=device)
        # Handle torch.compile() wrapping
        if any(k.startswith("_orig_mod.") for k in anon_state.keys()):
            anon_state = {k.replace("_orig_mod.", ""): v for k, v in anon_state.items()}
        model.anonymizer.load_state_dict(anon_state)
        print(f"Loaded Anonymizer from {anon_path}")
    else:
        raise FileNotFoundError(f"Anonymizer checkpoint not found: {anon_path}")

    model.eval()
    return model


def _load_whisper(model_name: str, device: torch.device):
    """Lazy-load Whisper model for WER computation.

    VALIDATION (§IV-B.1): "We employ Whisper [42] (large) speech recognition
    model to measure WER in our experiment."

    Requires: pip install openai-whisper
    Also requires ffmpeg system dependency.
    """
    try:
        import whisper
        print(f"Loading Whisper '{model_name}' for WER evaluation...")
        model = whisper.load_model(model_name, device=device)
        print(f"[OK] Whisper '{model_name}' loaded successfully")
        return model
    except ImportError:
        print("[WARN] openai-whisper not installed. Install with: pip install openai-whisper")
        print("       WER metric will be skipped.")
        return None
    except Exception as e:
        print(f"[WARN] Failed to load Whisper model: {e}")
        print("       WER metric will be skipped.")
        return None


def _transcribe_wav(whisper_model, wav_np: np.ndarray, sr: int) -> str:
    """Transcribe waveform using Whisper.

    VALIDATION: Whisper expects 16kHz audio internally — it handles resampling.
    We pass the audio at its native sample rate; Whisper pads/trims as needed.
    """
    import whisper

    # Whisper expects float32 numpy array at 16kHz
    if sr != 16000:
        wav_tensor = torch.from_numpy(wav_np).float()
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
        wav_np = wav_tensor.numpy()

    # Pad or trim to 30 seconds as Whisper expects
    audio = whisper.pad_or_trim(wav_np.astype(np.float32))
    mel = whisper.log_mel_spectrogram(audio).to(next(whisper_model.parameters()).device)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    result = whisper.decode(whisper_model, mel, options)
    return result.text


# ---------------------------------------------------------------------------
# Main evaluation loop — §IV-B, §IV-C
# ---------------------------------------------------------------------------

def evaluate(args):
    """Run full evaluation producing Table I metrics.

    VALIDATION (§IV-C): "The measurement of Rano is conducted in speaker-level,
    that is, we provide each speaker with a key for all of his/her utterances to
    ensure each original speaker is related to only one anonymous speaker during
    the measurement."
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MelProcessor(device=device)
    model = _load_model(args, device)

    # VALIDATION (§IV-B.1): "A pre-trained ASV from SpeechBrain [40] is employed
    # to perform speaker verification by directly extracting ECAPA-TDNN embeddings
    # from waveform." This is DIFFERENT from training ASV (AdaIN-VC).
    sb_device = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"
    eval_asv = ECAPATDNNEncoder(device=sb_device).to(device)

    # VALIDATION (§IV-B.1): Load Whisper for WER if requested
    whisper_model = None
    if args.compute_wer:
        whisper_model = _load_whisper(args.whisper_model, device)

    # Per-speaker key storage — speaker-level anonymization (§IV-C)
    speaker_keys: dict[str, torch.Tensor] = {}

    # Accumulators for each metric
    orig_embs_per_spk: dict[str, list] = defaultdict(list)
    anon_embs_per_spk: dict[str, list] = defaultdict(list)
    pitch_corrs: list[float] = []
    wer_scores: list[float] = []
    restoration_sim: list[float] = []
    restoration_mcd: list[float] = []

    # Discover test files
    test_files = sorted(
        f for ext in ("*.wav", "*.flac")
        for f in Path(args.test_dir).rglob(ext)
    )
    if not test_files:
        raise FileNotFoundError(f"No wav/flac files found in {args.test_dir}")

    # Optionally limit number of utterances for quick evaluation
    if args.max_utterances and args.max_utterances > 0:
        test_files = test_files[:args.max_utterances]
        print(f"[INFO] Limiting evaluation to {args.max_utterances} utterances")

    print(f"\n{'='*60}")
    print(f"Evaluation Configuration")
    print(f"  Device:            {device}")
    print(f"  Test files:        {len(test_files)}")
    print(f"  Compute WER:       {args.compute_wer}")
    print(f"  Eval restoration:  {args.eval_restoration}")
    print(f"  Whisper model:     {args.whisper_model}")
    print(f"{'='*60}\n")

    start_time = time.time()

    with torch.no_grad():
        for path in tqdm(test_files, desc="Evaluating"):
            # VALIDATION: Speaker ID from parent directory name
            # Expected structure: test_dir/speaker_id/utterance.wav
            spk = path.parent.name

            # Load and preprocess audio
            wav_np, sr = sf.read(str(path))
            wav = torch.from_numpy(wav_np).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.transpose(0, 1)
            wav = processor.resample(wav.mean(0), sr)  # mono, 22050 Hz

            mel = processor.wav_to_mel(wav).to(device)  # (1, 80, T_mel)

            # VALIDATION (§IV-C): Speaker-level anonymization — same key per speaker
            if spk not in speaker_keys:
                speaker_keys[spk] = torch.randn(1, args.embed_dim, device=device)

            # Anonymize: xa = f(x; c, θ) where c = ACG(key) — Eq. 1
            xa, cond = model.anonymize(mel, speaker_keys[spk])

            # Vocode anonymized mel → waveform via HiFi-GAN (§IV-A.2)
            xa_wav_22k = processor.mel_to_wav(xa.cpu()).squeeze(0)  # (T_wav,)

            # VALIDATION (§IV-B.1): Resample to 16 kHz for ECAPA-TDNN evaluation
            wav_16k = torchaudio.functional.resample(wav, processor.sample_rate, _ECAPA_SR)
            xa_wav_16k = torchaudio.functional.resample(xa_wav_22k, processor.sample_rate, _ECAPA_SR)

            # VALIDATION (§IV-B.1): Extract ECAPA-TDNN embeddings from waveform
            # "directly extracting ECAPA-TDNN embeddings from waveform"
            orig_emb = eval_asv(wav_16k.unsqueeze(0).to(device)).squeeze(0).cpu()
            anon_emb = eval_asv(xa_wav_16k.unsqueeze(0).to(device)).squeeze(0).cpu()

            orig_embs_per_spk[spk].append(orig_emb)
            anon_embs_per_spk[spk].append(anon_emb)

            # --- Pitch Correlation ρf0 (§IV-B.1) ---
            f0_orig = extract_f0(wav.numpy(), processor.sample_rate, processor.hop_length)
            f0_anon = extract_f0(xa_wav_22k.numpy(), processor.sample_rate, processor.hop_length)
            pitch_corrs.append(compute_pitch_correlation(f0_orig, f0_anon))

            # --- WER via Whisper (§IV-B.1) ---
            if whisper_model is not None:
                try:
                    ref_text = _transcribe_wav(whisper_model, wav.numpy(), processor.sample_rate)
                    hyp_text = _transcribe_wav(whisper_model, xa_wav_22k.numpy(), processor.sample_rate)
                    if ref_text.strip():
                        wer_scores.append(compute_wer_edit_distance(ref_text, hyp_text))
                except Exception as e:
                    tqdm.write(f"  [WARN] Whisper failed on {path.name}: {e}")

            # --- Restoration quality (§IV-D) ---
            if args.eval_restoration:
                xr = model.restore(xa, speaker_keys[spk])
                # MCD between original and restored mel
                restoration_mcd.append(compute_mcd(mel.squeeze().cpu(), xr.squeeze().cpu()))
                # Speaker similarity between original and restored (via vocoder)
                xr_wav_22k = processor.mel_to_wav(xr.cpu()).squeeze(0)
                xr_wav_16k = torchaudio.functional.resample(
                    xr_wav_22k, processor.sample_rate, _ECAPA_SR
                )
                xr_emb = eval_asv(xr_wav_16k.unsqueeze(0).to(device)).squeeze(0).cpu()
                sim = F.cosine_similarity(orig_emb.unsqueeze(0), xr_emb.unsqueeze(0)).item() * 100
                restoration_sim.append(sim)

    elapsed = time.time() - start_time

    # -----------------------------------------------------------------------
    # Compute aggregate metrics
    # -----------------------------------------------------------------------

    # --- EER (§IV-B.1) ---
    # VALIDATION: Enroll = mean original embedding per speaker.
    # Test = per-utterance anonymized embeddings.
    # Label = 1 if enroll speaker == test speaker, 0 otherwise.
    enroll = {s: torch.stack(v).mean(0) for s, v in orig_embs_per_spk.items()}
    eer_scores, eer_labels = [], []
    for enroll_spk, enroll_emb in enroll.items():
        for test_spk, utt_embs in anon_embs_per_spk.items():
            for utt_emb in utt_embs:
                score = F.cosine_similarity(
                    enroll_emb.unsqueeze(0), utt_emb.unsqueeze(0)
                ).item()
                eer_scores.append(score)
                eer_labels.append(1 if enroll_spk == test_spk else 0)

    eer = (
        compute_eer(eer_scores, eer_labels)
        if len(set(eer_labels)) == 2
        else float("nan")
    )

    # --- GVD (§IV-B.1, Noé et al. [41]) ---
    orig_mean = {s: torch.stack(v).mean(0) for s, v in orig_embs_per_spk.items()}
    anon_mean = {s: torch.stack(v).mean(0) for s, v in anon_embs_per_spk.items()}
    gvd = compute_gvd(orig_mean, anon_mean)

    # --- Aggregate scalar metrics ---
    rho_f0 = float(np.mean(pitch_corrs)) if pitch_corrs else 0.0
    wer = float(np.mean(wer_scores)) if wer_scores else float("nan")
    mean_rest_sim = float(np.mean(restoration_sim)) if restoration_sim else float("nan")
    mean_rest_mcd = float(np.mean(restoration_mcd)) if restoration_mcd else float("nan")

    # -----------------------------------------------------------------------
    # Print results in Table I format (§IV, Table I)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RANO Evaluation Results — Table I Format (§IV-B)")
    print(f"  Speakers: {len(enroll)}  |  Utterances: {len(test_files)}  |  Time: {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Value':>12}  {'Direction':>10}  {'Paper Ref'}")
    print(f"  {'-'*65}")
    print(f"  {'EER (%)':<20} {eer:>12.2f}  {'^ better':>10}  Sec IV-B.1")
    if not np.isnan(wer):
        print(f"  {'WER (%)':<20} {wer:>12.2f}  {'v better':>10}  Sec IV-B.1 (Whisper)")
    else:
        print(f"  {'WER (%)':<20} {'N/A':>12}  {'v better':>10}  Sec IV-B.1 (skipped)")
    print(f"  {'GVD (dB)':<20} {gvd:>12.2f}  {'>=0 ideal':>10}  Sec IV-B.1, [41]")
    print(f"  {'rho_f0':<20} {rho_f0:>12.3f}  {'^ better':>10}  Sec IV-B.1")

    if args.eval_restoration:
        print(f"  {'-'*65}")
        print(f"  {'Restoration Metrics — Sec IV-D (correct key)':}")
        print(f"  {'Sim_spk (%)':<20} {mean_rest_sim:>12.2f}  {'^ better':>10}  Table III")
        print(f"  {'MCD (dB)':<20} {mean_rest_mcd:>12.2f}  {'v better':>10}  Table III")

    print(f"{'='*70}")

    # --- Paper Table I comparison ---
    print(f"\n  Reference values from Table I (paper):")
    print(f"  {'Method':<18} {'EER%':>7} {'WER%':>7} {'GVD':>8} {'ρf0':>6}")
    print(f"  {'-'*50}")
    print(f"  {'Ground-Truth':<18} {'3.20':>7} {'6.58':>7} {'-':>8} {'-':>6}")
    print(f"  {'Rano (paper)':<18} {'47.81':>7} {'11.91':>7} {'0.39':>8} {'0.80':>6}")
    print(f"  {'Rano w/o ACG':<18} {'32.04':>7} {'11.97':>7} {'-4.33':>8} {'0.77':>6}")
    print()

    # -----------------------------------------------------------------------
    # Save results to JSON
    # -----------------------------------------------------------------------
    results = {
        "method": "Rano",
        "num_speakers": len(enroll),
        "num_utterances": len(test_files),
        "elapsed_seconds": round(elapsed, 1),
        "metrics": {
            "EER_pct": round(eer, 2),
            "WER_pct": round(wer, 2) if not np.isnan(wer) else None,
            "GVD_dB": round(gvd, 2),
            "rho_f0": round(rho_f0, 3),
        },
        "paper_reference": {
            "EER_pct": 47.81,
            "WER_pct": 11.91,
            "GVD_dB": 0.39,
            "rho_f0": 0.80,
        },
    }
    if args.eval_restoration:
        results["restoration"] = {
            "Sim_spk_pct": round(mean_rest_sim, 2) if not np.isnan(mean_rest_sim) else None,
            "MCD_dB": round(mean_rest_mcd, 2) if not np.isnan(mean_rest_mcd) else None,
        }

    output_json = Path(args.output_json) if args.output_json else Path(args.test_dir) / "eval_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate Rano speaker anonymization (§IV-B, Table I).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model checkpoints
    p.add_argument("--acg_checkpoint", default="checkpoints/acg/acg_best.pt",
                    help="Path to ACG (Stage 1) checkpoint.")
    p.add_argument("--anonymizer_ckpt", required=True,
                    help="Path to Anonymizer (Stage 2) checkpoint.")
    p.add_argument("--rano_checkpoint", default=None,
                    help="Path to combined rano_final.pt (overrides acg + anonymizer).")

    # Data
    p.add_argument("--test_dir", required=True,
                    help="Test directory with speaker_id/utterance.wav structure.")

    # Model architecture (must match training)
    p.add_argument("--embed_dim", type=int, default=256,
                    help="Speaker embedding dimension (§III: 256).")
    p.add_argument("--num_cinn_blocks", type=int, default=12,
                    help="Number of cINN blocks in Anonymizer (§III: 12).")
    p.add_argument("--num_acg_blocks", type=int, default=8,
                    help="Number of INN blocks in ACG (§III: 8).")

    # Metric flags
    p.add_argument("--compute_wer", action="store_true",
                    help="Compute WER using Whisper (requires: pip install openai-whisper).")
    p.add_argument("--whisper_model", default="large",
                    help="Whisper model size: tiny/base/small/medium/large (§IV-B.1: large).")
    p.add_argument("--eval_restoration", action="store_true",
                    help="Evaluate restoration quality (Sim_spk, MCD — §IV-D).")

    # Output
    p.add_argument("--output_json", default=None,
                    help="Path to save JSON results (default: test_dir/eval_results.json).")
    p.add_argument("--max_utterances", type=int, default=None,
                    help="Limit evaluation to N utterances (for quick testing).")

    args = p.parse_args()
    evaluate(args)