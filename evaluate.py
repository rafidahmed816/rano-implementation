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


# ============================================================
# AUDIO UTILS
# ============================================================

def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x.mean(axis=1)
    return x


def peak_normalize(wav: np.ndarray) -> np.ndarray:
    """Forces audio to exactly 100% maximum volume for consistent scaling and MCD."""
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
    if n < 10:
        return float("nan")
    a, b = f0_ref[:n], f0_deg[:n]
    valid = (a > 0) & (b > 0) & ~np.isnan(a) & ~np.isnan(b)
    if np.sum(valid) < 5:
        return float("nan")
    a_v, b_v = a[valid], b[valid]
    if np.std(a_v) < 1e-6 or np.std(b_v) < 1e-6:
        return 0.0
    return float(np.corrcoef(a_v, b_v)[0, 1])


def calculate_gvd(mel_ref: torch.Tensor, mel_deg: torch.Tensor) -> float:
    var_ref = torch.var(mel_ref.squeeze(), dim=-1)
    var_deg = torch.var(mel_deg.squeeze(), dim=-1)
    gv_ref_db = 10 * torch.log10(var_ref + 1e-8)
    gv_deg_db = 10 * torch.log10(var_deg + 1e-8)
    gvd = torch.sqrt(torch.mean((gv_ref_db - gv_deg_db) ** 2))
    return float(gvd.item())


def calculate_mcd(
    wav_ref: np.ndarray,
    wav_deg: np.ndarray,
    sr_ref: int,
    sr_deg: int,
    target_sr: int = MCD_SR,
) -> float:
    wav_ref = resample_to(np.asarray(wav_ref, dtype=np.float32), sr_ref, target_sr)
    wav_deg = resample_to(np.asarray(wav_deg, dtype=np.float32), sr_deg, target_sr)

    wav_ref = peak_normalize(wav_ref)
    wav_deg = peak_normalize(wav_deg)

    for name, w in [("ref", wav_ref), ("deg", wav_deg)]:
        if not np.all(np.isfinite(w)):
            return float("nan")

    mfcc_ref = librosa.feature.mfcc(y=wav_ref, sr=target_sr, n_mfcc=13)[1:, :]
    mfcc_deg = librosa.feature.mfcc(y=wav_deg, sr=target_sr, n_mfcc=13)[1:, :]

    if mfcc_ref.shape[1] == 0 or mfcc_deg.shape[1] == 0:
        return float("nan")

    # Dynamic Time Warping (DTW) to properly align vocoder delays
    D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_deg, metric='euclidean')
    mean_dist = D[-1, -1] / len(wp)
    
    return float((10.0 * np.sqrt(2.0) / np.log(10.0)) * mean_dist)


def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
    return float(np.dot(emb1, emb2) / denom)


def calculate_eer(y_true: list, y_score: list) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
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
        wav = torchaudio.functional.resample(
            torch.tensor(wav), sr, 16000
        ).numpy()
    wav = peak_normalize(wav)
    audio = whisper.pad_or_trim(wav)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
    return whisper.decode(whisper_model, mel, opts).text


def calculate_wer(ref: str, hyp: str) -> float:
    r, h = ref.lower().split(), hyp.lower().split()
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
    model = Rano(
        embed_dim=args.embed_dim,
        num_cinn_blocks=args.num_cinn_blocks,
    ).to(device)

    def load_weights(path: str, module: torch.nn.Module, name: str):
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        module.load_state_dict(ckpt, strict=False)
        print(f"  Loaded {name} from {path}")

    load_weights(args.acg_checkpoint,  model.acg,        "ACG")
    load_weights(args.anonymizer_ckpt, model.anonymizer,  "Anonymizer")
    model.eval()
    return model


def load_asv(device: torch.device):
    print("Loading SpeechBrain ECAPA-TDNN ASV model...")

    _orig_symlink = os.symlink
    def _patched_symlink(src, dst, target_is_directory=False, *a, **kw):
        try:
            _orig_symlink(src, dst, target_is_directory, *a, **kw)
        except OSError:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    os.symlink = _patched_symlink

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    asv = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="tmp_speechbrain_asv",
        run_opts={"device": str(device)},
    )
    os.symlink = _orig_symlink

    def extract_emb(wav_tensor: torch.Tensor) -> np.ndarray:
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        with torch.no_grad():
            return asv.encode_batch(wav_tensor).squeeze().cpu().numpy()

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

    save_audio_dir = Path(args.save_audio_dir)
    save_audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generated audio will be saved to: {save_audio_dir}")

    # RE-ENABLE NEURAL VOCODER (Should trigger SpeechT5 based on updated audio.py)
    # Forced 16kHz to match LibriSpeech and SpeechT5 native resolution
    processor = MelProcessor(device=device, use_hifigan=True, sample_rate=16000)

    proc_sr: int = getattr(processor, "sample_rate",
                  getattr(processor, "sr",
                  getattr(processor, "target_sr", 16000)))
    print(f"MelProcessor output sample rate: {proc_sr} Hz")

    rano_model = load_rano(args, device)
    extract_asv_emb = load_asv(device)

    whisper_model = None
    if args.compute_wer:
        import whisper
        whisper_model = whisper.load_model("large-v2").to(device)
        print("Whisper loaded.")

    files = (
        list(Path(args.test_dir).rglob("*.flac"))
        + list(Path(args.test_dir).rglob("*.wav"))
    )
    if not files:
        raise RuntimeError(f"No .flac or .wav files found in: {args.test_dir}")
    print(f"Found {len(files)} utterances.")

    speaker_keys: dict[str, torch.Tensor] = {}

    embeddings = {"orig": {}, "anon": {}, "restored": {}}
    metrics: dict[str, list] = {
        "gvd": [],
        "rho_f0": [],
        "wer": [],
        "sim_spk_restored": [],
        "mcd_restored": [],
    }

    with torch.no_grad():
        for path in tqdm(files, desc="Processing"):
            spk = path.parent.name
            wav_np, sr = sf.read(str(path))
            wav_np = to_mono(wav_np)

            wav_t = processor.resample(torch.tensor(wav_np).float(), sr)
            mel = processor.wav_to_mel(wav_t).to(device)  

            if spk not in speaker_keys:
                speaker_keys[spk] = torch.randn(1, args.embed_dim, device=device)
                for k in embeddings:
                    embeddings[k][spk] = []

            key = speaker_keys[spk] 

            # ── 1. Anonymize ─────────────────────────────────────────────────
            xa, cond = rano_model.anonymize(mel, key)
            
            # Vocode and peak normalize the output
            anon_wav = processor.mel_to_wav(xa.cpu()).squeeze().numpy()
            anon_wav = peak_normalize(anon_wav)

            # ── 2. Restore ───────────────────────────────────────────────────
            xr = rano_model.restore(xa, key)
            
            # Vocode and peak normalize the output
            restored_wav = processor.mel_to_wav(xr.cpu()).squeeze().numpy()
            restored_wav = peak_normalize(restored_wav)
            
            # ── 3. SAVE AUDIO TO DISK ────────────────────────────────────────
            spk_out_dir = save_audio_dir / spk
            spk_out_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = path.stem
            sf.write(str(spk_out_dir / f"{base_name}_orig.wav"), peak_normalize(wav_np), sr)
            sf.write(str(spk_out_dir / f"{base_name}_anon.wav"), anon_wav, proc_sr)
            sf.write(str(spk_out_dir / f"{base_name}_restored.wav"), restored_wav, proc_sr)

            # ── 4. True ASV embeddings (16 kHz) ──────────────────────────────
            def to_16k(wav_array: np.ndarray, source_sr: int) -> torch.Tensor:
                t = torch.tensor(wav_array).float()
                return torchaudio.functional.resample(t, source_sr, 16000).to(device)

            emb_o = extract_asv_emb(to_16k(wav_np, sr))
            emb_a = extract_asv_emb(to_16k(anon_wav, proc_sr))
            emb_r = extract_asv_emb(to_16k(restored_wav, proc_sr))

            embeddings["orig"][spk].append(emb_o)
            embeddings["anon"][spk].append(emb_a)
            embeddings["restored"][spk].append(emb_r)

            # ── 5. Metric Calculations ───────────────────────────────────────
            metrics["sim_spk_restored"].append(cosine_sim(emb_o, emb_r) * 100.0)
            metrics["gvd"].append(calculate_gvd(mel, xa))
            metrics["mcd_restored"].append(
                calculate_mcd(wav_np, restored_wav, sr_ref=sr, sr_deg=proc_sr)
            )

            wav_16k  = resample_to(wav_np,   sr,      MCD_SR)
            anon_16k = resample_to(anon_wav, proc_sr, MCD_SR)
            f0_orig = extract_f0(wav_16k,  MCD_SR)
            f0_anon = extract_f0(anon_16k, MCD_SR)
            metrics["rho_f0"].append(calculate_rho_f0(f0_orig, f0_anon))

            if whisper_model is not None:
                ref_text = transcribe(whisper_model, wav_np, sr)
                hyp_text = transcribe(whisper_model, anon_wav, proc_sr)
                if ref_text.strip():
                    metrics["wer"].append(
                        calculate_wer(ref_text, hyp_text) * 100.0
                    )

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

    wer_str = f"{result['wer']:>5.2f}" if args.compute_wer else "  N/A"

    print(f"\nRANO Evaluation Results — Table I Format (§IV-B)")
    print(f" Speakers: {len(speaker_keys)}  |  Utterances: {len(files)}  |  Time: {elapsed:.1f}s")
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

    out_path = "eval_results_true_asv.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RANO evaluation script")
    p.add_argument("--test_dir",        required=True)
    p.add_argument("--acg_checkpoint",  required=True)
    p.add_argument("--anonymizer_ckpt", required=True)
    p.add_argument("--embed_dim",       type=int, default=256)
    p.add_argument("--num_cinn_blocks", type=int, default=12)
    p.add_argument("--compute_wer",     action="store_true")
    
    # CHANGED DEFAULT DIR TO vocoderinference
    p.add_argument("--save_audio_dir",  type=str, default="vocoderinference", 
                   help="Directory to save the orig, anon, and restored audio files")
                   
    args = p.parse_args()
    evaluate(args)