"""
Diagnose HiFi-GAN vocoder issues.

Tests which vocoder loads and whether its output is usable.
Checks mel range compatibility, output energy, and spectral quality.
"""

import sys
import torch
import numpy as np
import soundfile as sf
import torchaudio
from pathlib import Path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Load a real audio file ────────────────────────────────────────
    test_dir = Path("test_audio")
    test_files = list(test_dir.rglob("*.flac"))
    if not test_files:
        print("[ERROR] No .flac files found in test_audio/")
        sys.exit(1)

    test_file = test_files[0]
    print(f"Test file: {test_file}")
    wav_np, sr = sf.read(str(test_file))
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    print(f"  Original: sr={sr}, samples={len(wav_np)}, duration={len(wav_np)/sr:.1f}s")

    # ── 2. Create mel with MelProcessor ──────────────────────────────────
    from audio import MelProcessor

    processor = MelProcessor(device=device, use_hifigan=False, sample_rate=22050)
    wav_22k = torchaudio.functional.resample(
        torch.tensor(wav_np, dtype=torch.float32), sr, 22050
    )
    mel = processor.wav_to_mel(wav_22k).to(device)  # (1, 80, T)
    print(f"\n  Mel shape: {mel.shape}")
    print(f"  Mel range: [{mel.min():.2f}, {mel.max():.2f}]")
    print(f"  Mel mean:  {mel.mean():.2f}")
    print(f"  Mel std:   {mel.std():.2f}")

    # ── 3. Test Griffin-Lim (baseline) ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GRIFFIN-LIM (baseline)")
    print(f"{'='*60}")
    gl_wav = processor.mel_to_wav_grifflim(mel.cpu())
    gl_np = gl_wav.squeeze(0).numpy()
    gl_energy = np.max(np.abs(gl_np))
    print(f"  Output samples: {len(gl_np)}")
    print(f"  Output energy:  {gl_energy:.6f}")
    print(f"  Output range:   [{gl_np.min():.4f}, {gl_np.max():.4f}]")
    sf.write("_diag_griffinlim.wav", gl_np / max(gl_energy, 1e-8), 22050)
    print(f"  Saved: _diag_griffinlim.wav")

    # ── 4. Test SpeechT5 HiFi-GAN ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SPEECHT5 HiFi-GAN (transformers)")
    print(f"{'='*60}")
    try:
        from transformers import SpeechT5HifiGan
        speecht5_model = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(device).eval()
        print(f"  Loaded OK")

        with torch.no_grad():
            # SpeechT5 expects (B, T, 80) — transpose from (1, 80, T)
            mel_t5 = mel.transpose(1, 2)  # (1, T, 80)
            print(f"  Input shape: {mel_t5.shape}")

            # Test with raw log-mel
            wav_t5 = speecht5_model(mel_t5)  # <-- REMOVED .waveform
            t5_np = wav_t5.squeeze().cpu().numpy()
            t5_energy = np.max(np.abs(t5_np))
            print(f"  [raw log-mel] Output energy: {t5_energy:.6f}")
            print(f"  [raw log-mel] Output range:  [{t5_np.min():.4f}, {t5_np.max():.4f}]")
            sf.write("_diag_speecht5_raw.wav", t5_np / max(t5_energy, 1e-8), 22050)

            # Test with exp(mel) — maybe SpeechT5 expects linear mel?
            mel_linear = torch.exp(mel).transpose(1, 2)
            wav_t5_lin = speecht5_model(mel_linear) # <-- REMOVED .waveform
            t5_lin_np = wav_t5_lin.squeeze().cpu().numpy()
            t5_lin_energy = np.max(np.abs(t5_lin_np))
            print(f"  [exp(mel)]   Output energy: {t5_lin_energy:.6f}")
            sf.write("_diag_speecht5_exp.wav", t5_lin_np / max(t5_lin_energy, 1e-8), 22050)

            # Test with clamped log-mel
            mel_clamped = mel.clamp(-11.5, 2.0).transpose(1, 2)
            wav_t5_clamp = speecht5_model(mel_clamped) # <-- REMOVED .waveform
            t5_clamp_np = wav_t5_clamp.squeeze().cpu().numpy()
            t5_clamp_energy = np.max(np.abs(t5_clamp_np))
            print(f"  [clamped]    Output energy: {t5_clamp_energy:.6f}")
            sf.write("_diag_speecht5_clamped.wav", t5_clamp_np / max(t5_clamp_energy, 1e-8), 22050)

        print(f"  Saved: _diag_speecht5_raw.wav, _diag_speecht5_exp.wav, _diag_speecht5_clamped.wav")

    except Exception as e:
        print(f"  [FAILED] {e}")

    # ── 5. Test SpeechBrain HiFi-GAN ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SPEECHBRAIN HiFi-GAN (tts-hifigan-ljspeech)")
    print(f"{'='*60}")
    try:
        from speechbrain.inference.vocoders import HIFIGAN
        from huggingface_hub import snapshot_download

        savedir = "pretrained_models/tts-hifigan-ljspeech"
        snapshot_download(
            "speechbrain/tts-hifigan-ljspeech",
            local_dir=savedir,
            local_dir_use_symlinks=False,
        )
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            sb_device = f"cuda:{idx}"
        else:
            sb_device = "cpu"
        sb_model = HIFIGAN.from_hparams(
            source=savedir, savedir=savedir,
            run_opts={"device": sb_device},
        )
        print(f"  Loaded OK")

        with torch.no_grad():
            # SpeechBrain expects (B, 80, T) — channels first
            print(f"  Input shape: {mel.shape}")

            # Test with raw log-mel
            wav_sb = sb_model.decode_batch(mel.to(device))  # (B, 1, T_wav)
            if wav_sb.dim() == 2:
                wav_sb = wav_sb.unsqueeze(1)
            sb_np = wav_sb.squeeze().cpu().numpy()
            sb_energy = np.max(np.abs(sb_np))
            print(f"  [raw log-mel] Output energy: {sb_energy:.6f}")
            print(f"  [raw log-mel] Output range:  [{sb_np.min():.4f}, {sb_np.max():.4f}]")
            sf.write("_diag_speechbrain_raw.wav", sb_np / max(sb_energy, 1e-8), 22050)

            # Test with clamped log-mel
            mel_clamped = mel.clamp(-11.5, 2.0)
            wav_sb_clamp = sb_model.decode_batch(mel_clamped.to(device))
            if wav_sb_clamp.dim() == 2:
                wav_sb_clamp = wav_sb_clamp.unsqueeze(1)
            sb_clamp_np = wav_sb_clamp.squeeze().cpu().numpy()
            sb_clamp_energy = np.max(np.abs(sb_clamp_np))
            print(f"  [clamped]    Output energy: {sb_clamp_energy:.6f}")
            sf.write("_diag_speechbrain_clamped.wav", sb_clamp_np / max(sb_clamp_energy, 1e-8), 22050)

        print(f"  Saved: _diag_speechbrain_raw.wav, _diag_speechbrain_clamped.wav")

    except Exception as e:
        print(f"  [FAILED] {e}")

    # ── 6. Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"{'='*60}")
    print(f"  Listen to the _diag_*.wav files and compare:")
    print(f"    _diag_griffinlim.wav        — Griffin-Lim baseline (always works)")
    print(f"    _diag_speecht5_raw.wav       — SpeechT5 with log-mel input")
    print(f"    _diag_speecht5_exp.wav       — SpeechT5 with linear mel input")
    print(f"    _diag_speecht5_clamped.wav   — SpeechT5 with clamped log-mel")
    print(f"    _diag_speechbrain_raw.wav    — SpeechBrain with log-mel input")
    print(f"    _diag_speechbrain_clamped.wav— SpeechBrain with clamped log-mel")
    print(f"\n  The one that sounds closest to the original is your correct vocoder.")
    print(f"\n  IMPORTANT: Your audio.py currently tries SpeechT5 FIRST.")
    print(f"  If SpeechBrain sounds better, we need to swap the order.")


if __name__ == "__main__":
    main()
