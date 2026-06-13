"""Audio ↔ Mel-spectrogram conversion utilities.

All mel-spectrograms are shape (B, 80, T) — 1D along time axis.
Uses HiFi-GAN vocoder (§IV-A) for mel→wav conversion.
"""

import torch
import torchaudio
import torchaudio.transforms as T


class HiFiGANVocoder:
    """HiFi-GAN neural vocoder for high-quality mel→wav synthesis.

    Primary: SpeechBrain tts-hifigan-ljspeech (80 mel, 22050 Hz — matches paper §IV-A).
    Fallback: transformers SpeechT5HifiGan, then Griffin-Lim.
    Expects log-mel spectrograms: (B, 80, T).
    Always returns CPU tensors.
    """

    def __init__(self, device: torch.device = None, fallback_to_griffin_lim: bool = True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.fallback_to_griffin_lim = fallback_to_griffin_lim
        self.use_grifflim_fallback = False
        self.hifigan_type = None

        # 1. SpeechBrain tts-hifigan-ljspeech — mel-compatible (22050Hz, 80mel, n_fft=1024)
        if self._try_speechbrain_hifigan():
            print(f"[OK] HiFi-GAN loaded from SpeechBrain (tts-hifigan-ljspeech)")
            return

        # 2. Transformers SpeechT5HifiGan — fallback (different mel config, may be lower quality)
        if self._try_transformers_hifigan():
            print(f"[OK] HiFi-GAN loaded from transformers (SpeechT5HifiGan)")
            return

        # 3. GRIFFIN-LIM AS FINAL FALLBACK
        if fallback_to_griffin_lim:
            print(f"[WARN] HiFi-GAN unavailable. Will use Griffin-Lim vocoding.")
            self.use_grifflim_fallback = True
        else:
            raise RuntimeError("Vocoders failed to load.")

    def _try_speechbrain_hifigan(self) -> bool:
        """SpeechBrain tts-hifigan-ljspeech: 80 mel channels, 22050 Hz."""
        try:
            from huggingface_hub import snapshot_download
            from speechbrain.inference.vocoders import HIFIGAN

            savedir = "pretrained_models/tts-hifigan-ljspeech"
            # Download without symlinks — Windows requires elevated privileges for symlinks
            snapshot_download(
                "speechbrain/tts-hifigan-ljspeech",
                local_dir=savedir,
                local_dir_use_symlinks=False,
            )
            # SpeechBrain needs "cuda:0" not "cuda"
            if self.device.type == "cuda":
                idx = self.device.index if self.device.index is not None else 0
                sb_device = f"cuda:{idx}"
            else:
                sb_device = "cpu"
            self.model = HIFIGAN.from_hparams(
                source=savedir,
                savedir=savedir,
                run_opts={"device": sb_device},
            )
            self.hifigan_type = "speechbrain"
            return True
        except Exception as e:
            print(f"[!] SpeechBrain HiFi-GAN failed: {e}")
            return False

    def _try_transformers_hifigan(self) -> bool:
        """Transformers SpeechT5HifiGan fallback."""
        try:
            from transformers import SpeechT5HifiGan
            self.model = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(
                self.device
            )
            self.model.eval()
            self.hifigan_type = "transformers"
            return True
        except Exception as e:
            print(f"[!] Transformers HiFi-GAN failed: {e}")
            return False

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, 80, T) log-mel. Returns: (B, 1, T_wav) waveform on CPU."""
        if self.use_grifflim_fallback:
            return torch.zeros(mel.shape[0], 1, mel.shape[-1] * 256)

        try:
            if self.hifigan_type == "speechbrain":
                # decode_batch passes mel directly to hifi_gan Conv1d — no internal transpose.
                # Model expects (B, 80, T) channels-first.
                wav = self.model.decode_batch(mel.to(self.device))  # (B, 1, T_wav)
                if wav.dim() == 2:
                    wav = wav.unsqueeze(1)
            else:
                # transformers SpeechT5HifiGan expects (B, T, 80)
                mel_t = mel.transpose(1, 2).to(self.device)
                wav = self.model(mel_t)  # Returns tensor directly (B, T_wav)
                if wav.dim() == 2:
                    wav = wav.unsqueeze(1)  # (B, 1, T_wav)

            return wav.cpu()
        except Exception as e:
            print(f"[ERROR] HiFi-GAN forward pass failed: {e}")
            print(f"    Mel shape: {mel.shape}, type: {self.hifigan_type}")
            self.use_grifflim_fallback = True
            return torch.zeros(mel.shape[0], 1, mel.shape[-1] * 256)


class MelProcessor:
    """
    Converts waveforms ↔ mel-spectrograms at 22050 Hz (paper's sample rate).
    All spectrograms are log-scaled internally.
    Output shape: (B, n_mels, T) — no channel dim.
    Uses HiFi-GAN for high-quality vocoding (§IV-A paper).
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: float = 8000.0,
        use_hifigan: bool = True,
        device: torch.device | None = None,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.use_hifigan = use_hifigan
        # Resolve device once at construction time, always preferring GPU.
        # Stored so the lazy vocoder load uses the same device without re-detecting.
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
        )

        # HiFi-GAN vocoder — lazy-loaded on first mel_to_wav call.
        # Eager loading breaks DataLoader workers on Windows (spawn): the
        # torch.hub repo directory is in sys.path only in the main process,
        # so workers can't unpickle a live HiFiGANVocoder instance.
        self._vocoder = None

        # Griffin-Lim fallback always available (no external deps)
        self.inv_mel = T.InverseMelScale(
            n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
        )
        self.grifflim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _get_vocoder(self) -> "HiFiGANVocoder":
        """Lazy-load HiFi-GAN on first call so DataLoader workers don't need it."""
        if self._vocoder is None:
            self._vocoder = HiFiGANVocoder(device=self.device, fallback_to_griffin_lim=True)
        return self._vocoder

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) or (T,) → mel: (B, n_mels, T_frames), log-scaled."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel_transform(wav)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel  # (B, 80, T_frames)

    def mel_to_wav_hifigan(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, 80, T) log-mel → (B, T_wav) waveform at 22050 Hz."""
        with torch.no_grad():
            wav = self._get_vocoder().forward(mel)  # (B, 1, T_wav) on CPU
        return wav.squeeze(1)  # (B, T_wav)

    def mel_to_wav_grifflim(self, mel: torch.Tensor) -> torch.Tensor:
        """Fallback: Griffin-Lim vocoding (low quality, for debugging).

        mel: (B, 80, T) log-scaled mel-spectrogram
        Returns: (B, T_wav) waveform
        """
        mel = torch.exp(mel)
        spec = self.inv_mel(mel)
        return self.grifflim(spec)

    def mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, 80, T) log-mel → (B, T_wav) waveform. Uses HiFi-GAN, Griffin-Lim on failure."""
        if not self.use_hifigan:
            return self.mel_to_wav_grifflim(mel)
        vocoder = self._get_vocoder()
        if vocoder.use_grifflim_fallback:
            return self.mel_to_wav_grifflim(mel)
        wav = self.mel_to_wav_hifigan(mel)
        # If HiFi-GAN failed mid-call and flipped to grifflim, fall back silently
        if vocoder.use_grifflim_fallback:
            return self.mel_to_wav_grifflim(mel)
        return wav

    def resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample to target sample_rate."""
        if orig_sr == self.sample_rate:
            return wav
        return torchaudio.functional.resample(wav, orig_sr, self.sample_rate)
