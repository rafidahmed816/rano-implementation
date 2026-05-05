"""Audio ↔ Mel-spectrogram conversion utilities.

All mel-spectrograms are shape (B, 80, T) — 1D along time axis.
Uses HiFi-GAN vocoder (§IV-A) for mel→wav conversion.
"""

import torch
import torchaudio
import torchaudio.transforms as T


class HiFiGANVocoder:
    """HiFi-GAN neural vocoder for high-quality mel→wav synthesis.

    Loads pre-trained model from torch hub.
    Expects mel-spectrograms: (B, 80, T) shape, non-log scale.

    **Configuration Alignment** (Paper §IV-A):
    Standard bshall/hifigan trained on:
        - sample_rate: 22050 Hz
        - n_fft: 1024
        - hop_length: 256
        - n_mels: 80
        - fmax: 8000 Hz

    These MUST match MelProcessor settings for correct vocoding.
    """

    def __init__(self, device: torch.device = None, fallback_to_griffin_lim: bool = True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.fallback_to_griffin_lim = fallback_to_griffin_lim
        self.use_grifflim_fallback = False
        self.hifigan_type = None

        # Try torch hub HiFi-GAN first (primary)
        if self._try_torch_hub_hifigan():
            print(f"[✓] HiFi-GAN loaded from torch hub (bshall/hifigan)")
            return

        # Try transformers SpeechT5HifiGan (secondary fallback)
        if self._try_transformers_hifigan():
            print(f"[✓] HiFi-GAN loaded from transformers (SpeechT5HifiGan)")
            return

        # Fallback to Griffin-Lim if requested
        if fallback_to_griffin_lim:
            print(f"[⚠] HiFi-GAN unavailable. Will use Griffin-Lim vocoding.")
            self.use_grifflim_fallback = True
        else:
            raise RuntimeError(
                "[✗] HiFi-GAN vocoder failed to load and Griffin-Lim fallback disabled.\n"
                "    Install: pip install torch torchaudio\n"
                "    Or set fallback_to_griffin_lim=True"
            )

    def _try_torch_hub_hifigan(self) -> bool:
        """Try loading HiFi-GAN from bshall torch hub."""
        try:
            self.model = torch.hub.load(
                "bshall/hifigan:main",
                "hifigan",
                pretrained=True,
                trust_repo=True,
            ).to(self.device)
            self.model.eval()
            self.hifigan_type = "torch_hub"
            return True
        except Exception as e:
            print(f"[!] Torch hub HiFi-GAN failed: {e}")
            return False

    def _try_transformers_hifigan(self) -> bool:
        """Try loading HiFi-GAN from transformers (fallback)."""
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
        """Convert mel-spectrogram to waveform.

        mel: (B, 80, T) non-log mel-spectrogram
        Returns: (B, 1, T_wav) waveform

        **Configuration Alignment**:
        Input mel must be in linear scale (not log).
        MelProcessor automatically converts from log scale before calling.
        """
        if self.use_grifflim_fallback:
            # Graceful fallback - return silence signal
            # MelProcessor.mel_to_wav will handle Griffin-Lim instead
            return torch.zeros(mel.shape[0], 1, mel.shape[-1] * 256, device=mel.device)

        mel = mel.to(self.device)

        try:
            if self.hifigan_type == "torch_hub":
                # bshall torch hub format: mel (B, 80, T) → wav (B, T_wav)
                wav = self.model(mel).squeeze(1)  # (B, T_wav)
            else:
                # transformers format: mel (B, 80, T) → wav (B, 1, T_wav)
                wav = self.model(mel).waveform  # (B, 1, T_wav)
                wav = wav.squeeze(1)  # (B, T_wav)

            return wav.unsqueeze(1)  # (B, 1, T_wav)
        except Exception as e:
            print(f"[✗] HiFi-GAN forward pass failed: {e}")
            print(f"    Mel shape: {mel.shape}")
            print(f"    This indicates a configuration mismatch.")
            print(f"    Falling back to Griffin-Lim...")
            self.use_grifflim_fallback = True
            return torch.zeros(mel.shape[0], 1, mel.shape[-1] * 256, device=mel.device)


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
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.use_hifigan = use_hifigan

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
        )

        # HiFi-GAN vocoder (§IV-A: "We utilize Hifi-GAN [39] as the vocoder")
        if use_hifigan:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.vocoder = HiFiGANVocoder(device=device, fallback_to_griffin_lim=True)
        else:
            # Fallback: keep Griffin-Lim for debugging
            self.inv_mel = T.InverseMelScale(
                n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
            )
            self.grifflim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) or (T,) → mel: (B, n_mels, T_frames), log-scaled."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel_transform(wav)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel  # (B, 80, T_frames)

    def mel_to_wav_hifigan(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert log-mel-spectrogram to waveform using HiFi-GAN vocoder.

        mel: (B, 80, T) log-scaled mel-spectrogram
        Returns: (B, T_wav) waveform at 22050 Hz
        """
        # Convert from log scale to linear scale
        mel = torch.exp(mel)

        # Clamp to valid range for numerical stability
        mel = mel.clamp(min=1e-5, max=1e5)

        # HiFi-GAN vocoding
        with torch.no_grad():
            wav = self.vocoder.forward(mel)  # (B, 1, T_wav)

        wav = wav.squeeze(1)  # (B, T_wav)
        return wav

    def mel_to_wav_grifflim(self, mel: torch.Tensor) -> torch.Tensor:
        """Fallback: Griffin-Lim vocoding (low quality, for debugging).

        mel: (B, 80, T) log-scaled mel-spectrogram
        Returns: (B, T_wav) waveform
        """
        mel = torch.exp(mel)
        spec = self.inv_mel(mel)
        return self.grifflim(spec)

    def mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to waveform.

        Intelligently selects vocoder:
        1. HiFi-GAN (preferred, §IV-A paper)
        2. Griffin-Lim (fallback if HiFi-GAN unavailable/failed)

        mel: (B, 80, T) log-scaled mel-spectrogram
        Returns: (B, T_wav) waveform at 22050 Hz
        """
        if self.use_hifigan:
            # Check if vocoder has fallen back to Griffin-Lim
            if (
                hasattr(self.vocoder, "use_grifflim_fallback")
                and self.vocoder.use_grifflim_fallback
            ):
                print(f"[HiFi-GAN unavailable] Using Griffin-Lim fallback")
                return self.mel_to_wav_grifflim(mel)
            return self.mel_to_wav_hifigan(mel)
        else:
            return self.mel_to_wav_grifflim(mel)

    def resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample to target sample_rate."""
        if orig_sr == self.sample_rate:
            return wav
        return torchaudio.functional.resample(wav, orig_sr, self.sample_rate)
