"""Audio ↔ Mel-spectrogram conversion utilities."""

import torch
import torchaudio
import torchaudio.transforms as T


class MelProcessor:
    """
    Converts waveforms ↔ mel-spectrograms at 22050 Hz (paper's sample rate).
    All spectrograms are log-scaled and normalised to [-1, 1].
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
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
        )
        self.inv_mel = T.InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels,
                                          sample_rate=sample_rate)
        self.grifflim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def wav_to_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T) or (T,) → mel: (B, 1, F, T_frames), log-scaled."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel_transform(wav)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel.unsqueeze(1)  # add channel dim

    def mel_to_wav_grifflim(self, mel: torch.Tensor) -> torch.Tensor:
        """Approximate inversion via Griffin-Lim (vocoder-free baseline)."""
        mel = mel.squeeze(1)
        mel = torch.exp(mel)
        spec = self.inv_mel(mel)
        return self.grifflim(spec)

    def resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample to target sample_rate in-place."""
        if orig_sr == self.sample_rate:
            return wav
        return torchaudio.functional.resample(wav, orig_sr, self.sample_rate)
