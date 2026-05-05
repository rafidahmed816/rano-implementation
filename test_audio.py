import torch
from audio import MelProcessor

processor = MelProcessor()
# Generate a dummy waveform (sine wave)
t = torch.linspace(0, 1, 22050)
wav = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

# Convert to mel
mel = processor.wav_to_mel(wav)
print("Mel mean:", mel.mean().item(), "min:", mel.min().item(), "max:", mel.max().item())

# Convert back to wav
wav_recon = processor.mel_to_wav_grifflim(mel)
print("Recon mean:", wav_recon.mean().item(), "min:", wav_recon.min().item(), "max:", wav_recon.max().item())

