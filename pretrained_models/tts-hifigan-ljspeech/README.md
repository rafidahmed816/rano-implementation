---
language: "en"
inference: false
tags:
- Vocoder
- HiFIGAN
- text-to-speech
- TTS
- speech-synthesis
- speechbrain
license: "apache-2.0"
datasets:
- LJSpeech
---

# Vocoder with HiFIGAN trained on LJSpeech

This repository provides all the necessary tools for using a [HiFIGAN](https://arxiv.org/abs/2010.05646) vocoder trained with [LJSpeech](https://keithito.com/LJ-Speech-Dataset/). 

The pre-trained model takes in input a spectrogram and produces a waveform in output. Typically, a vocoder is used after a TTS model that converts an input text into a spectrogram.

The sampling frequency is 22050 Hz.

**NOTES**
- This vocoder model is trained on a single speaker. Although it has some ability to generalize to different speakers, for better results, we recommend using a multi-speaker vocoder like [this model trained on LibriTTS at 16,000 Hz](https://huggingface.co/speechbrain/tts-hifigan-libritts-16kHz) or [this one trained on LibriTTS at 22,050 Hz](https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz).
- If you specifically require a vocoder with a 16,000 Hz sampling rate, please follow the provided link above for a suitable option.

## Install SpeechBrain

```bash
pip install speechbrain
```


Please notice that we encourage you to read our tutorials and learn more about
[SpeechBrain](https://speechbrain.github.io).

### Using the Vocoder

- *Basic Usage:*
```python
import torch
from speechbrain.inference.vocoders import HIFIGAN
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")
mel_specs = torch.rand(2, 80,298)
waveforms = hifi_gan.decode_batch(mel_specs)
```

- *Convert a Spectrogram into a Waveform:*

```python
import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Load a pretrained HIFIGAN Vocoder
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")

# Load an audio file (an example file can be found in this repository)
# Ensure that the audio signal is sampled at 22050 Hz; refer to the provided link for a 16 kHz Vocoder.
signal, rate = torchaudio.load('speechbrain/tts-hifigan-ljspeech/example.wav')

# Compute the mel spectrogram.
# IMPORTANT: Use these specific parameters to match the Vocoder's training settings for optimal results.
spectrogram, _ = mel_spectogram(
    audio=signal.squeeze(),
    sample_rate=22050,
    hop_length=256,
    win_length=None,
    n_mels=80,
    n_fft=1024,
    f_min=0.0,
    f_max=8000.0,
    power=1,
    normalized=False,
    min_max_energy_norm=True,
    norm="slaney",
    mel_scale="slaney",
    compression=True
)

# Convert the spectrogram to waveform
waveforms = hifi_gan.decode_batch(spectrogram)

# Save the reconstructed audio as a waveform
torchaudio.save('waveform_reconstructed.wav', waveforms.squeeze(1), 22050)

# If everything is set up correctly, the original and reconstructed audio should be nearly indistinguishable.
# Keep in mind that this Vocoder is trained for a single speaker; for multi-speaker Vocoder options, refer to the provided links.

```

### Using the Vocoder with the TTS
```python
import torchaudio
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="pretrained_models/tts-tacotron2-ljspeech")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_model/tts-hifigan-ljspeech")

# Running the TTS
mel_output, mel_length, alignment = tacotron2.encode_text("Mary had a little lamb")

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waverform
torchaudio.save('example_TTS.wav',waveforms.squeeze(1), 22050)
```

### Inference on GPU
To perform inference on the GPU, add  `run_opts={"device":"cuda"}`  when calling the `from_hparams` method.

### Training
The model was trained with SpeechBrain.
To train it from scratch follow these steps:
1. Clone SpeechBrain:
```bash
git clone https://github.com/speechbrain/speechbrain/
```
2. Install it:
```bash
cd speechbrain
pip install -r requirements.txt
pip install -e .
```
3. Run Training:
```bash
cd recipes/LJSpeech/TTS/vocoder/hifi_gan/
python train.py hparams/train.yaml --data_folder /path/to/LJspeech
```
You can find our training results (models, logs, etc) [here](https://drive.google.com/drive/folders/19sLwV7nAsnUuLkoTu5vafURA9Fo2WZgG?usp=sharing).