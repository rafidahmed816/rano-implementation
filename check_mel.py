import torch
import soundfile as sf
import warnings
from pathlib import Path

from model import Rano
from audio import MelProcessor

warnings.filterwarnings("ignore")

def load_weights(path: str, module: torch.nn.Module, name: str, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
    module.load_state_dict(ckpt, strict=False)

def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    processor = MelProcessor(device=device)
    model = Rano(embed_dim=256, num_cinn_blocks=12).to(device)
    
    # Load your checkpoints (make sure paths match yours)
    load_weights("checkpoints/acg/acg_final.pt", model.acg, "ACG", device)
    load_weights("checkpoints/rano/anonymizer_step108000.pt", model.anonymizer, "Anonymizer", device)
    model.eval()

    # --- CHANGE THIS TO A REAL FILE PATH ---
    test_file = r"D:\Thesis_2.0\rano-implementation\data\LibriSpeech\train-clean-100\19\198\19-198-0001.flac"
    
    if not Path(test_file).exists():
        print(f"Please point 'test_file' to a valid audio file. Cannot find: {test_file}")
        return

    wav_np, sr = sf.read(test_file)
    if wav_np.ndim == 2: wav_np = wav_np.mean(axis=1) # to mono
    wav_t = processor.resample(torch.tensor(wav_np).float(), sr)
    
    print("=========================================")
    # 1. Test Mel Extraction
    mel = processor.wav_to_mel(wav_t).to(device)
    print(f"--- 1. ORIGINAL MEL ---")
    print(f"Shape: {tuple(mel.shape)}")
    print(f"Min:   {mel.min().item():.4f}  |  Max: {mel.max().item():.4f}  |  Mean: {mel.mean().item():.4f}")

    # 2. Test Vocoder (Griffin-Lim)
    # We bypass the neural vocoder and use pure math
    baseline_wav = processor.mel_to_wav_grifflim(mel.cpu()).squeeze().numpy()
    sf.write("DEBUG_baseline_GL.wav", baseline_wav, processor.sample_rate)
    print("\n-> Saved 'DEBUG_baseline_GL.wav'")
    print("   (Listen to this! It should sound like a human, maybe slightly echoey.)")

    # 3. Test Model Forward (Anonymize)
    with torch.no_grad():
        key = torch.randn(1, 256, device=device)
        xa, _ = model.anonymize(mel, key)
    
    # 4. Generate Anonymized Audio
    anon_wav = processor.mel_to_wav_grifflim(xa.cpu()).squeeze().numpy()
    sf.write("DEBUG_anon_GL.wav", anon_wav, processor.sample_rate)
    print("\n-> Saved 'DEBUG_anon_GL.wav'")
    print("   (Listen to this! This is your model's actual anonymized output.)")

    # 5. Test Model Inverse (Restore)
    with torch.no_grad():
        xr = model.restore(xa, key)
        
    restored_wav = processor.mel_to_wav_grifflim(xr.cpu()).squeeze().numpy()
    sf.write("DEBUG_restored_GL.wav", restored_wav, processor.sample_rate)
    print("\n-> Saved 'DEBUG_restored_GL.wav'")
    print("   (Listen to this! It should sound identical to the baseline.)")
    print("=========================================")
if __name__ == "__main__":
    run_diagnostic()