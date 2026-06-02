import os
import torch
import soundfile as sf
from pathlib import Path

# Import your Rano modules
from model import Rano
from audio import MelProcessor

def track_the_explosion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Initialize Model
    model = Rano(embed_dim=256, num_cinn_blocks=12).to(device)
    
    ckpt_acg = torch.load("checkpoints/acg/acg_final.pt", map_location=device)
    model.acg.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_acg.get("state_dict", ckpt_acg).items()}, strict=False)
    
    ckpt_anon = torch.load("checkpoints/rano/anonymizer_step108000.pt", map_location=device)
    model.anonymizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_anon.get("state_dict", ckpt_anon).items()}, strict=False)
    
    # CRITICAL: Ensure evaluation mode
    model.eval()

    # 2. Setup MelProcessor
    processor = MelProcessor(device=device, use_hifigan=False, sample_rate=22050)

    # --- CHANGE THIS TO A REAL FILE PATH ---
    test_file = r"D:\Thesis_2.0\rano-implementation\data\LibriSpeech\train-clean-100\19\198\19-198-0001.flac" 

    if not os.path.exists(test_file):
        print(f"ERROR: Target file not found at: {test_file}")
        return

    wav_np, sr = sf.read(test_file)
    if wav_np.ndim == 2: 
        wav_np = wav_np.mean(axis=1)
    
    wav_t = processor.resample(torch.tensor(wav_np).float(), sr)
    
    # ============================================================
    # CHECK 1: The Input Mel (Is the MelProcessor breaking?)
    # ============================================================
    mel = processor.wav_to_mel(wav_t).to(device)
    
    print("--- 1. INPUT MEL DIAGNOSTICS ---")
    print(f"Dtype: {mel.dtype}")
    print(f"Min:   {mel.min().item():.4f}")
    print(f"Max:   {mel.max().item():.4f}")
    print(f"Mean:  {mel.mean().item():.4f}")
    print(f"Infs:  {torch.isinf(mel).sum().item()}")
    print(f"NaNs:  {torch.isnan(mel).sum().item()}")
    print("--------------------------------\n")
    
    # ============================================================
    # CHECK 2: The Rano Anonymization (Is the Model breaking?)
    # ============================================================
    print("Generating Anonymized Latents...")
    
    # CRITICAL: Ensure no gradients are accumulating
    with torch.no_grad():
        key = torch.randn(1, 256, device=device)
        xa, cond = model.anonymize(mel, key)
        
    print("--- 2. RANO OUTPUT (xa) DIAGNOSTICS ---")
    print(f"Dtype: {xa.dtype}")
    print(f"Min:   {xa.min().item():.4f}")
    print(f"Max:   {xa.max().item():.4f}")
    print(f"Mean:  {xa.mean().item():.4f}")
    print(f"Infs:  {torch.isinf(xa).sum().item()}")
    print(f"NaNs:  {torch.isnan(xa).sum().item()}")
    print("---------------------------------------\n")
    
    # Check what the condition tensor looks like just in case
    if cond is not None:
        print("--- 3. RANO CONDITION (cond) DIAGNOSTICS ---")
        print(f"Min:   {cond.min().item():.4f}")
        print(f"Max:   {cond.max().item():.4f}")
        print(f"Infs:  {torch.isinf(cond).sum().item()}")
        print(f"NaNs:  {torch.isnan(cond).sum().item()}")
        print("--------------------------------------------\n")

if __name__ == "__main__":
    track_the_explosion()