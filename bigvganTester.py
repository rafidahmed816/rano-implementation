import sys
import os
import torch
import soundfile as sf
import math
from pathlib import Path

# Import BigVGAN and your Rano modules
import bigvgan
from model import Rano
from audio import MelProcessor

# ==========================================
# BUG FIX: Patch BigVGAN to work with latest Hugging Face Hub
# ==========================================
orig_from_pretrained = bigvgan.BigVGAN._from_pretrained

@classmethod
def patched_from_pretrained(cls, *args, **kwargs):
    if 'proxies' not in kwargs:
        kwargs['proxies'] = None
    if 'resume_download' not in kwargs:
        kwargs['resume_download'] = None
    return orig_from_pretrained.__func__(cls, *args, **kwargs)

bigvgan.BigVGAN._from_pretrained = patched_from_pretrained
# ==========================================

def test_bigvgan_vocoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # 1. Initialize Rano Model
    model = Rano(embed_dim=256, num_cinn_blocks=12).to(device)
    
    # Load your Rano checkpoints (ensure paths are correct)
    ckpt_acg = torch.load("checkpoints/acg/acg_final.pt", map_location=device)
    model.acg.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_acg.get("state_dict", ckpt_acg).items()}, strict=False)
    
    ckpt_anon = torch.load("checkpoints/rano/anonymizer_step108000.pt", map_location=device)
    model.anonymizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_anon.get("state_dict", ckpt_anon).items()}, strict=False)
    model.eval()

    # 2. Setup your MelProcessor at 22050Hz (Matches BigVGAN exactly)
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
    mel = processor.wav_to_mel(wav_t).to(device)
    
    # 3. Load NVIDIA's Universal BigVGAN (22kHz, 80 Band)
    print("Downloading/Loading BigVGAN...")
    bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_22khz_80band', use_cuda_kernel=False).to(device)
    bigvgan_model.remove_weight_norm()
    bigvgan_model.eval()

    print("Generating Anonymized Latents...")
    with torch.no_grad():
        key = torch.randn(1, 256, device=device)
        xa, _ = model.anonymize(mel, key)
        
        # ============================================================
        # EXPLICIT DISTRIBUTION DIAGNOSTICS & CONVERSION FIX
        # ============================================================
        print("\n--- LATENT DISTRIBUTION SANITY CHECK ---")
        print(f"Raw Model Output (xa) Min:  {xa.min().item():.4f}")
        print(f"Raw Model Output (xa) Max:  {xa.max().item():.4f}")
        print(f"Raw Model Output (xa) Mean: {xa.mean().item():.4f}")
        
        # Rigorous dB-to-Natural-Log conversion scale factor
        scale_factor = math.log(10.0) / 20.0  # Approx 0.115129
        xa_converted = xa * scale_factor
        
        # Clamp to BigVGAN's specific operating range
        xa_corrected = torch.clamp(xa_converted, min=-11.0, max=2.0)
        
        print("\n--- POST-CONVERSION CHECK ---")
        print(f"Corrected Tensor Min:       {xa_corrected.min().item():.4f}")
        print(f"Corrected Tensor Max:       {xa_corrected.max().item():.4f}")
        print(f"Corrected Tensor Mean:      {xa_corrected.mean().item():.4f}")
        print("----------------------------------------\n")
        # ============================================================
        
        # 4. Synthesize Audio using BigVGAN
        print("Synthesizing audio through BigVGAN...")
        anon_wav_tensor = bigvgan_model(xa_corrected) 
        anon_wav = anon_wav_tensor.squeeze().cpu().numpy()

    # Save the output
    out_path = "BigVGAN_test_anon.wav"
    sf.write(out_path, anon_wav, 22050)
    print(f"\nSUCCESS! Audio saved to: {out_path}")
    print("Run the script and check the printed distribution statistics.")

if __name__ == "__main__":
    test_bigvgan_vocoder()