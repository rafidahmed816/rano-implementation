"""Wrong-key security experiment (matches Table III of the Rano paper).

Question: if an attacker holds a key CLOSE to the correct one, does restoration leak
the original speaker? A usable key-based scheme must fail for every wrong key, not
just for distant ones.

Method. Anonymize with the correct condition c, then attempt restoration with a
perturbed condition c' at a controlled distance D_key = ||c - c'||_2. Because both
conditions are L2-normalised, c' is constructed exactly on the unit sphere:

    c' = cos(t) c + sin(t) v,   v unit, v _|_ c,   cos(t) = 1 - D^2/2

which gives ||c - c'|| = sqrt(2 - 2cos t) = D exactly. Since the ACG is invertible,
every such c' corresponds to a unique wrong key, so perturbing the condition is
equivalent to perturbing the key while allowing exact control of the distance bins.

Restored audio is vocoded with the fine-tuned HiFi-GAN and scored with ECAPA-TDNN at
16 kHz, matching the paper's note that measurement is done on vocoder output.

  python security_wrongkey.py --n_utts 120
"""
from __future__ import annotations
import argparse, glob, os, sys, types, warnings
warnings.filterwarnings("ignore")
PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ); sys.path.insert(0, PROJ)
sys.modules.setdefault("k2", types.ModuleType("k2"))   # speechbrain optional dep

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

from model import Rano
from audio import MelProcessor
from evaluate3 import (load_asv, cosine_sim, to_mono, resample_to,
                       peak_normalize, load_custom_vocoder, MEL_MIN, MEL_MAX)

try:  # py3.14 speechbrain lazy-module guard
    from speechbrain.utils import importutils as _iu
    _o = _iu.LazyModule.__getattr__
    def _s(self, n):
        try: return _o(self, n)
        except ImportError: raise AttributeError(n)
    _iu.LazyModule.__getattr__ = _s
except Exception:
    pass

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 22050


def anon_with_cond(model, x, cond):
    """Forward pass with an explicit condition (float64, as in Rano.anonymize)."""
    dt = next(model.anonymizer.parameters()).dtype
    model.anonymizer.double()
    xa, _ = model.anonymizer(x.double(), cond.double())
    model.anonymizer.to(dt)
    return xa.float()


def restore_with_cond(model, xa, cond):
    """Inverse pass with an explicit (possibly WRONG) condition."""
    dt = next(model.anonymizer.parameters()).dtype
    model.anonymizer.double()
    xr = model.anonymizer.inverse(xa.double(), cond.double())
    model.anonymizer.to(dt)
    return xr.float()


def perturb_condition(c, d):
    """Return c' on the unit sphere with ||c - c'||_2 exactly d."""
    if d <= 0:
        return c.clone()
    v = torch.randn_like(c)
    v = v - (v * c).sum(dim=-1, keepdim=True) * c      # make orthogonal to c
    v = F.normalize(v, dim=-1)
    cos_t = 1.0 - (d ** 2) / 2.0
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t ** 2)))
    return cos_t * c + sin_t * v


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    proc = MelProcessor(device=DEV, use_hifigan=False, sample_rate=SR)
    proc.mel_transform = proc.mel_transform.to(DEV)

    model = Rano(embed_dim=256, num_cinn_blocks=12).to(DEV).eval()
    model.acg.load_state_dict(torch.load(args.acg_checkpoint, map_location=DEV, weights_only=True))
    model.anonymizer.load_state_dict(torch.load(args.anonymizer_ckpt, map_location=DEV, weights_only=True))
    print(f"[OK] model loaded", flush=True)

    vocode = load_custom_vocoder(args.vocoder_ckpt, DEV)
    extract = load_asv(DEV)

    def emb(wav_np):
        t = torch.tensor(np.asarray(wav_np, np.float32)).float().to(DEV)
        return extract(t, SR)          # evaluate3's extract_emb resamples to 16 kHz

    files = sorted(glob.glob(os.path.join(args.test_dir, "*", "*.wav"))) + \
            sorted(glob.glob(os.path.join(args.test_dir, "*", "*.flac")))
    files = files[:args.n_utts]
    if not files:
        raise SystemExit(f"no audio under {args.test_dir}")
    print(f"utterances: {len(files)}", flush=True)

    # (label, low, high) -- paper's bins, plus the correct-key reference
    BINS = [("correct key (D=0)", 0.0, 0.0),
            ("0.1 < D_key < 0.2", 0.1, 0.2),
            ("0.2 < D_key < 0.3", 0.2, 0.3),
            ("0.3 < D_key < 0.4", 0.3, 0.4)]
    sims = {b[0]: [] for b in BINS}
    dists = {b[0]: [] for b in BINS}

    with torch.no_grad():
        for i, f in enumerate(files, 1):
            w, sr = sf.read(f)
            w = resample_to(to_mono(w), sr, SR)
            if len(w) < 2048:
                continue
            X = proc.wav_to_mel(torch.tensor(w).float().to(DEV))

            k = torch.randn(1, 256, device=DEV)
            c = model.acg.generate(k)                     # correct condition (unit norm)
            Xa = anon_with_cond(model, X, c)

            emb_o = emb(peak_normalize(w))                # reference: the true original

            for label, lo, hi in BINS:
                d = 0.0 if hi == 0.0 else float(np.random.uniform(lo, hi))
                c2 = perturb_condition(c, d)
                dists[label].append(float(torch.norm(c - c2).item()))
                Xr = restore_with_cond(model, Xa, c2)
                wav_r = vocode(Xr.clamp(MEL_MIN, MEL_MAX))
                sims[label].append(cosine_sim(emb_o, emb(peak_normalize(wav_r))) * 100.0)

            if i % 20 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    print("\n" + "=" * 68)
    print("WRONG-KEY SECURITY  (Sim_spk on vocoder output, higher = more leakage)")
    print("=" * 68)
    print(f"{'Case':22s}{'mean D_key':>12s}{'Sim_spk (%)':>14s}{'std':>10s}")
    print("-" * 68)
    for label, _, _ in BINS:
        v = np.array(sims[label]); dd = np.array(dists[label])
        print(f"{label:22s}{dd.mean():12.3f}{v.mean():14.2f}{v.std():10.2f}")
    print("-" * 68)
    print("Paper Table III reference: 12.78 / 9.49 / 4.66 for the three wrong-key")
    print("bins (nearest to farthest), 87.75 with the correct key.")


if __name__ == "__main__":
    CK = r"D:\Thesis_2.0\cloud_ckpts"
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", default="test_unseen")
    p.add_argument("--acg_checkpoint", default=fr"{CK}\asv_acg\acg_ecapa\acg_final.pt")
    p.add_argument("--anonymizer_ckpt", default=fr"{CK}\rano_ecapa_bal\anonymizer_step8000.pt")
    p.add_argument("--vocoder_ckpt", default="checkpoints/vocoder_ft/generator.pt")
    p.add_argument("--n_utts", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
