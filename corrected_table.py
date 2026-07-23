"""Recompute EER + Sim_spk with ECAPA fed at the CORRECT 16 kHz.

Background: evaluate3.py line ~404 builds `wav_16k = resample_to(wav_np, sr, proc_sr)`
-- proc_sr is 22050, NOT 16000 despite the variable name -- and fed that to ECAPA, which
expects 16 kHz. That cripples the attacker (clean-speech baseline EER 17.67% @22050 vs
5.67% @16000) and INFLATES every reported EER. evaluate3.py is now fixed; this script
re-scores the already-generated eval audio so the old numbers can be corrected without
re-running anything on the cloud.

WER is deliberately NOT recomputed: transcribe() always resampled to 16k, so the bug never
touched it and the existing WER numbers stand.

  cd D:\\Thesis_2.0\\rano-implementation
  .\\venv\\Scripts\\python.exe corrected_table.py
"""
import sys, os, glob, types, warnings
warnings.filterwarnings("ignore")
PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ); sys.path.insert(0, PROJ)
sys.modules.setdefault("k2", types.ModuleType("k2"))  # speechbrain optional dep (py3.14)

import numpy as np, torch, soundfile as sf
from evaluate3 import calculate_eer, cosine_sim, to_mono, resample_to, load_asv

try:  # speechbrain lazy-module guard on py3.14
    from speechbrain.utils import importutils as _sb_iu
    _o = _sb_iu.LazyModule.__getattr__
    def _s(self, n):
        try: return _o(self, n)
        except ImportError: raise AttributeError(n)
    _sb_iu.LazyModule.__getattr__ = _s
except Exception:
    pass

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = r"D:\Thesis_2.0\eval_audio"
FILE_SR = 22050          # rate evaluate3 wrote the wavs at
# (name, audio dir, WER already measured with Whisper large -- unaffected by the SR bug)
RUNS = [
    ("rano_v2 (old ASV)",       "eval_math_vocoder_v2",      8.71),
    ("bal step8000 (HEADLINE)", "eval_bal_gl_fixed",        17.00),
    ("bal2",                    "eval_bal2_best",           27.05),
    ("rano_ecapa",              "eval_rano_ecapa_gl_fixed", 39.52),
]
extract = load_asv(DEV)


def embed(w, target):
    if target != FILE_SR:
        w = resample_to(w, FILE_SR, target)
    return extract(torch.tensor(np.asarray(w, np.float32)).float().to(DEV))


def score(root, target):
    O, A, R = {}, {}, {}
    for spk in sorted(os.listdir(root)):
        d = os.path.join(root, spk)
        if not os.path.isdir(d): continue
        fs = sorted(glob.glob(os.path.join(d, "*_orig.wav")))
        if not fs: continue
        O[spk], A[spk], R[spk] = [], [], []
        for f in fs:
            af = f.replace("_orig.wav", "_anon.wav")
            rf = f.replace("_orig.wav", "_restored.wav")
            if not (os.path.exists(af) and os.path.exists(rf)): continue
            wo, _ = sf.read(f); wa, _ = sf.read(af); wr, _ = sf.read(rf)
            O[spk].append(embed(to_mono(wo).astype(np.float32), target))
            A[spk].append(embed(to_mono(wa).astype(np.float32), target))
            R[spk].append(embed(to_mono(wr).astype(np.float32), target))

    def eer(P, Q, drop_self=False):
        yt, ys = [], []
        for sa, ea in P.items():
            for i, va in enumerate(ea):
                for sb, eb in Q.items():
                    for j, vb in enumerate(eb):
                        if drop_self and sa == sb and i == j: continue
                        ys.append(cosine_sim(va, vb)); yt.append(1 if sa == sb else 0)
        return calculate_eer(yt, ys)

    base = eer(O, O, drop_self=True)          # attacker floor on CLEAN speech
    anon = eer(A, O)                          # the privacy number
    sim = float(np.mean([cosine_sim(O[s][i], R[s][i]) * 100.0
                         for s in O for i in range(len(O[s]))]))   # Sim_spk
    n = sum(len(v) for v in O.values())
    return base, anon, sim, n, len(O)


rows = []
for name, d, wer in RUNS:
    root = os.path.join(BASE, d)
    if not os.path.isdir(root):
        print(f"[skip] not found: {root}")
        continue
    out = {t: score(root, t) for t in (22050, 16000)}
    rows.append((name, wer, out))
    b22, a22, s22, n, ns = out[22050]
    b16, a16, s16, _, _ = out[16000]
    print(f"\n{name}   [{n} utts / {ns} spk]")
    print(f"   {'':22s} {'22050 (buggy)':>15s} {'16000 (CORRECT)':>17s}")
    print(f"   {'attacker floor EER':22s} {b22:14.2f}% {b16:16.2f}%")
    print(f"   {'anon EER':22s} {a22:14.2f}% {a16:16.2f}%")
    print(f"   {'Sim_spk (restore)':22s} {s22:14.2f}% {s16:16.2f}%")

print("\n" + "=" * 74)
print("CORRECTED TABLE  (ECAPA @16 kHz; WER unchanged -- Whisper always used 16k)")
print("=" * 74)
print(f"{'Model':26s} {'floor EER':>10s} {'EER':>8s} {'WER':>8s} {'Sim_spk':>9s}")
print("-" * 74)
for name, wer, out in rows:
    b16, a16, s16, _, _ = out[16000]
    print(f"{name:26s} {b16:9.2f}% {a16:7.2f}% {wer:7.2f}% {s16:8.2f}%")
print("-" * 74)
print("floor EER = ECAPA on CLEAN originals: the attacker's error rate with NO")
print("anonymization at all. That is the scale your EER must be read against.")
