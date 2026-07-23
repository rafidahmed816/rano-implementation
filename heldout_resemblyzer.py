"""Held-out attacker: Resemblyzer (GE2E d-vector) -- weak, industry-used, cosine-native,
and NEVER used in training. The real test of whether the anonymization generalizes beyond
the distilled ECAPA.

Gate first: a verifier's anon EER is meaningless unless it can verify CLEAN speakers on
this audio. So per model we report the clean-baseline EER and genuine/impostor separation
BEFORE the anon number. (x-vector failed this gate: separation 0.034.)

Resemblyzer.preprocess_wav resamples to 16k + VAD-trims internally, so no SR bug here.

  cd D:\\Thesis_2.0\\rano-implementation
  .\\venv\\Scripts\\python.exe heldout_resemblyzer.py
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from evaluate3 import calculate_eer, cosine_sim, to_mono

FILE_SR = 22050
RUNS = [  # (name, absolute audio dir) -- fixed-code audio, ECAPA-16k already known
    ("rano_v2 (old ASV)",       r"D:\Thesis_2.0\rano-implementation\eval_rano_v2_local"),
    ("bal (anchor1.0 l2=3)",    r"D:\Thesis_2.0\eval_audio\eval_bal_gl_fixed"),
    ("bal2 (anchor1.2 l2=6)",   r"D:\Thesis_2.0\rano-implementation\eval_bal2_local"),
    ("rano_ecapa (anchor0.2)",  r"D:\Thesis_2.0\eval_audio\eval_rano_ecapa_gl_fixed"),
]
enc = VoiceEncoder()


def emb(w):
    return enc.embed_utterance(preprocess_wav(np.asarray(w, np.float32), source_sr=FILE_SR))


def eer(P, Q, drop_self=False):
    yt, ys, gen, imp = [], [], [], []
    for sa, ea in P.items():
        for i, va in enumerate(ea):
            for sb, eb in Q.items():
                for j, vb in enumerate(eb):
                    if drop_self and sa == sb and i == j: continue
                    c = cosine_sim(va, vb); ys.append(c); yt.append(1 if sa == sb else 0)
                    (gen if sa == sb else imp).append(c)
    return calculate_eer(yt, ys), (np.mean(gen) - np.mean(imp) if gen and imp else 0.0)


rows = []
for name, d in RUNS:
    root = d
    if not os.path.isdir(root):
        print(f"[skip] {d} not found"); continue
    O, A = {}, {}
    for spk in sorted(os.listdir(root)):
        sd = os.path.join(root, spk)
        if not os.path.isdir(sd): continue
        fs = sorted(glob.glob(os.path.join(sd, "*_orig.wav")))
        if not fs: continue
        O[spk], A[spk] = [], []
        for f in fs:
            af = f.replace("_orig.wav", "_anon.wav")
            if not os.path.exists(af): continue
            wo, _ = sf.read(f); wa, _ = sf.read(af)
            O[spk].append(emb(to_mono(wo))); A[spk].append(emb(to_mono(wa)))
    base, sep = eer(O, O, drop_self=True)     # GATE
    anon, _ = eer(A, O)
    n = sum(len(v) for v in O.values())
    rows.append((name, base, sep, anon, n, len(O)))
    ok = "OK" if (base < 12 and sep > 0.15) else "BROKEN -> anon EER meaningless"
    print(f"\n{name}  [{n} utts / {len(O)} spk]")
    print(f"   clean baseline EER {base:6.2f}% | separation {sep:5.3f}  [{ok}]")
    print(f"   anon EER (Resemblyzer, HELD OUT) {anon:6.2f}%")

print("\n" + "=" * 66)
print("HELD-OUT (Resemblyzer)  vs  ECAPA (from corrected table)")
print("=" * 66)
ecapa = {"rano_v2 (old ASV)": 18.35, "bal (anchor1.0 l2=3)": 29.31,
         "bal2 (anchor1.2 l2=6)": 32.81, "rano_ecapa (anchor0.2)": 39.58}
print(f"{'Model':26s} {'floor':>7s} {'Resmb EER':>10s} {'ECAPA EER':>10s}")
print("-" * 66)
for name, base, sep, anon, n, ns in rows:
    e = ecapa.get(name)
    print(f"{name:26s} {base:6.2f}% {anon:9.2f}% {('%.2f%%'%e) if e else '   n/a':>10s}")
print("-" * 66)
print("If Resemblyzer EER tracks ECAPA (both well above the ~5-6% floor), the")
print("privacy generalizes to an unseen attacker. If it sits near the floor,")
print("the anonymization only fooled ECAPA-like models.")
