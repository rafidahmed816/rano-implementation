"""Strong held-out attacker: SpeechBrain VoxCeleb ResNet (spkrec-resnet-voxceleb).

WeSpeaker has no installable wheel for this env (py3.14); this ResNet is the
equivalent strong, cosine-native, DIFFERENT-ARCHITECTURE verifier never used in
training. Fed at the CORRECT 16 kHz (unlike evaluate3's old bug). Sanity gate first.
"""
import os, sys, glob, types, shutil, warnings
warnings.filterwarnings("ignore")
PROJ = r"D:\Thesis_2.0\rano-implementation"
os.chdir(PROJ); sys.path.insert(0, PROJ)
sys.modules.setdefault("k2", types.ModuleType("k2"))
import numpy as np, torch, soundfile as sf
from evaluate3 import calculate_eer, cosine_sim, to_mono, resample_to

try:
    from speechbrain.utils import importutils as _iu
    _o = _iu.LazyModule.__getattr__
    def _s(self, n):
        try: return _o(self, n)
        except ImportError: raise AttributeError(n)
    _iu.LazyModule.__getattr__ = _s
except Exception:
    pass

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_SR = 22050


def load_resnet():
    _sl = os.symlink
    def _p(src, dst, tid=False, *a, **k):
        try: _sl(src, dst, tid, *a, **k)
        except OSError:
            shutil.copytree(src, dst, dirs_exist_ok=True) if os.path.isdir(src) else shutil.copy2(src, dst)
    os.symlink = _p
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-resnet-voxceleb",
                                       savedir="tmp_resnet", run_opts={"device": str(DEV)})
    os.symlink = _sl
    def emb(w):
        w = resample_to(np.asarray(w, np.float32), FILE_SR, 16000)   # ResNet expects 16k
        t = torch.tensor(w).float().to(DEV)
        if t.ndim == 1: t = t.unsqueeze(0)
        with torch.no_grad():
            return m.encode_batch(t).squeeze().cpu().numpy()
    return emb


RUNS = [  # (name, dir, ECAPA-16k, Resemblyzer)
    ("rano_v2 (old ASV)",   r"D:\Thesis_2.0\rano-implementation\eval_rano_v2_local", 18.35, 26.67),
    ("bal (a1.0 l2=3)",     r"D:\Thesis_2.0\eval_audio\eval_bal_gl_fixed",           29.31, 27.36),
    ("bal2 (a1.2 l2=6)",    r"D:\Thesis_2.0\rano-implementation\eval_bal2_local",    32.81, 32.64),
    ("rano_ecapa (a0.2)",   r"D:\Thesis_2.0\eval_audio\eval_rano_ecapa_gl_fixed",    39.58, 35.28),
]
emb = load_resnet()


def eer(P, Q, drop_self=False):
    yt, ys, g, i = [], [], [], []
    for sa, ea in P.items():
        for a, va in enumerate(ea):
            for sb, eb in Q.items():
                for b, vb in enumerate(eb):
                    if drop_self and sa == sb and a == b: continue
                    c = cosine_sim(va, vb); ys.append(c); yt.append(1 if sa == sb else 0)
                    (g if sa == sb else i).append(c)
    return calculate_eer(yt, ys), (np.mean(g) - np.mean(i) if g and i else 0.0)


rows = []
for name, root, ec, rz in RUNS:
    if not os.path.isdir(root):
        print(f"[skip] {root}"); continue
    O, A = {}, {}
    for spk in sorted(os.listdir(root)):
        d = os.path.join(root, spk)
        if not os.path.isdir(d): continue
        fs = sorted(glob.glob(os.path.join(d, "*_orig.wav")))
        if not fs: continue
        O[spk], A[spk] = [], []
        for f in fs:
            af = f.replace("_orig.wav", "_anon.wav")
            if not os.path.exists(af): continue
            wo, _ = sf.read(f); wa, _ = sf.read(af)
            O[spk].append(emb(to_mono(wo))); A[spk].append(emb(to_mono(wa)))
    base, sep = eer(O, O, drop_self=True)
    anon, _ = eer(A, O)
    rows.append((name, base, sep, anon, ec, rz))
    gate = "OK" if sep > 0.15 else "WEAK/BROKEN"
    print(f"{name:20s} floor {base:6.2f}%  sep {sep:5.3f} [{gate}]  anon {anon:6.2f}%", flush=True)

print("\n" + "=" * 74)
print("ALL ATTACKERS (anon EER %) -- ResNet is strong + HELD OUT")
print("=" * 74)
print(f"{'Model':20s}{'floor':>8s}{'ResNet(ho)':>11s}{'ECAPA':>8s}{'Resmb(ho)':>11s}")
print("-" * 74)
for name, base, sep, anon, ec, rz in rows:
    print(f"{name:20s}{base:7.2f}%{anon:10.2f}%{ec:7.2f}%{rz:10.2f}%")
print("-" * 74)
print("ho = held out (never used in training). If ResNet tracks ECAPA, privacy")
print("generalizes to a STRONG unseen attacker -- the airtight version of the claim.")
