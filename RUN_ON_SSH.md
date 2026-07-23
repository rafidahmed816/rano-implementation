# Running Rano on a Cloud GPU over SSH — Step by Step

Target machine: **RTX 6000 Ada 48 GB**, 60 GB RAM, **120 GB SSD** (OS + venv + code),
**1 TB HDD at `/mnt/storage`** (datasets + training outputs).

Each command block is labelled:
- **🖥️ PC** = run in Git Bash on your Windows PC.
- **☁️ CLOUD** = run in the SSH session on the rented machine.

> Set these once so you can copy-paste. On your PC (🖥️) and again after you connect (☁️):
> ```bash
> HOST=user@YOUR.SERVER.IP     # your provider's SSH user@host
> PORT=22                       # your provider's SSH port (often not 22)
> ```
> Connect: `ssh -p $PORT $HOST`

---

## Disk layout (what goes where)

| Location | Disk | Contents |
|---|---|---|
| `~/rano-implementation/` | SSD | the code (`*.py`) |
| `~/rano_env/` | SSD | the Python virtual environment |
| `~/rano-implementation/checkpoints/` | SSD | trained models (small, ~100–300 MB each) |
| `/mnt/storage/data/` | **HDD** | **VCTK + LibriTTS datasets** |
| `/mnt/storage/checkpoints/`, `/mnt/storage/logs/` | **HDD** | Stage-2 outputs + TensorBoard logs |

Rule of thumb: **big things (datasets, run outputs) → HDD**; code + venv → SSD.

---

## Step 1 — Copy the code to the machine  🖥️ PC

Only the code is transferred; datasets are downloaded directly on the cloud (Step 3).

```bash
# 🖥️ PC — from the project folder
cd /d/Thesis_2.0/rano-implementation
ssh -p $PORT $HOST 'mkdir -p ~/rano-implementation'
scp -P $PORT *.py *.md $HOST:~/rano-implementation/
```

(That copies every `.py` and the docs. It does NOT copy `venv/`, datasets, or old
checkpoints — you don't need the old checkpoints for a full retrain.)

---

## Step 2 — Python virtual environment  ☁️ CLOUD  (keeps the OS clean)

**Never `pip install` into the system Python** — it can break OS tools. Everything
below lives inside an isolated venv on the SSD.

```bash
# ☁️ CLOUD
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip libsndfile1

python3 -m venv ~/rano_env          # create the isolated env
source ~/rano_env/bin/activate      # activate it — your prompt should show (rano_env)

pip install --upgrade pip
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install soundfile tqdm tensorboard numpy

# verify the GPU is visible from inside the venv
python -c "import torch;print('CUDA:',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

> **Every time** you open a new SSH session, re-activate first:
> `source ~/rano_env/bin/activate`

Training needs only the packages above. (Evaluation later also needs
`librosa openai-whisper speechbrain scipy scikit-learn transformers huggingface_hub`.)

---

## Step 3 — Datasets onto the HDD  ☁️ CLOUD

```bash
# ☁️ CLOUD
mkdir -p /mnt/storage/data && cd /mnt/storage/data

# --- LibriTTS train-clean-100 (~7 GB; 'test-clean' is too small for training) ---
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
tar xzf train-clean-100.tar.gz        # -> /mnt/storage/data/LibriTTS/train-clean-100/...
rm train-clean-100.tar.gz

# --- VCTK 0.92 (~11 GB) ---
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip -q VCTK-Corpus-0.92.zip -d VCTK-Corpus-0.92
rm VCTK-Corpus-0.92.zip
# VCTK 0.92 ships two mics per utterance — drop mic2 to avoid near-duplicate data:
find VCTK-Corpus-0.92 -name '*_mic2.flac' -delete
```

Resulting roots (used in the training commands):
- `--vctk_root /mnt/storage/data/VCTK-Corpus-0.92`
- `--libritts_root /mnt/storage/data/LibriTTS --librispeech_subsets train-clean-100`

> **HDD note:** datasets on the HDD are correct here, but random reads of many small
> files are slower than SSD. The `--num_workers 8` in the commands prefetches enough to
> keep the GPU fed. If you ever see low GPU utilization, raise `--num_workers` to 12–16;
> only if it's still starved, copy just `VCTK-Corpus-0.92` to the SSD.

> **Alternative (use your local copies instead of downloading):** from 🖥️ PC:
> `tar czf vctk.tgz VCTK-Corpus && scp -P $PORT vctk.tgz $HOST:/mnt/storage/data/`
> then ☁️ `cd /mnt/storage/data && tar xzf vctk.tgz && rm vctk.tgz`. (Your local
> LibriTTS is only `test-clean`, so still download `train-clean-100` as above.)

---

## Step 4 — Sanity checks before training  ☁️ CLOUD

```bash
# ☁️ CLOUD
source ~/rano_env/bin/activate
cd ~/rano-implementation

# (a) datasets load and speakers look right
python - <<'PY'
from data import build_dataset
ds = build_dataset(vctk_root="/mnt/storage/data/VCTK-Corpus-0.92",
                   libritts_root="/mnt/storage/data/LibriTTS",
                   split="train", libritts_subsets=["train-clean-100"])
print("combined training samples:", len(ds))
PY

# (b) build a small multi-speaker set for monitoring (from LibriTTS)
python build_eval_set.py --src /mnt/storage/data/LibriTTS/train-clean-100 \
    --out test_multi --num_speakers 20 --per_speaker 6
```

---

## Step 5 — Train, in tmux (survives SSH drops)  ☁️ CLOUD

Start a persistent session so training keeps running if your SSH disconnects:
```bash
# ☁️ CLOUD
tmux new -s rano
source ~/rano_env/bin/activate
cd ~/rano-implementation
```
Detach anytime with `Ctrl-b d`; reattach later with `tmux attach -t rano`.

Run the three stages **in order** (each needs the previous output). Because the
ASV/ACG are retrained, **Stage C is from scratch — no `--init_from`.**

**Stage A — ASV speaker encoder (VCTK, ~109 speakers):**
```bash
python train_asv.py --vctk_root /mnt/storage/data/VCTK-Corpus-0.92 \
  --output checkpoints/asv.pt \
  --batch_size 128 --epochs 50 --num_workers 8 --amp
```

**Stage B — ACG (VCTK + LibriTTS; needs the new asv.pt):**
```bash
python train_stage1.py \
  --vctk_root /mnt/storage/data/VCTK-Corpus-0.92 \
  --libritts_root /mnt/storage/data/LibriTTS --librispeech_subsets train-clean-100 \
  --asv_checkpoint checkpoints/asv.pt \
  --output_dir checkpoints/acg \
  --iterations 100000 --batch_size 64 --num_workers 8
```

**Stage C — Anonymizer, from scratch (needs new asv.pt + acg_final.pt):**
```bash
python train_stage2.py \
  --vctk_root /mnt/storage/data/VCTK-Corpus-0.92 \
  --libritts_root /mnt/storage/data/LibriTTS --librispeech_subsets train-clean-100 \
  --acg_checkpoint checkpoints/acg/acg_final.pt \
  --asv_checkpoint checkpoints/asv.pt \
  --output_dir /mnt/storage/checkpoints/rano_full \
  --log_dir    /mnt/storage/logs/rano_full \
  --batch_size 96 --iterations 200000 --lr 1e-5 \
  --lambda1 1 --lambda2 5 --lambda_logdet 0.01 \
  --distance_threshold 0.5 --margin 0.3 \
  --val_every 1000 --num_workers 8 --amp
```

Sizes are tuned for 48 GB: ASV batch 128, ACG batch 64, anonymizer **batch 96
(~36 GB peak — do NOT use 128, it OOMs)**. Rough time on RTX 6000 Ada: A a few hours,
B a few hours, C ~1–2 days.

> **Reusing the old LibriSpeech ASV/ACG instead (cheaper warm-start)?** Then skip
> Stages A/B, `scp` your old `checkpoints/` up, and add
> `--init_from checkpoints/rano/anonymizer_final.pt --triplet_check_step 4000` to the
> Stage-C command with `--iterations 60000`.

---

## Step 6 — Monitor  ☁️ CLOUD + 🖥️ PC

**TensorBoard** — on ☁️ CLOUD:
```bash
source ~/rano_env/bin/activate
tensorboard --logdir /mnt/storage/logs/rano_full --port 6006
```
Then on 🖥️ PC forward the port and open `http://localhost:6006`:
```bash
ssh -p $PORT -L 6006:localhost:6006 $HOST
```
Watch **`val/triplet` fall** (that's the cINN starting to anonymize) and
`val/consistency` stay low. No `[EXPLOSION ABORT]`.

**Truth-serum (vocoder-free) at any checkpoint** — ☁️ CLOUD:
```bash
python check_anon.py \
  --anonymizer_ckpt /mnt/storage/checkpoints/rano_full/anonymizer_step20000.pt \
  --acg_checkpoint checkpoints/acg/acg_final.pt --asv_checkpoint checkpoints/asv.pt \
  --test_dir test_multi
# want cos(ASV(xa),ASV(x)) dropping from 1.000 toward 0
```

---

## Step 7 — If a run crashes or the machine is preempted

Just **re-run the exact same command** in tmux. Stage B and Stage C write a
`training_state.pt` and **auto-resume from the last checkpoint** — nothing else to do.
(Stage A is epoch-based; if it dies, restart it — it's short.)

---

## Troubleshooting

- **`CUDA out of memory`** → lower `--batch_size` (anonymizer 96 → 64).
- **GPU sits idle / slow** → HDD I/O bound; raise `--num_workers` to 12–16, or copy
  `VCTK-Corpus-0.92` to the SSD.
- **`No usable samples`** → wrong `--libritts_root`/subset; the root must contain
  `train-clean-100/<speaker>/<chapter>/`.
- **`command not found: python`** → you forgot `source ~/rano_env/bin/activate`.
- **`ASV: provide exactly ONE of --vctk_root/--librispeech_root`** → give the ASV a
  single corpus (VCTK), not both.
- **Final EER looks off** → the pseudo-inverse vocoder is unreliable for EER; use
  `check_anon.py`'s cosine as the real anonymization signal, and a matched neural
  vocoder (BigVGAN/HiFi-GAN) for final numbers.


$Server = "cse@103.82.172.195"; $Port = 2222
python train_stage2.py --vctk_root /mnt/storage/data/VCTK-Corpus-0.92 \
  --libritts_root /mnt/storage/data/LibriTTS --librispeech_subsets train-clean-100 \
  --asv_checkpoint checkpoints/asv_ecapa.pt \
  --acg_checkpoint checkpoints/acg_ecapa/acg_final.pt \
  --output_dir /mnt/storage/checkpoints/rano_ecapa --log_dir /mnt/storage/logs/rano_ecapa \
  --batch_size 128 --iterations 200000 --lr 1e-5 \
  --lambda1 1 --lambda2 5 --lambda_logdet 0.01 --lambda_anchor 0.2 --lambda_range 1.0 \
  --val_every 1000 --num_workers 8 --amp