# Rano Training on RunPod — A100 80GB Guide

Complete guide to train the Rano model from scratch on RunPod with an A100 80GB GPU.

**Dataset**: LibriSpeech `train-clean-360` (~360h, ~921 speakers)  
**GPU**: NVIDIA A100 80GB HBM2e  
**Estimated total training time**: ~20–30 hours (all 3 stages)

---

## 1. Pod Setup

### 1.1 Create Pod

- **GPU**: NVIDIA A100 80GB
- **Template**: `RunPod PyTorch 2.x` (comes with CUDA 12.x + PyTorch pre-installed)
- **Disk**: 100GB+ (dataset is ~23GB compressed, ~70GB extracted)
- **Volume**: Mount persistent volume at `/workspace` (your checkpoints survive pod restarts)

### 1.2 Connect to Pod

Use the Web Terminal or SSH into the pod.

```bash
# Verify GPU
nvidia-smi
# Should show: A100-SXM4-80GB or A100-PCIE-80GB

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}, BF16: {torch.cuda.is_bf16_supported()}')"
```

---

## 2. Clone Repo & Install Dependencies

```bash
cd /workspace

# Clone your repo
git clone https://github.com/rafidahmed816/rano-implementation.git thesis
cd thesis
git checkout 5070ti_

# Install dependencies
pip install -r requirements.txt

# Verify imports work
python -c "from model import Rano; print('Rano model OK')"
```

---

## 3. Download LibriSpeech train-clean-360

```bash
# Create data directory
mkdir -p /workspace/data
cd /workspace/data

# Download train-clean-360 (~23GB)
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz

# Extract (~70GB when extracted)
tar -xzf train-clean-360.tar.gz

# This creates: /workspace/data/LibriSpeech/train-clean-360/
# Verify structure
ls /workspace/data/LibriSpeech/train-clean-360/ | head -20
# Should show speaker ID directories: 100, 1001, 1006, ...

# Count speakers
echo "Speakers: $(ls /workspace/data/LibriSpeech/train-clean-360/ | wc -l)"
# Expected: ~921

# Optional: remove the archive to save disk space
rm train-clean-360.tar.gz

cd /workspace/thesis
```

### Validate Dataset (Optional but Recommended)

```bash
python validate_dataset.py \
    --root /workspace/data/LibriSpeech \
    --subsets train-clean-360
```

---

## 4. Google Drive Backup Setup

### 4.1 Install rclone

```bash
curl https://rclone.org/install.sh | bash
```

### 4.2 Configure Google Drive

```bash
rclone config
```

When prompted:
1. `n` → New remote
2. Name: `gdrive`
3. Storage type: `drive` (Google Drive)
4. Client ID: leave blank (press Enter)
5. Client Secret: leave blank (press Enter)
6. Scope: `1` (Full access)
7. Root folder ID: leave blank
8. Service account file: leave blank
9. Advanced config: `n`
10. Auto config: `n` (since you're on a remote machine)
11. It will give you a URL — open it in your browser, authorize, paste the code back

> **Alternative (headless setup)**: If the interactive auth is difficult on RunPod, configure rclone on your local machine first, then copy `~/.config/rclone/rclone.conf` to the pod.

### 4.3 Test Google Drive Connection

```bash
# List root of your Google Drive
rclone lsd gdrive:

# Create a backup folder
rclone mkdir gdrive:rano-checkpoints
```

### 4.4 Backup Script

Create a script that syncs checkpoints to Google Drive. Save as `/workspace/thesis/backup_to_gdrive.sh`:

```bash
cat > /workspace/thesis/backup_to_gdrive.sh << 'EOF'
#!/bin/bash
# Sync checkpoints to Google Drive
# Usage: ./backup_to_gdrive.sh
# Or run in background: nohup ./backup_to_gdrive.sh &

CKPT_DIR="/workspace/thesis/checkpoints"
GDRIVE_DIR="gdrive:rano-checkpoints"

echo "[$(date)] Starting backup..."
rclone sync "$CKPT_DIR" "$GDRIVE_DIR" \
    --progress \
    --transfers 4 \
    --checkers 8 \
    --log-file /workspace/thesis/logs/rclone.log
echo "[$(date)] Backup complete."
EOF
chmod +x /workspace/thesis/backup_to_gdrive.sh
```

### 4.5 Auto-Backup with Cron (Every 30 Minutes)

```bash
# Open crontab
crontab -e

# Add this line (syncs every 30 minutes):
*/30 * * * * /workspace/thesis/backup_to_gdrive.sh >> /workspace/thesis/logs/cron_backup.log 2>&1
```

---

## 5. Training Commands

> **IMPORTANT**: Run all commands from `/workspace/thesis`

```bash
cd /workspace/thesis
mkdir -p checkpoints logs
```

### 5.1 Step 1: Train ASV Speaker Encoder

The ASV provides speaker embeddings for the triplet loss. Must be trained first.

```bash
python train_asv.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --output checkpoints/asv.pt \
    --batch_size 256 \
    --epochs 50 \
    --lr 2e-3 \
    --num_workers 8 \
    --amp
```

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | 256 | Tiny model (~0.5M params), A100 can handle easily |
| `lr` | 2e-3 | Scaled from 1e-3 with √(256/32) ≈ 2.8× for larger batch |
| `epochs` | 50 | Paper spec. Early stopping (patience=10) handles convergence |
| `num_workers` | 8 | Match RunPod vCPU count |

**Estimated time**: ~30–60 minutes  
**Output**: `checkpoints/asv.pt`

After completion, **backup immediately**:
```bash
./backup_to_gdrive.sh
```

---

### 5.2 Step 2: Train ACG (Stage 1)

Pre-trains the Anonymization Condition Generator.

```bash
python train_stage1.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --asv_checkpoint checkpoints/asv.pt \
    --output_dir checkpoints/acg \
    --batch_size 256 \
    --iterations 100000 \
    --num_workers 12
```

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | 512 | ACG is tiny (~2M params), max batch for throughput |
| `iterations` | 100,000 | Paper spec (§7) |
| `lr` | 1e-5 | Paper spec — don't change |
| `num_workers` | 8 | Match vCPU count |

**Estimated time**: ~1–2 hours  
**Output**: `checkpoints/acg/acg_final.pt`, `checkpoints/acg/acg_best.pt`

After completion, **backup**:
```bash
./backup_to_gdrive.sh
```

---

### 5.3 Step 3: Train Rano Anonymizer (Stage 2) — THE BIG ONE

This is the main training loop. Takes the longest.

```bash
python train_stage2.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --acg_checkpoint checkpoints/acg/acg_final.pt \
    --asv_checkpoint checkpoints/asv.pt \
    --output_dir checkpoints/rano \
    --log_dir logs/rano \
    --batch_size 128 \
    --accumulate_steps 1 \
    --iterations 200000 \
    --lr 1e-5 \
    --val_every 500 \
    --num_workers 14 \
    --amp \
    --compile
```

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | **128** | A100 80GB with BF16 AMP should fit. training_step processes 2×128=256 internally |
| `accumulate_steps` | 1 | bs=128 is already large enough — no accumulation needed |
| `iterations` | 200,000 | Paper spec (§7) |
| `lr` | 1e-5 | Paper spec (§7) — do NOT change |
| `--amp` | enabled | A100 auto-selects BF16 (code detects `bf16_supported()`) |
| `--compile` | enabled | torch.compile kernel fusion — first step slow (~5min), rest ~1.3× faster |
| `num_workers` | 8 | Match vCPU count |

**Estimated time**: ~15–25 hours  
**Output**: `checkpoints/rano/anonymizer_final.pt`, `checkpoints/rano/rano_final.pt`

> ⚠️ **If batch_size=128 causes OOM**, reduce to **96**, then **64**. The memory bottleneck is the 12 cINN blocks with RRDB subnets — activations accumulate during backprop.

> 💡 **Monitor GPU memory**: Open a second terminal and run:
> ```bash
> watch -n 5 nvidia-smi
> ```

---

## 6. Monitoring Training

### TensorBoard

```bash
# In a second terminal
tensorboard --logdir /workspace/thesis/logs --bind_all --port 6006 &
```

Access TensorBoard via RunPod's port forwarding (port 6006).

### Training Progress

Stage 2 prints progress every 1000 steps:
```
step=1000  total=X.XXXX  cons=X.XXXX  tri=X.XXXX  logdet=X.XXXX  lr=1.00e-05  step_t=0.XXs  ETA=XXh
```

Checkpoints are saved every 1000 steps automatically.

---

## 7. Run Everything in Background (Recommended)

Use `nohup` + `screen`/`tmux` so training survives SSH disconnects:

```bash
# Option A: Using screen
screen -S training

# Then run the training command inside screen
# Detach: Ctrl+A, D
# Reattach: screen -r training

# Option B: Using nohup (simpler)
nohup python train_asv.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --output checkpoints/asv.pt \
    --batch_size 256 --epochs 50 --lr 2e-3 \
    --num_workers 8 --amp \
    > logs/asv_training.log 2>&1 &

# Check progress
tail -f logs/asv_training.log
```

### Full Pipeline Script (Run All 3 Stages Sequentially)

```bash
cat > /workspace/thesis/train_all.sh << 'EOF'
#!/bin/bash
set -e
cd /workspace/thesis

echo "=========================================="
echo " STAGE 0: Training ASV Speaker Encoder"
echo "=========================================="
python train_asv.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --output checkpoints/asv.pt \
    --batch_size 256 --epochs 50 --lr 2e-3 \
    --num_workers 8 --amp

echo "[BACKUP] Syncing ASV checkpoint to Google Drive..."
./backup_to_gdrive.sh

echo "=========================================="
echo " STAGE 1: Training ACG"
echo "=========================================="
python train_stage1.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --asv_checkpoint checkpoints/asv.pt \
    --output_dir checkpoints/acg \
    --batch_size 512 --iterations 100000 \
    --num_workers 8

echo "[BACKUP] Syncing ACG checkpoint to Google Drive..."
./backup_to_gdrive.sh

echo "=========================================="
echo " STAGE 2: Training Rano Anonymizer"
echo "=========================================="
python train_stage2.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --acg_checkpoint checkpoints/acg/acg_final.pt \
    --asv_checkpoint checkpoints/asv.pt \
    --output_dir checkpoints/rano \
    --log_dir logs/rano \
    --batch_size 128 --accumulate_steps 1 \
    --iterations 200000 --lr 1e-5 \
    --val_every 500 --num_workers 8 \
    --amp --compile

echo "[BACKUP] Final sync to Google Drive..."
./backup_to_gdrive.sh

echo "=========================================="
echo " ALL TRAINING COMPLETE!"
echo "=========================================="
EOF
chmod +x /workspace/thesis/train_all.sh
```

Then run:
```bash
nohup /workspace/thesis/train_all.sh > /workspace/thesis/logs/full_training.log 2>&1 &

# Monitor:
tail -f /workspace/thesis/logs/full_training.log
```

---

## 8. After Training — Download Checkpoints

### From Google Drive
Your checkpoints are already backed up to `gdrive:rano-checkpoints/` via the cron job.

### Direct Download from Pod
```bash
# Download specific files to your local machine (run on YOUR machine, not the pod)
scp -P <PORT> root@<POD_IP>:/workspace/thesis/checkpoints/rano/rano_final.pt ./
scp -P <PORT> root@<POD_IP>:/workspace/thesis/checkpoints/rano/anonymizer_final.pt ./
scp -P <PORT> root@<POD_IP>:/workspace/thesis/checkpoints/asv.pt ./
scp -P <PORT> root@<POD_IP>:/workspace/thesis/checkpoints/acg/acg_final.pt ./
```

### Final Manual Backup
```bash
# On the pod — one last sync
./backup_to_gdrive.sh

# Verify everything is there
rclone ls gdrive:rano-checkpoints/
```

---

## 9. Quick Reference — Parameter Summary

### Paper-Specified (DO NOT CHANGE)

| Parameter | Value | Paper Reference |
|---|---|---|
| `--lr` (Stage 2) | 1e-5 | §7 |
| `--lr_step` | 50,000 | §7 |
| `--lambda1` | 1.0 | Eq. 7 |
| `--lambda2` | 5.0 | Eq. 7 |
| `--margin` | 0.3 | §7 |
| `--distance_threshold` | 0.5 | §7 |
| `--iterations` (Stage 2) | 200,000 | §7 |
| `--num_cinn_blocks` | 12 | §2.2 |
| `--num_acg_blocks` | 8 | §2.1 |
| `--embed_dim` | 256 | §2 |

### Hardware-Tuned for A100 80GB

| Parameter | Value | Notes |
|---|---|---|
| `--batch_size` (ASV) | 256 | Tiny model, max throughput |
| `--batch_size` (Stage 1) | 512 | ACG is small, max throughput |
| `--batch_size` (Stage 2) | **128** | Doubled to 256 internally. OOM? Try 96 → 64 |
| `--num_workers` | 8 | Match RunPod vCPUs |
| `--amp` | always on | A100 uses BF16 automatically |
| `--compile` | Stage 2 only | torch.compile kernel fusion |

---

## 10. Troubleshooting

### OOM on Stage 2

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix**: Reduce `--batch_size` from 128 → 96 → 64. Add `--accumulate_steps 2` if using bs=64 to keep effective batch large.

### Dataset Not Found

```
ValueError: No usable LibriSpeech samples found
```

**Fix**: Verify the directory structure:
```bash
# Should show speaker directories (numbers)
ls /workspace/data/LibriSpeech/train-clean-360/ | head -5
# Should show chapter directories under each speaker
ls /workspace/data/LibriSpeech/train-clean-360/100/
```

### torch.compile Fails

```
RuntimeError: torch.compile is not supported...
```

**Fix**: Remove `--compile` flag. Some PyTorch versions have issues. Training still works, just ~20% slower.

### Pod Restarts / Crashes

Stage 2 **auto-resumes** from `checkpoints/rano/training_state.pt` if it exists. Just re-run the same command — it will pick up where it left off.

```bash
# After pod restart, just re-run:
python train_stage2.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --acg_checkpoint checkpoints/acg/acg_final.pt \
    --asv_checkpoint checkpoints/asv.pt \
    --output_dir checkpoints/rano \
    --log_dir logs/rano \
    --batch_size 128 --accumulate_steps 1 \
    --iterations 200000 --lr 1e-5 \
    --val_every 500 --num_workers 8 \
    --amp --compile
```

### Slow Data Loading

If GPU utilization is low (<90%) and `nvidia-smi` shows low GPU usage:
```bash
# Increase workers
--num_workers 12

# Or copy dataset to local NVMe (faster than network volume)
cp -r /workspace/data/LibriSpeech /tmp/LibriSpeech
# Then use --librispeech_root /tmp/LibriSpeech
```

python train_stage2.py \
    --librispeech_root /workspace/data/LibriSpeech \
    --librispeech_subsets train-clean-360 \
    --acg_checkpoint checkpoints/acg/acg_final.pt \
    --asv_checkpoint checkpoints/asv.pt \
    --output_dir checkpoints/rano \
    --log_dir logs/rano \
    --batch_size 128 \
    --accumulate_steps 1 \
    --iterations 200000 \
    --lr 1e-5 \
    --val_every 1000 \
    --num_workers 10 \
    --amp