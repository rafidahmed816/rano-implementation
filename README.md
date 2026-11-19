# The Content Leash: Mitigating Degenerate Solutions in Invertible Speaker Anonymization

**Nirjon Mobarak Talukder, Rafid Ahmed, Mohammad Ishrak Abedin, Abu Raihan Mostofa Kamal**
Department of Computer Science and Engineering, Islamic University of Technology (IUT), Dhaka, Bangladesh

📄 Paper: [arXiv:2610.36768](https://arxiv.org/abs/2610.36768)

This repository holds the code for our paper. We re-implement **Rano** (Wang, Zhang and Qu, IJCNN 2025),
a conditional invertible neural network (cINN) that anonymizes a speaker's voice and can restore the
original exactly with a secret key. We then fix two problems we observed in our re-implementation.

> No official implementation of Rano is available. Everything here is our own re-implementation, and
> the observations below describe our configuration, not a defect of the published method.

---

## What the paper finds

1. **A degenerate solution.** With only the published losses (consistency + triplet), our model reached
   **41.6% EER at 117.6% WER**: the triplet loss was satisfied by destroying the spectrogram, not by
   changing the speaker.
2. **Content-preserving losses fix it.** An *anchor loss* `MSE(Xa, X)` and a *range loss* that keeps
   `Xa` inside the original utterance's value range bring WER down to **8.3%** under matched conditions.
3. **Verifier transfer.** An anonymizer trained against a lightweight mel-domain verifier does not fool
   an independent verifier. Distilling **ECAPA-TDNN** into a mel-domain student verifier is worth
   **+21 EER points** at matched settings.

Full objective:

```
L = λ1·Lcons + λ2·Ltri + λanchor·Lanchor + λrange·Lrange + λdet·Ldet
```

`Ldet` is a log-determinant regularizer that keeps the inverse pass numerically well conditioned.

## Main results

38 **unseen** speakers, 303 utterances from LibriTTS `test-clean`, HiFi-GAN fine-tuned on VCTK only.
EER is the mean ± standard deviation of three seeded runs.

| Metric | Floor (clean speech) | Ours | McAdams baseline |
|---|---|---|---|
| EER, ECAPA-TDNN (%) ↑ | 2.83 | **26.31 ± 0.25** | 8.44 |
| EER, held-out ResNet (%) ↑ | 2.08 | **27.86** | 6.37 |
| WER, Whisper large-v3 (%) ↓ | 2.48 | 11.75 | 3.61 |
| Restoration Sim<sub>spk</sub> (%) ↑ | 94.90 | **94.90** | n/a |

- Restoration with the correct key matches the vocoder round-trip floor, so the inverse pass itself costs
  nothing measurable.
- With a **wrong key**, restored speaker similarity stays flat at about **4%** at every key distance
  tested, so an attacker gets no gradient to follow.
- The attacker is the *ignorant attacker* of the VoicePrivacy challenges. Informed attackers are not
  tested. See the paper's Limitations section.

---

## Pipeline

```
key k ~ N(0, I) ──► ACG ──► pseudo-speaker condition c
                                   │
original mel X ──► cINN forward fθ(X; c) ──► anonymized mel Xa ──► HiFi-GAN ──► anonymized audio
                                   │
anonymized mel Xa ──► cINN inverse f⁻¹θ(Xa; c) ──► restored mel ≈ X   (same weights, same key)
```

Training runs in four stages:

| Stage | Script | Output |
|---|---|---|
| 1. Distill ECAPA-TDNN into a mel-domain verifier `Vmel` | `distill_ecapa.py` | `asv_ecapa.pt` |
| 2. Train the ACG in `Vmel`'s embedding space | `train_stage1.py` | `acg_final.pt` |
| 3. Train the cINN anonymizer | `train_stage2.py` | `anonymizer_step*.pt` |
| 4. Fine-tune HiFi-GAN on our mel configuration | `train_vocoder.py` | `generator.pt` |

`train_asv.py` trains the lightweight verifier `Vlight` used by the original design. It is only needed to
reproduce the verifier-transfer comparison.

---

## Repository layout

| File | Purpose |
|---|---|
| `blocks.py`, `anonymizer.py`, `acg.py`, `model.py` | cINN coupling blocks, anonymizer/restorer, ACG |
| `speaker_encoder.py` | Mel-domain speaker encoder (`Vlight` / `Vmel`) |
| `loss.py` | Consistency, triplet, anchor, range and log-det losses |
| `audio.py`, `data.py` | Mel extraction (80 bands, 22.05 kHz, hop 256, win 1024); VCTK/LibriTTS loaders |
| `hifigan.py`, `bigvgan.py` | Vocoders |
| `distill_ecapa.py`, `train_asv.py`, `train_stage1.py`, `train_stage2.py`, `train_vocoder.py` | Training |
| `evaluate3.py` | Main evaluation: EER, Sim<sub>spk</sub>, optional WER |
| `wer_from_audio.py` | WER with Whisper large-v3 as a separate pass (fits an 8 GB GPU) |
| `heldout_resnet.py`, `heldout_resemblyzer.py` | Held-out attackers never used in training |
| `security_wrongkey.py` | Wrong-key restoration experiment |
| `baseline_mcadams.py` | McAdams-coefficient baseline |
| `build_libritts_eval.py`, `build_vctk_eval.py`, `build_eval_set.py` | Build evaluation sets with transcripts |
| `check_anon.py`, `check_intel.py` | Quick diagnostics: identity shift without a vocoder; garbling check |
| `infer.py` | Command-line anonymize / restore |
| `splits/test_unseen.txt` | The 303 evaluation utterance IDs |
| `RUN_ON_SSH.md` | Step-by-step guide for training on a remote GPU server |

Older scripts (`evaluate.py`, `evaluate2.py`, `condSwapTest.py`, `debug*.py`, `diagnose_vocoder.py`) are
kept for the record of how we found the bugs. Use `evaluate3.py` for results.

---

## Setup

Python 3.10+ and a CUDA GPU. We trained on one RTX 6000 Ada (48 GB) and evaluated on an RTX 4060 (8 GB).

```bash
git clone https://github.com/rafidahmed816/rano-implementation.git
cd rano-implementation
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

SpeechBrain models (ECAPA-TDNN, ResNet) are downloaded automatically on first use.

### Data

- [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443)
- [LibriTTS](https://www.openslr.org/60/): `train-clean-100` for training, `test-clean` for evaluation
- [HiFi-GAN UNIVERSAL_V1](https://github.com/jik876/hifi-gan) weights for vocoder fine-tuning, placed in
  `pretrained/UNIVERSAL_V1/`

Training uses VCTK + LibriTTS `train-clean-100`, split 85/15 at utterance level. Evaluation speakers in
`splits/test_unseen.txt` are disjoint from all training data.

### Checkpoints

Checkpoints are not stored in git.
<!-- TODO: add the link where the trained checkpoints are published (e.g. a GitHub Release). -->

---

## Training

Paths below are examples; set `--vctk_root` / `--libritts_root` to your data.

**1. Distill ECAPA-TDNN into `Vmel`** (30k steps)

```bash
python distill_ecapa.py --vctk_root data/VCTK-Corpus-0.92 \
  --libritts_root data/LibriTTS --libritts_subset train-clean-100 \
  --output checkpoints/asv_ecapa.pt --steps 30000 --batch_size 64 --lambda_rel 1.0
```

**2. Train the ACG** (80k steps)

```bash
python train_stage1.py --vctk_root data/VCTK-Corpus-0.92 \
  --libritts_root data/LibriTTS --librispeech_subsets train-clean-100 \
  --asv_checkpoint checkpoints/asv_ecapa.pt \
  --output_dir checkpoints/acg_ecapa --iterations 80000
```

**3a. Train the anonymizer from scratch** (52k steps, λanchor = 0.2, λ2 = 5)

```bash
python train_stage2.py --vctk_root data/VCTK-Corpus-0.92 \
  --libritts_root data/LibriTTS --librispeech_subsets train-clean-100 \
  --asv_checkpoint checkpoints/asv_ecapa.pt \
  --acg_checkpoint checkpoints/acg_ecapa/acg_final.pt \
  --output_dir checkpoints/rano_ecapa \
  --lambda1 1 --lambda2 5 --lambda_anchor 0.2 --lambda_range 1.0 --lambda_logdet 0.01 \
  --margin 0.3 --lr 1e-5 --batch_size 128 --iterations 52000
```

**3b. Warm-started stage with the final weights** (λ2 = 3, λanchor = 1.0). The reported system is step 8k.

```bash
python train_stage2.py <same data / verifier / ACG flags as 3a> \
  --init_from checkpoints/rano_ecapa/anonymizer_step52000.pt \
  --output_dir checkpoints/rano_ecapa_bal \
  --lambda1 1 --lambda2 3 --lambda_anchor 1.0 --lambda_range 1.0 --lambda_logdet 0.01 \
  --margin 0.3 --lr 1e-5 --batch_size 128 --iterations 12000
```

**4. Fine-tune HiFi-GAN on VCTK only** (30k steps)

```bash
python train_vocoder.py --vctk_root data/VCTK-Corpus-0.92 \
  --pretrained_g pretrained/UNIVERSAL_V1/g_02500000 \
  --pretrained_do pretrained/UNIVERSAL_V1/do_02500000 \
  --output_dir checkpoints/vocoder_vctk --steps 30000
```

Batch 128 fits a 48 GB GPU; lower `--batch_size` if you run out of memory. For a remote GPU server (tmux, venv, data on a separate disk), see `RUN_ON_SSH.md`.

---

## Evaluation

```bash
# EER (ECAPA-TDNN) + restoration similarity; run once per seed (0, 1, 2)
python evaluate3.py --test_dir test_unseen \
  --acg_checkpoint checkpoints/acg_ecapa/acg_final.pt \
  --anonymizer_ckpt checkpoints/rano_ecapa_bal/anonymizer_step8000.pt \
  --vocoder hifigan_custom --vocoder_ckpt checkpoints/vocoder_vctk/generator.pt \
  --seed 0 --save_audio_dir eval_unseen_seed0

# WER with Whisper large-v3 on the saved audio
python wer_from_audio.py --audio_dir eval_unseen_seed0 --test_dir test_unseen

# Held-out attacker, wrong-key test, McAdams baseline
python heldout_resnet.py          # edit the paths at the top of the script
python security_wrongkey.py --test_dir test_unseen
python baseline_mcadams.py --test_dir test_unseen --out_dir eval_mcadams
```

`test_unseen/` holds the utterances in `splits/test_unseen.txt`, copied from LibriTTS `test-clean`
into `<speaker>/` folders with a LibriSpeech-style `<speaker>.trans.txt` per speaker
(the same layout `build_libritts_eval.py` writes).

### Anonymize or restore your own audio

```bash
python infer.py anonymize --input audio/ --output anon/ --checkpoint <anonymizer.pt> --acg_checkpoint <acg.pt>
python infer.py restore --input anon/ --output restored/ --checkpoint <anonymizer.pt> --acg_checkpoint <acg.pt> --key_file anon/keys.json
```

Keep `keys.json` secret: anyone holding it can restore the original voice.

---

## Citation

If you use this code, please cite our paper and the original Rano paper.

```bibtex
@misc{talukder2026contentleash,
  title         = {The Content Leash: Mitigating Degenerate Solutions in Invertible Speaker Anonymization},
  author        = {Talukder, Nirjon Mobarak and Ahmed, Rafid and Abedin, Mohammad Ishrak and Kamal, Abu Raihan Mostofa},
  year          = {2026},
  eprint        = {2610.36768},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.36768}
}

@inproceedings{wang2025rano,
  title     = {Rano: Restorable Speaker Anonymization via Conditional Invertible Neural Network},
  author    = {Wang, Jianzong and Zhang, Xulong and Qu, Xiaoyang},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025},
  doi       = {10.1109/IJCNN64981.2025.11227971}
}
```
