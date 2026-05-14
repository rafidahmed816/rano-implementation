# Full evaluation — all Table I + Table III metrics (paper §IV-B)
```bash
.venv/bin/python3 evaluate.py \
    --acg_checkpoint checkpoints/acg/acg_best.pt \
    --anonymizer_ckpt checkpoints/rano/anonymizer_final.pt \
    --test_dir data/test/ \
    --compute_wer \
    --whisper_model large \
    --eval_restoration \
    --embed_dim 256 \
    --num_cinn_blocks 12 \
    --num_acg_blocks 8 \
    --output_json results/eval_results.json
```

