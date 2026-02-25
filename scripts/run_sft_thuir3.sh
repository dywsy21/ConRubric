#!/bin/bash
# SFT on RubricHub full data — thuir3 server (8x A100-40GB)
set -euo pipefail

WORK=/data-share/yeesuanAI08/lihaitao/wsy/GRM
cd "$WORK"
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH

# Proxy for HuggingFace model downloads
export ALL_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export HTTP_PROXY=http://127.0.0.1:7897
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

# ── Step 1: Download RubricHub and convert to JSONL (if not done) ──
if [ ! -f data/rubrichub_sft.jsonl ]; then
    echo "=== Downloading RubricHub from HuggingFace ==="
    mkdir -p data/rubrichub_raw
    python -c "
from datasets import load_dataset
ds = load_dataset('sojuL/RubricHub_v1', split='train')
ds.to_parquet('data/rubrichub_raw/RuRL/rubrichub.parquet')
print(f'Downloaded {len(ds)} rows')
"
    echo "=== Converting to SFT JSONL ==="
    python -m src.data.prepare_rubrichub \
        --input-dir data/rubrichub_raw/RuRL \
        --output data/rubrichub_sft.jsonl
fi

echo "=== Data ready ==="
wc -l data/rubrichub_sft.jsonl

# ── Step 2: Launch SFT training ──
# Qwen3-4B, cosine LR 5e-5 → ~5e-6 with 5% warmup, 2 epochs
# 8 GPU × micro_batch=2, global_batch=64
python -m src.training.sft_trainer \
    --no-healthbench \
    --synthetic-path data/rubrichub_sft.jsonl \
    --max-length 2048 \
    --train-batch-size 64 \
    --micro-batch-size-per-gpu 2 \
    --epochs 2 \
    --lr 5e-5 \
    --lr-scheduler-type cosine \
    --lr-warmup-ratio 0.05 \
    --min-lr-ratio 0.1 \
    --nproc-per-node 8 \
    --project-name grm-sft \
    --experiment-name rubrichub_qwen3_4b_cosine \
    --output-dir out/sft \
    --save-freq 500 \
    --test-freq -1 \
    --preview-samples 2
