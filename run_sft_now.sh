#!/bin/bash
set -e
cd /ssd/GRM
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/ssd/GRM/.venv/lib/python3.10/site-packages/cusparselt/lib:/ssd/GRM/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

python -m src.training.sft_trainer \
  --no-healthbench \
  --synthetic-path data/rubrichub_sft.jsonl \
  --max-length 2048 \
  --train-batch-size 64 \
  --micro-batch-size-per-gpu 2 \
  --epochs 2 \
  --lr 1e-5 \
  --nproc-per-node 8 \
  --project-name grm-sft \
  --experiment-name rubrichub_qwen3_4b \
  --output-dir out/sft \
  --save-freq 500 \
  --test-freq -1 \
  --preview-samples 2
