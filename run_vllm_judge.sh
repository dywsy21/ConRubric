#!/usr/bin/env bash
set -euo pipefail
# ═══════════════════════════════════════════════════════════════
# run_vllm_judge.sh — Launch vLLM server for solver & oracle
#
# Serves the frozen model used by MetaRewardFunction for:
#   - Solver: generates answers given rubrics
#   - Oracle/Judge: scores answers against rubrics
#
# Usage:
#   ./run_vllm_judge.sh              # defaults
#   GPU=1 PORT=8202 ./run_vllm_judge.sh
# ═══════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

# Load .env
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

# Configurable via env
GPU="${JUDGE_GPU:-1}"
PORT="${JUDGE_PORT:-8202}"
MODEL="${JUDGE_MODEL:-models/Qwen3-4B}"
MODEL_NAME="${JUDGE_SERVED_NAME:-qwen3-4b}"
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${JUDGE_GPU_MEM_UTIL:-0.9}"

echo "[vllm-judge] GPU=$GPU  PORT=$PORT  MODEL=$MODEL  NAME=$MODEL_NAME"
echo "[vllm-judge] max_model_len=$MAX_MODEL_LEN  gpu_memory_utilization=$GPU_MEM_UTIL"

CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 \
  2>&1 | tee vllm-judge.log
