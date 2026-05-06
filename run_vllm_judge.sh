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

# Use dedicated judge venv with vLLM 0.16+ (supports Qwen3.5)
JUDGE_VENV="${JUDGE_VENV:-$ROOT_DIR/.venv_judge}"
if [[ -d "$JUDGE_VENV" ]]; then
  source "$JUDGE_VENV/bin/activate"
else
  source "$ROOT_DIR/.venv/bin/activate"
fi

# Load .env
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

# Configurable via env
GPU="${JUDGE_GPU:-1}"
PORT="${JUDGE_PORT:-8202}"
MODEL="${JUDGE_MODEL:-models/Qwen3.5-35B-A3B}"
MODEL_NAME="${JUDGE_SERVED_NAME:-qwen3.5-35b-a3b}"
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${JUDGE_GPU_MEM_UTIL:-0.9}"
LANG_ONLY="${JUDGE_LANG_ONLY:-1}"  # Qwen3.5 is multimodal; use --language-model-only for text

echo "[vllm-judge] GPU=$GPU  PORT=$PORT  MODEL=$MODEL  NAME=$MODEL_NAME"
echo "[vllm-judge] max_model_len=$MAX_MODEL_LEN  gpu_memory_utilization=$GPU_MEM_UTIL  lang_only=$LANG_ONLY"

EXTRA_ARGS=()
# --language-model-only was removed in vLLM 0.7+; skip it
# Qwen3.5-35B-A3B declares Qwen3_5MoeForConditionalGeneration but we use text-only;
# override to the vLLM-supported Qwen3MoeForCausalLM architecture
EXTRA_ARGS+=(--hf-overrides '{"architectures": ["Qwen3MoeForCausalLM"]}')

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee vllm-judge.log
