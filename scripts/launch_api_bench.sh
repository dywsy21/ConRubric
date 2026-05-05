#!/usr/bin/env bash
# Launch script for frontier API model benchmark on server.
# Run this on autodl2 inside a tmux session:
#
#   tmux new -s bench
#   cd /root/autodl-tmp/GRM
#   bash scripts/launch_api_bench.sh
#
# It will:
#   1. Start vLLM judge (qwen3.5-35b-a3b) in background tmux pane
#   2. Wait for it to be ready
#   3. Run bench_api_models.py for all 5 frontier models

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── Activate venv ────────────────────────────────────────────────
VENV_JUDGE="$ROOT/.venv_judge"
if [[ -d "$VENV_JUDGE" ]]; then
  source "$VENV_JUDGE/bin/activate"
else
  source "$ROOT/.venv/bin/activate"
fi

# ── Load .env ────────────────────────────────────────────────────
set -a; source "$ROOT/.env"; set +a

# ── Config ───────────────────────────────────────────────────────
JUDGE_PORT="${JUDGE_PORT:-8202}"
JUDGE_GPU="${JUDGE_GPU:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-models/Qwen3.5-35B-A3B}"
JUDGE_MODEL_NAME="${JUDGE_SERVED_NAME:-qwen3.5-35b-a3b}"
JUDGE_MEM="${JUDGE_GPU_MEM_UTIL:-0.85}"

EXT_API_KEY="${EXT_API_KEY:-sk-Dk3063EgG7Lezr9BJ3nkpgtGsv89KR4CKrODFLB0lgqr0E4d}"
export EXT_API_KEY

# Override .env placeholders for local vLLM
export ORACLE_MODEL_NAME="$JUDGE_MODEL_NAME"
export ORACLE_API_KEY="sk-local-vllm-noauth"
export ORACLE_API_BASE="http://localhost:${JUDGE_PORT}/v1"

# ── Start vLLM judge if not already running ──────────────────────
if curl -sf --max-time 3 "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1; then
  echo "[launch] vLLM judge already running on port $JUDGE_PORT"
else
  echo "[launch] Starting SGLang judge on GPU $JUDGE_GPU, port $JUDGE_PORT..."

  SGLANG_DISABLE_CUDNN_CHECK=1 CUDA_VISIBLE_DEVICES="$JUDGE_GPU" python -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" \
    --port "$JUDGE_PORT" \
    --tp-size 1 \
    --mem-fraction-static "$JUDGE_MEM" \
    --context-length 16384 \
    --served-model-name "$JUDGE_MODEL_NAME" \
    --trust-remote-code \
    --attention-backend triton \
    > "$ROOT/sglang-judge.log" 2>&1 &

  JUDGE_PID=$!
  echo "[launch] SGLang judge PID=$JUDGE_PID (log: $ROOT/sglang-judge.log)"

  # Wait up to 5 minutes for vLLM to be ready
  echo "[launch] Waiting for vLLM to be ready..."
  for i in $(seq 1 60); do
    if curl -sf --max-time 3 "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1; then
      echo "[launch] vLLM judge ready after ${i}×5s"
      break
    fi
    if ! kill -0 $JUDGE_PID 2>/dev/null; then
      echo "[launch] ERROR: SGLang process died! Last 20 lines of log:"
      tail -20 "$ROOT/sglang-judge.log"
      exit 1
    fi
    echo "[launch] Waiting... (${i}/60)"
    sleep 5
  done

  if ! curl -sf --max-time 3 "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1; then
    echo "[launch] ERROR: vLLM did not start within 5 minutes"
    exit 1
  fi
fi

echo ""
echo "[launch] Judge: $JUDGE_MODEL_NAME @ http://localhost:${JUDGE_PORT}/v1"
echo "[launch] Starting benchmark for all 5 frontier models..."
echo ""

# ── Run benchmark ────────────────────────────────────────────────
# Use main project venv for benchmark (has scipy, etc.)
source "$ROOT/.venv/bin/activate"

python scripts/bench_api_models.py \
  --judge_workers 40 \
  --gen_workers 20 \
  2>&1 | tee out/bench/api_models/bench_run.log

echo ""
echo "[launch] Done. Results: out/bench/api_models/"
