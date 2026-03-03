#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# run_rl.sh — RL training pipeline with stage selection
#
# Usage:
#   ./run_rl.sh                        # Run all stages (train + bench)
#   ./run_rl.sh --stage train          # Only RL training
#   ./run_rl.sh --stage bench          # Only benchmark the RL model
#   ./run_rl.sh --stage train,bench    # Both (default)
#   ./run_rl.sh --limit 50            # Cap training samples
#   ./run_rl.sh -- key=value           # Pass extra Hydra overrides
# ═══════════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment
source "$ROOT_DIR/.venv/bin/activate"

# Load .env variables
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
else
  echo "[ERROR] .env not found at $ROOT_DIR/.env"
  exit 1
fi

# Disable proxies — models are local/LAN, no internet needed
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export HYDRA_FULL_ERROR=1

# ── Parse arguments ───────────────────────────────────────────────
STAGES=""
HYDRA_ARGS=()
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage|-s)
      STAGES="$2"; shift 2 ;;
    --limit)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --limit requires a value"; exit 1
      fi
      HYDRA_ARGS+=("data.train_max_samples=$2" "data.val_max_samples=$2")
      shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--stage STAGE[,STAGE,...]] [--limit N] [-- HYDRA_ARGS...]"
      echo ""
      echo "Stages:"
      echo "  train   Run RL training (default model: SFT checkpoint or BASE_MODEL)"
      echo "  bench   Benchmark the RL-trained model"
      echo ""
      echo "Default: run all stages in order."
      echo ""
      echo "The input model is resolved in order:"
      echo "  1. GRM_MODEL_NAME env var (if set to a local path)"
      echo "  2. Latest SFT checkpoint from SFT_OUTPUT_DIR"
      echo "  3. BASE_MODEL (Qwen/Qwen3-0.6B)"
      exit 0 ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift ;;
  esac
done

function should_run() {
  [[ -z "$STAGES" ]] && return 0
  [[ ",$STAGES," == *",$1,"* ]]
}

# ── Resolve the best input model ─────────────────────────────────
# Priority: explicit GRM_MODEL_NAME (if it's a local path) > SFT checkpoint > BASE_MODEL
RESOLVED_MODEL="${GRM_MODEL_NAME:-}"

function resolve_sft_model() {
  local sft_dir="${SFT_OUTPUT_DIR:-out/sft}"
  if [[ -f "$sft_dir/latest_checkpointed_iteration.txt" ]]; then
    local step
    step=$(cat "$sft_dir/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
    local cand="$sft_dir/global_step_${step}/huggingface"
    if [[ -d "$cand" && -f "$cand/config.json" ]]; then
      echo "$cand"
      return 0
    fi
  fi
  return 1
}

# If GRM_MODEL_NAME is a HuggingFace hub name (contains /), try upgrading to SFT checkpoint
if [[ "$RESOLVED_MODEL" == *"/"* && ! -d "$RESOLVED_MODEL" ]]; then
  SFT_MODEL=$(resolve_sft_model 2>/dev/null || true)
  if [[ -n "$SFT_MODEL" ]]; then
    echo "[INFO] Found SFT checkpoint, using as RL input: $SFT_MODEL"
    RESOLVED_MODEL="$SFT_MODEL"
    # Override for the Hydra config
    HYDRA_ARGS+=("actor_rollout_ref.model.path=$RESOLVED_MODEL")
  else
    echo "[INFO] No SFT checkpoint found, using base model: $RESOLVED_MODEL"
  fi
else
  echo "[INFO] Using model: $RESOLVED_MODEL"
fi

echo "RL algorithm: ${RL_ALGORITHM:-dapo}"

# ── RL output directory & checkpoint resolution ──────────────────
RL_CHECKPOINT_DIR="${RL_CHECKPOINT_DIR:-out/rl}"
RL_BENCH_DIR="${RL_BENCH_DIR:-out/bench/rl}"

function resolve_rl_model() {
  if [[ -f "$RL_CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
    local step
    step=$(cat "$RL_CHECKPOINT_DIR/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
    local cand="$RL_CHECKPOINT_DIR/global_step_${step}/huggingface"
    if [[ -d "$cand" && -f "$cand/config.json" ]]; then
      echo "$cand"
      return 0
    fi
  fi
  return 1
}

# ═══════════════════════════════════════════════════════════════════
# Stage 1: RL Training
# ═══════════════════════════════════════════════════════════════════
if should_run "train"; then
  echo "\n[STEP] RL Training (${RL_ALGORITHM:-dapo})"
  python -u -m src.training.verl_main \
    "${HYDRA_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}" \
    2>&1 | tee rl.log
fi

# ═══════════════════════════════════════════════════════════════════
# Stage 2: Benchmark the RL-trained model
# ═══════════════════════════════════════════════════════════════════
if should_run "bench"; then
  RL_MODEL=$(resolve_rl_model 2>/dev/null || true)
  if [[ -z "$RL_MODEL" ]]; then
    echo "[WARN] No RL checkpoint found in $RL_CHECKPOINT_DIR, skipping benchmark."
  else
    echo "\n[STEP] Post-RL benchmark"
    echo "[INFO] Using RL model: $RL_MODEL"

    PYTHON_BIN="${PYTHON_BIN:-python}"
    BENCHMARKS="${BENCHMARKS:-healthbench_rubric}"
    BENCH_NUM_SAMPLES="${BENCH_NUM_SAMPLES:-120}"
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-24}"
    EVAL_WORKERS="${EVAL_WORKERS:-6}"

    mkdir -p "$RL_BENCH_DIR"

    "$PYTHON_BIN" -m src.evaluation.run_benchmark \
      --model_path "$RL_MODEL" \
      --benchmarks $BENCHMARKS \
      --num_samples "$BENCH_NUM_SAMPLES" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --eval_workers "$EVAL_WORKERS" \
      --output_dir "$RL_BENCH_DIR"

    echo "[INFO] RL benchmark results saved to $RL_BENCH_DIR"
  fi
fi

echo "\n[DONE] RL pipeline completed."
