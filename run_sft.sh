#!/usr/bin/env zsh
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# run_sft.sh — Full SFT pipeline with stage selection
#
# Usage:
#   ./run_sft.sh                        # Run all stages
#   ./run_sft.sh --stage bench-pre      # Only pre-SFT benchmark
#   ./run_sft.sh --stage prep           # Only prepare data
#   ./run_sft.sh --stage train          # Only train
#   ./run_sft.sh --stage bench-post     # Only post-SFT benchmark
#   ./run_sft.sh --stage compare        # Only rubric comparison
#   ./run_sft.sh --stage writeup        # Only generate writeup
#   ./run_sft.sh --stage train,bench-post  # Comma-separated stages
# ═══════════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
else
  echo "[ERROR] .env not found at $ROOT_DIR/.env"
  exit 1
fi

# ── Parse arguments ───────────────────────────────────────────────
STAGES=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage|-s)
      STAGES="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--stage STAGE[,STAGE,...]]"
      echo ""
      echo "Stages:"
      echo "  bench-pre   Pre-SFT benchmark (auto-skipped if results exist)"
      echo "  prep        Prepare mixed SFT data"
      echo "  train       Run SFT training"
      echo "  bench-post  Post-SFT benchmark"
      echo "  compare     Generate pre/post rubric comparison"
      echo "  writeup     Write sft_writeup.md"
      echo ""
      echo "Default: run all stages in order."
      exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

function should_run() {
  # If no --stage given, run everything
  [[ -z "$STAGES" ]] && return 0
  # Check if this stage is in the comma-separated list
  [[ ",$STAGES," == *",$1,"* ]]
}

# ── Validate required env vars ────────────────────────────────────
function require_env() {
  local name="$1"
  if [[ -z "${(P)name:-}" ]]; then
    echo "[ERROR] Required env var not set: $name"
    exit 1
  fi
}

require_env PYTHON_BIN
require_env BASE_MODEL
require_env LIMIT_PER_DATASET
require_env BENCH_NUM_SAMPLES
require_env BENCHMARKS
require_env EVAL_BATCH_SIZE
require_env EVAL_WORKERS
require_env SFT_EPOCHS
require_env SFT_TRAIN_BATCH
require_env SFT_MICRO_BATCH
require_env SFT_NPROC
require_env SFT_OUTPUT_DIR
require_env SFT_EXPERIMENT_NAME
require_env SFT_JSONL
require_env PRE_DIR
require_env POST_DIR
require_env COMPARISON_MD
require_env COMPARISON_JSONL
require_env WRITEUP_OUT

POST_MODEL_PATH="${POST_MODEL_PATH:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found: $PYTHON_BIN"
  exit 1
fi

mkdir -p "$PRE_DIR" "$POST_DIR"

function run_step() {
  echo "\n[STEP] $1"
}

# ── Helper: resolve POST_MODEL_PATH from verl checkpoint dir ─────
function resolve_post_model() {
  # Already set and valid (has config + weights)?
  if [[ -n "$POST_MODEL_PATH" && -d "$POST_MODEL_PATH" && -f "$POST_MODEL_PATH/config.json" ]]; then
    if compgen -G "$POST_MODEL_PATH/model*.safetensors" >/dev/null 2>&1 || \
       compgen -G "$POST_MODEL_PATH/pytorch_model*.bin" >/dev/null 2>&1; then
      return 0
    fi
  fi

  # verl convention: $SFT_OUTPUT_DIR/global_step_N/huggingface
  local step_dir=""
  if [[ -f "$SFT_OUTPUT_DIR/latest_checkpointed_iteration.txt" ]]; then
    local step
    step=$(cat "$SFT_OUTPUT_DIR/latest_checkpointed_iteration.txt" | tr -d '[:space:]')
    step_dir="$SFT_OUTPUT_DIR/global_step_${step}"
    local cand="$step_dir/huggingface"
    if [[ -d "$cand" && -f "$cand/config.json" ]]; then
      POST_MODEL_PATH="$cand"
    fi
  fi

  # If POST_MODEL_PATH resolved but weights are missing, auto-convert
  if [[ -n "$POST_MODEL_PATH" && -d "$POST_MODEL_PATH" && -f "$POST_MODEL_PATH/config.json" ]]; then
    if ! compgen -G "$POST_MODEL_PATH/model*.safetensors" >/dev/null 2>&1 && \
       ! compgen -G "$POST_MODEL_PATH/pytorch_model*.bin" >/dev/null 2>&1; then
      echo "[INFO] verl checkpoint has no HF weights — auto-converting …"
      "$PYTHON_BIN" -m src.training.convert_verl_ckpt "$step_dir" || {
        echo "[ERROR] Checkpoint conversion failed."
        return 1
      }
    fi
    return 0
  fi

  # Fallback candidates
  for cand in \
    "$SFT_OUTPUT_DIR" \
    "$SFT_OUTPUT_DIR/latest" \
    "$SFT_OUTPUT_DIR/hf_model"; do
    if [[ -d "$cand" && -f "$cand/config.json" ]]; then
      POST_MODEL_PATH="$cand"
      return 0
    fi
  done

  return 1
}

# ── Helper: check if benchmark results already exist ─────────────
function bench_results_exist() {
  local dir="$1"
  local count=0
  for bm in ${(z)BENCHMARKS}; do
    case "$bm" in
      rewardbench)        [[ -f "$dir/reward_bench_results.json" ]] && (( count++ )) ;;
      ppe)                [[ -f "$dir/ppe_results.json" ]] && (( count++ )) ;;
      rmb)                [[ -f "$dir/rmb_results.json" ]] && (( count++ )) ;;
      healthbench_rubric) [[ -f "$dir/healthbench_rubric_quality.json" ]] && (( count++ )) ;;
      *)                  [[ -f "$dir/${bm}_results.json" ]] && (( count++ )) ;;
    esac
  done
  local expected=${#${(z)BENCHMARKS}}
  [[ "$count" -ge "$expected" ]]
}

# ═══════════════════════════════════════════════════════════════════
# Stage 1: Pre-SFT benchmark
# ═══════════════════════════════════════════════════════════════════
if should_run "bench-pre"; then
  if bench_results_exist "$PRE_DIR"; then
    echo "\n[SKIP] Pre-SFT benchmark — results already exist in $PRE_DIR"
  else
    run_step "1/6 Pre-SFT benchmark"
    "$PYTHON_BIN" -m src.evaluation.run_benchmark \
      --model_path "$BASE_MODEL" \
      --benchmarks ${(z)BENCHMARKS} \
      --num_samples "$BENCH_NUM_SAMPLES" \
      --eval_batch_size "$EVAL_BATCH_SIZE" \
      --eval_workers "$EVAL_WORKERS" \
      --output_dir "$PRE_DIR"
  fi
fi

# ═══════════════════════════════════════════════════════════════════
# Stage 2: Prepare mixed SFT data
# ═══════════════════════════════════════════════════════════════════
if should_run "prep"; then
  run_step "2/6 Prepare mixed SFT data (HealthBench+synthetic, each limit=$LIMIT_PER_DATASET)"
  PREPARE_ARGS=(
    -m src.training.sft_trainer
    --prepare-only
    --healthbench-limit "$LIMIT_PER_DATASET"
    --synthetic-limit "$LIMIT_PER_DATASET"
    --output-jsonl "$SFT_JSONL"
    --preview-samples 2
  )
  if [[ "${REBUILD_SPLIT:-0}" == "1" ]]; then
    PREPARE_ARGS+=(--rebuild-healthbench-split)
  fi
  "$PYTHON_BIN" ${PREPARE_ARGS[@]}
fi

# ═══════════════════════════════════════════════════════════════════
# Stage 3: Run SFT training
# ═══════════════════════════════════════════════════════════════════
if should_run "train"; then
  if [[ "${RUN_SFT:-1}" == "1" ]]; then
    run_step "3/6 Run weighted verl SFT"
    TRAIN_ARGS=(
      -m src.training.sft_trainer
      --healthbench-limit "$LIMIT_PER_DATASET"
      --synthetic-limit "$LIMIT_PER_DATASET"
      --epochs "$SFT_EPOCHS"
      --train-batch-size "$SFT_TRAIN_BATCH"
      --micro-batch-size-per-gpu "$SFT_MICRO_BATCH"
      --nproc-per-node "$SFT_NPROC"
      --project-name grm-sft
      --experiment-name "$SFT_EXPERIMENT_NAME"
      --output-dir "$SFT_OUTPUT_DIR"
      --save-freq -1
      --test-freq -1
    )
    "$PYTHON_BIN" ${TRAIN_ARGS[@]}
  else
    echo "\n[SKIP] SFT training (RUN_SFT=0)"
  fi
fi

# ═══════════════════════════════════════════════════════════════════
# Stage 4: Post-SFT benchmark
# ═══════════════════════════════════════════════════════════════════
if should_run "bench-post"; then
  resolve_post_model || {
    echo "[ERROR] POST_MODEL_PATH not found. Train first or set POST_MODEL_PATH."
    exit 2
  }
  echo "[INFO] Using post-SFT model: $POST_MODEL_PATH"

  run_step "4/6 Post-SFT benchmark"
  "$PYTHON_BIN" -m src.evaluation.run_benchmark \
    --model_path "$POST_MODEL_PATH" \
    --benchmarks ${(z)BENCHMARKS} \
    --num_samples "$BENCH_NUM_SAMPLES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_workers "$EVAL_WORKERS" \
    --output_dir "$POST_DIR"
fi

# ═══════════════════════════════════════════════════════════════════
# Stage 5: Rubric comparison
# ═══════════════════════════════════════════════════════════════════
if should_run "compare"; then
  resolve_post_model || {
    echo "[ERROR] POST_MODEL_PATH not found. Train first or set POST_MODEL_PATH."
    exit 2
  }

  run_step "5/6 Generate pre/post rubric comparison (>$BENCH_NUM_SAMPLES same prompts)"
  "$PYTHON_BIN" -m src.evaluation.build_rubric_comparison \
    --pre_dir "$PRE_DIR" \
    --post_dir "$POST_DIR" \
    --pre_model "$BASE_MODEL" \
    --post_model "$POST_MODEL_PATH" \
    --num_rubric_samples 10 \
    --output_md "$COMPARISON_MD" \
    --output_jsonl "$COMPARISON_JSONL"
fi

# ═══════════════════════════════════════════════════════════════════
# Stage 6: Write up
# ═══════════════════════════════════════════════════════════════════
if should_run "writeup"; then
  resolve_post_model || {
    echo "[ERROR] POST_MODEL_PATH not found. Train first or set POST_MODEL_PATH."
    exit 2
  }

  run_step "6/6 Write sft_writeup.md"
  "$PYTHON_BIN" -m src.evaluation.write_sft_writeup \
    --pre_dir "$PRE_DIR" \
    --post_dir "$POST_DIR" \
    --base_model "$BASE_MODEL" \
    --post_model "$POST_MODEL_PATH" \
    --sft_jsonl "$SFT_JSONL" \
    --comparison_md "$COMPARISON_MD" \
    --comparison_jsonl "$COMPARISON_JSONL" \
    --output "$WRITEUP_OUT"
fi

echo "\n[DONE] SFT pipeline completed."
echo "- writeup:          $WRITEUP_OUT"
echo "- pre benchmark:    $PRE_DIR"
echo "- post benchmark:   $POST_DIR"
echo "- rubric comparison: $COMPARISON_MD"
