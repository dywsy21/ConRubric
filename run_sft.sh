#!/usr/bin/env bash
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

# ── Apply proxy / CUDA from .env ─────────────────────────────────
if [[ -n "${PROXY_URL:-}" ]]; then
  export ALL_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" HTTP_PROXY="$PROXY_URL"
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
  export no_proxy="${NO_PROXY:-localhost,127.0.0.1}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# ── Defaults (override in .env) ──────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-${GRM_MODEL_NAME:-}}"
LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-100}"
BENCH_NUM_SAMPLES="${BENCH_NUM_SAMPLES:-120}"
BENCHMARKS="${BENCHMARKS:-rewardbench ppe rmb healthbench_rubric}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-24}"
EVAL_WORKERS="${EVAL_WORKERS:-6}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_TRAIN_BATCH="${SFT_TRAIN_BATCH:-32}"
SFT_MICRO_BATCH="${SFT_MICRO_BATCH:-4}"
SFT_NPROC="${SFT_NPROC:-1}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-out/sft}"
SFT_EXPERIMENT_NAME="${SFT_EXPERIMENT_NAME:-grm_sft}"
SFT_JSONL="${SFT_JSONL:-data/sft_train.jsonl}"
SFT_LR="${SFT_LR:-1e-4}"
SFT_LR_SCHEDULER="${SFT_LR_SCHEDULER:-cosine}"
SFT_LR_WARMUP_RATIO="${SFT_LR_WARMUP_RATIO:-0.05}"
SFT_MIN_LR_RATIO="${SFT_MIN_LR_RATIO:-0.1}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-2048}"
SFT_SAVE_FREQ="${SFT_SAVE_FREQ:-500}"
SFT_TEST_FREQ="${SFT_TEST_FREQ:--1}"
SFT_NO_HEALTHBENCH="${SFT_NO_HEALTHBENCH:-0}"
SFT_NO_SYNTHETIC="${SFT_NO_SYNTHETIC:-0}"
SFT_SYNTHETIC_PATH="${SFT_SYNTHETIC_PATH:-}"
SFT_LORA_RANK="${SFT_LORA_RANK:-0}"
SFT_LORA_ALPHA="${SFT_LORA_ALPHA:-64}"
SFT_LORA_TARGET_MODULES="${SFT_LORA_TARGET_MODULES:-all-linear}"
PRE_DIR="${PRE_DIR:-out/bench/base}"
POST_DIR="${POST_DIR:-out/bench/sft}"
COMPARISON_MD="${COMPARISON_MD:-out/bench/sft/rubric_comparison.md}"
COMPARISON_JSONL="${COMPARISON_JSONL:-out/bench/sft/rubric_comparison.jsonl}"
WRITEUP_OUT="${WRITEUP_OUT:-out/sft_writeup.md}"
POST_MODEL_PATH="${POST_MODEL_PATH:-}"

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
      echo "All config is read from .env (see .env.example)."
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

# ── Validate essentials ──────────────────────────────────────────
if [[ -z "$BASE_MODEL" ]]; then
  echo "[ERROR] BASE_MODEL (or GRM_MODEL_NAME) not set in .env"
  exit 1
fi
if [[ ! -f "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found: $PYTHON_BIN — run ./run_env_preparing.sh first"
  exit 1
fi

mkdir -p "$PRE_DIR" "$POST_DIR"

function run_step() {
  printf '\n[STEP] %s\n' "$1"
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
  for bm in $BENCHMARKS; do
    case "$bm" in
      rewardbench)        [[ -f "$dir/reward_bench_results.json" ]] && (( count++ )) ;;
      ppe)                [[ -f "$dir/ppe_results.json" ]] && (( count++ )) ;;
      rmb)                [[ -f "$dir/rmb_results.json" ]] && (( count++ )) ;;
      healthbench_rubric) [[ -f "$dir/healthbench_rubric_quality.json" ]] && (( count++ )) ;;
      *)                  [[ -f "$dir/${bm}_results.json" ]] && (( count++ )) ;;
    esac
  done
  local -a bm_arr=($BENCHMARKS)
  [[ "$count" -ge "${#bm_arr[@]}" ]]
}

# ═══════════════════════════════════════════════════════════════════
# Stage 1: Pre-SFT benchmark
# ═══════════════════════════════════════════════════════════════════
if should_run "bench-pre"; then
  if bench_results_exist "$PRE_DIR"; then
    printf '\n[SKIP] Pre-SFT benchmark — results already exist in %s\n' "$PRE_DIR"
  else
    run_step "1/6 Pre-SFT benchmark"
    "$PYTHON_BIN" -m src.evaluation.run_benchmark \
      --model_path "$BASE_MODEL" \
      --benchmarks $BENCHMARKS \
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
  run_step "2/6 Prepare mixed SFT data"
  PREPARE_ARGS=(
    -m src.training.sft_trainer
    --prepare-only
    --output-jsonl "$SFT_JSONL"
    --preview-samples 2
  )
  if [[ "$SFT_NO_HEALTHBENCH" == "1" ]]; then
    PREPARE_ARGS+=(--no-healthbench)
  else
    PREPARE_ARGS+=(--healthbench-limit "$LIMIT_PER_DATASET")
  fi
  if [[ "$SFT_NO_SYNTHETIC" == "1" ]]; then
    PREPARE_ARGS+=(--no-synthetic)
  elif [[ -n "$SFT_SYNTHETIC_PATH" ]]; then
    PREPARE_ARGS+=(--synthetic-path "$SFT_SYNTHETIC_PATH")
  else
    PREPARE_ARGS+=(--synthetic-limit "$LIMIT_PER_DATASET")
  fi
  if [[ "${REBUILD_SPLIT:-0}" == "1" ]]; then
    PREPARE_ARGS+=(--rebuild-healthbench-split)
  fi
  "$PYTHON_BIN" "${PREPARE_ARGS[@]}"
fi

# ═══════════════════════════════════════════════════════════════════
# Stage 3: Run SFT training
# ═══════════════════════════════════════════════════════════════════
if should_run "train"; then
  if [[ "${RUN_SFT:-1}" == "1" ]]; then
    run_step "3/6 Run weighted verl SFT"
    TRAIN_ARGS=(
      -m src.training.sft_trainer
      --epochs "$SFT_EPOCHS"
      --train-batch-size "$SFT_TRAIN_BATCH"
      --micro-batch-size-per-gpu "$SFT_MICRO_BATCH"
      --nproc-per-node "$SFT_NPROC"
      --max-length "$SFT_MAX_LENGTH"
      --lr "$SFT_LR"
      --lr-scheduler-type "$SFT_LR_SCHEDULER"
      --lr-warmup-ratio "$SFT_LR_WARMUP_RATIO"
      --min-lr-ratio "$SFT_MIN_LR_RATIO"
      --project-name grm-sft
      --experiment-name "$SFT_EXPERIMENT_NAME"
      --output-dir "$SFT_OUTPUT_DIR"
      --save-freq "$SFT_SAVE_FREQ"
      --test-freq "$SFT_TEST_FREQ"
      --preview-samples 2
    )
    # Data source selection
    if [[ "$SFT_NO_HEALTHBENCH" == "1" ]]; then
      TRAIN_ARGS+=(--no-healthbench)
    else
      TRAIN_ARGS+=(--healthbench-limit "$LIMIT_PER_DATASET")
    fi
    if [[ "$SFT_NO_SYNTHETIC" == "1" ]]; then
      TRAIN_ARGS+=(--no-synthetic)
    elif [[ -n "$SFT_SYNTHETIC_PATH" ]]; then
      TRAIN_ARGS+=(--synthetic-path "$SFT_SYNTHETIC_PATH")
    else
      TRAIN_ARGS+=(--synthetic-limit "$LIMIT_PER_DATASET")
    fi
    # LoRA / PEFT
    if [[ "$SFT_LORA_RANK" -gt 0 ]]; then
      TRAIN_ARGS+=(--lora-rank "$SFT_LORA_RANK" --lora-alpha "$SFT_LORA_ALPHA" --lora-target-modules "$SFT_LORA_TARGET_MODULES")
    fi
    "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
  else
    printf '\n[SKIP] SFT training (RUN_SFT=0)\n'
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
    --benchmarks $BENCHMARKS \
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

printf '\n[DONE] SFT pipeline completed.\n'
echo "- writeup:          $WRITEUP_OUT"
echo "- pre benchmark:    $PRE_DIR"
echo "- post benchmark:   $POST_DIR"
echo "- rubric comparison: $COMPARISON_MD"
