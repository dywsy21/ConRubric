#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
else
  echo "[ERROR] .env not found at $ROOT_DIR/.env"
  exit 1
fi

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
require_env RUN_SFT
require_env REBUILD_SPLIT
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

mkdir -p "$ROOT_DIR/results/sft_eval" "$PRE_DIR" "$POST_DIR"

function run_step() {
  echo "\n[STEP] $1"
}

run_step "1/6 Pre-SFT benchmark"
"$PYTHON_BIN" -m src.evaluation.run_benchmark \
  --model_path "$BASE_MODEL" \
  --benchmarks ${(z)BENCHMARKS} \
  --num_samples "$BENCH_NUM_SAMPLES" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --eval_workers "$EVAL_WORKERS" \
  --output_dir "$PRE_DIR"

run_step "2/6 Prepare mixed SFT data (HealthBench+synthetic, each limit=100)"
PREPARE_ARGS=(
  -m src.training.sft_trainer
  --prepare-only
  --healthbench-limit "$LIMIT_PER_DATASET"
  --synthetic-limit "$LIMIT_PER_DATASET"
  --output-jsonl "$SFT_JSONL"
  --preview-samples 2
)
if [[ "$REBUILD_SPLIT" == "1" ]]; then
  PREPARE_ARGS+=(--rebuild-healthbench-split)
fi
"$PYTHON_BIN" ${PREPARE_ARGS[@]}

if [[ "$RUN_SFT" == "1" ]]; then
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
fi

if [[ -z "$POST_MODEL_PATH" ]]; then
  # User can provide explicit POST_MODEL_PATH. If not provided, try common candidate paths.
  for cand in \
    "$SFT_OUTPUT_DIR" \
    "$SFT_OUTPUT_DIR/latest" \
    "$SFT_OUTPUT_DIR/hf_model"; do
    if [[ -d "$cand" && -f "$cand/config.json" ]]; then
      POST_MODEL_PATH="$cand"
      break
    fi
  done
fi

if [[ -z "$POST_MODEL_PATH" ]]; then
  echo "[ERROR] POST_MODEL_PATH not found. Please set POST_MODEL_PATH to a valid post-SFT HF model directory."
  exit 2
fi

echo "[INFO] Using post-SFT model: $POST_MODEL_PATH"

run_step "4/6 Post-SFT benchmark"
"$PYTHON_BIN" -m src.evaluation.run_benchmark \
  --model_path "$POST_MODEL_PATH" \
  --benchmarks ${(z)BENCHMARKS} \
  --num_samples "$BENCH_NUM_SAMPLES" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --eval_workers "$EVAL_WORKERS" \
  --output_dir "$POST_DIR"

run_step "5/6 Generate pre/post rubric comparison (>100 same prompts)"
"$PYTHON_BIN" -m src.evaluation.build_rubric_comparison \
  --pre_model "$BASE_MODEL" \
  --post_model "$POST_MODEL_PATH" \
  --num_samples "$BENCH_NUM_SAMPLES" \
  --output_md "$COMPARISON_MD" \
  --output_jsonl "$COMPARISON_JSONL"

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

echo "\n[DONE] Full SFT pipeline completed."
echo "- writeup: $WRITEUP_OUT"
echo "- pre benchmark: $PRE_DIR"
echo "- post benchmark: $POST_DIR"
echo "- rubric comparison: $COMPARISON_MD"
