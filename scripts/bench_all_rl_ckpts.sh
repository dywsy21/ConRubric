#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# bench_all_rl_ckpts.sh — Benchmark ALL RL checkpoints in parallel
#
# Runs 2 benchmark processes simultaneously (1 per GPU) to evaluate
# every RL checkpoint.  Results go to out/bench/rl_step_<N>/.
#
# Usage:
#   ./scripts/bench_all_rl_ckpts.sh                 # all ckpts
#   ./scripts/bench_all_rl_ckpts.sh --steps 100,500,960   # selected
# ═══════════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

# Load .env
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

# Disable proxies
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy 2>/dev/null || true

NUM_SAMPLES="${BENCH_NUM_SAMPLES:-120}"
BENCHMARKS="${BENCHMARKS:-healthbench_rubric}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-24}"
EVAL_WORKERS="${EVAL_WORKERS:-6}"
BENCH_BASE_DIR="${BENCH_BASE_DIR:-out/bench}"
RL_DIR="${RL_CHECKPOINT_DIR:-out/rl}"
NUM_GPUS=2
# GPU IDs to use for benchmark (avoid GPUs running the judge vLLM)
# Default: both on GPU 0 since GPU 1 has the judge server
BENCH_GPUS="${BENCH_GPUS:-0,0}"

# ── Parse arguments ───────────────────────────────────────────────
SELECTED_STEPS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps|-s) SELECTED_STEPS="$2"; shift 2 ;;
    --num-samples|-n) NUM_SAMPLES="$2"; shift 2 ;;
    --benchmarks|-b) BENCHMARKS="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--steps STEP1,STEP2,...] [--num-samples N] [--benchmarks BENCH]"
      exit 0 ;;
    *) shift ;;
  esac
done

# ── Discover checkpoints ─────────────────────────────────────────
CKPTS=()
if [[ -n "$SELECTED_STEPS" ]]; then
  IFS=',' read -ra STEPS <<< "$SELECTED_STEPS"
  for step in "${STEPS[@]}"; do
    ckpt="$RL_DIR/global_step_${step}/actor/huggingface"
    if [[ -d "$ckpt" && -f "$ckpt/config.json" ]]; then
      CKPTS+=("$step:$ckpt")
    else
      echo "[WARN] Checkpoint not found: $ckpt"
    fi
  done
else
  # Discover all checkpoints
  for d in "$RL_DIR"/global_step_*/actor/huggingface; do
    if [[ -f "$d/config.json" ]]; then
      step=$(echo "$d" | grep -oP 'global_step_\K\d+')
      CKPTS+=("$step:$d")
    fi
  done
fi

# Sort by step number
IFS=$'\n' CKPTS=($(printf '%s\n' "${CKPTS[@]}" | sort -t: -k1 -n)); unset IFS

echo "═══════════════════════════════════════════════════════════════"
echo " Benchmarking ${#CKPTS[@]} RL checkpoints (${NUM_GPUS} GPUs parallel)"
echo " Benchmark: ${BENCHMARKS}"
echo " Samples:   ${NUM_SAMPLES}"
echo "═══════════════════════════════════════════════════════════════"

# ── Skip already-completed benchmarks ─────────────────────────────
TODO=()
for entry in "${CKPTS[@]}"; do
  step="${entry%%:*}"
  out_dir="$BENCH_BASE_DIR/rl_step_${step}"
  # Check if results already exist
  if [[ -f "$out_dir/healthbench_rubric_quality.json" ]]; then
    echo "[SKIP] Step $step — results already exist at $out_dir"
  else
    TODO+=("$entry")
  fi
done

if [[ ${#TODO[@]} -eq 0 ]]; then
  echo "[INFO] All benchmarks already completed."
  exit 0
fi
echo "[INFO] ${#TODO[@]} checkpoints to benchmark (${#CKPTS[@]} - $((${#CKPTS[@]} - ${#TODO[@]})) skipped)"

# ── Run benchmarks: NUM_GPUS at a time ────────────────────────────
IFS=',' read -ra GPU_LIST <<< "$BENCH_GPUS"
NUM_SLOTS=${#GPU_LIST[@]}
PIDS=()      # background PIDs
SLOT_BUSY=() # which slot each PID is using
LOG_DIR="$BENCH_BASE_DIR/logs"
mkdir -p "$LOG_DIR"

function wait_for_slot() {
  # Wait until at least one slot is free
  while [[ ${#PIDS[@]} -ge $NUM_SLOTS ]]; do
    NEW_PIDS=()
    NEW_SLOT=()
    for i in "${!PIDS[@]}"; do
      if kill -0 "${PIDS[$i]}" 2>/dev/null; then
        NEW_PIDS+=("${PIDS[$i]}")
        NEW_SLOT+=("${SLOT_BUSY[$i]}")
      else
        wait "${PIDS[$i]}" || echo "[WARN] Process ${PIDS[$i]} (slot ${SLOT_BUSY[$i]}, GPU ${GPU_LIST[${SLOT_BUSY[$i]}]}) exited with error"
        echo "[FREE] Slot ${SLOT_BUSY[$i]} (GPU ${GPU_LIST[${SLOT_BUSY[$i]}]}) is now available"
      fi
    done
    PIDS=("${NEW_PIDS[@]}")
    SLOT_BUSY=("${NEW_SLOT[@]}")
    if [[ ${#PIDS[@]} -ge $NUM_SLOTS ]]; then
      sleep 5
    fi
  done
}

function get_free_slot() {
  for slot in $(seq 0 $((NUM_SLOTS - 1))); do
    local in_use=false
    for busy_slot in "${SLOT_BUSY[@]}"; do
      if [[ "$busy_slot" == "$slot" ]]; then
        in_use=true
        break
      fi
    done
    if ! $in_use; then
      echo "$slot"
      return
    fi
  done
  echo "0"  # fallback
}

for entry in "${TODO[@]}"; do
  step="${entry%%:*}"
  ckpt="${entry#*:}"
  out_dir="$BENCH_BASE_DIR/rl_step_${step}"
  log_file="$LOG_DIR/bench_step_${step}.log"

  wait_for_slot
  slot=$(get_free_slot)
  gpu="${GPU_LIST[$slot]}"

  echo "[START] Step $step on GPU $gpu (slot $slot) → $out_dir (log: $log_file)"
  mkdir -p "$out_dir"

  CUDA_VISIBLE_DEVICES=$gpu python -m src.evaluation.run_benchmark \
    --model_path "$ckpt" \
    --benchmarks $BENCHMARKS \
    --num_samples "$NUM_SAMPLES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_workers "$EVAL_WORKERS" \
    --output_dir "$out_dir" \
    > "$log_file" 2>&1 &

  PIDS+=($!)
  SLOT_BUSY+=("$slot")
  echo "[LAUNCHED] PID $! for step $step on GPU $gpu (slot $slot)"

  # Small delay to avoid race conditions on model loading
  sleep 3
done

# ── Wait for all remaining processes ──────────────────────────────
echo ""
echo "[INFO] Waiting for remaining ${#PIDS[@]} processes to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "[ERROR] Process ${PIDS[$i]} (slot ${SLOT_BUSY[$i]}, GPU ${GPU_LIST[${SLOT_BUSY[$i]}]}) failed"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Benchmarking complete! ($FAILED failures out of ${#TODO[@]})"
echo "═══════════════════════════════════════════════════════════════"

# ── Collect and print summary ─────────────────────────────────────
echo ""
echo "Step | Metric | Value"
echo "-----|--------|------"
for entry in "${CKPTS[@]}"; do
  step="${entry%%:*}"
  results_file="$BENCH_BASE_DIR/rl_step_${step}/healthbench_rubric_quality.json"
  if [[ -f "$results_file" ]]; then
    # Extract key metrics
    python3 -c "
import json, sys
with open('$results_file') as f:
    d = json.load(f)
step = $step
s = d.get('summary', d)
pa = s.get('avg_pairwise_acc', 0)
res = s.get('avg_resolution', 0)
disc = s.get('discrimination', {})
sp = disc.get('avg_spearman', 0)
tbg = disc.get('avg_top_bottom_gap', 0)
print(f'{step:>5} | pairwise_acc={pa:.4f}  resolution={res:.4f}  spearman={sp:.4f}  top_bottom_gap={tbg:.4f}')
" 2>/dev/null || echo " $step | (parse error)"
  else
    echo " $step | (no results)"
  fi
done
