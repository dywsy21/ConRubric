#!/usr/bin/env bash
set -euo pipefail
# ═══════════════════════════════════════════════════════════════
# bench_comparison.sh — Run full benchmark suite on multiple models
#
# Compares: base model, SFT model, RL step 400, RL step 460
# Benchmarks: rewardbench, ppe, rmb, healthbench_rubric
#
# Usage:
#   PROXY_URL=http://127.0.0.1:7897 ./scripts/bench_comparison.sh
# ═══════════════════════════════════════════════════════════════

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

# Load .env
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a; source "$ROOT_DIR/.env"; set +a
fi

# Proxy: needed for HF dataset downloads, but exclude localhost (judge API)
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7897}"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export ALL_PROXY="$PROXY_URL"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export all_proxy="$PROXY_URL"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

NUM_SAMPLES="${NUM_SAMPLES:-192}"
BENCHMARKS="${BENCH_LIST:-rewardbench ppe rmb healthbench_rubric}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
BENCH_BASE_DIR="${BENCH_BASE_DIR:-out/bench}"
GPU="${BENCH_GPU:-0}"

LOG_DIR="$BENCH_BASE_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Define models to benchmark ────────────────────────────────
declare -A MODELS
MODELS[base]="models/Qwen3-4B"
MODELS[sft]="out/sft_lora_a16/huggingface"
MODELS[rl_step_400]="out/rl/global_step_400/actor/huggingface"
MODELS[rl_step_460]="out/rl/global_step_460/actor/huggingface"

# Order matters for display
MODEL_ORDER=(base sft rl_step_400 rl_step_460)

echo "═══════════════════════════════════════════════════════════════"
echo " Full Benchmark Comparison"
echo " Models:     ${MODEL_ORDER[*]}"
echo " Benchmarks: ${BENCHMARKS}"
echo " Samples:    ${NUM_SAMPLES}"
echo " GPU:        ${GPU}"
echo " Proxy:      ${PROXY_URL}"
echo "═══════════════════════════════════════════════════════════════"

# ── Pre-flight checks ─────────────────────────────────────────
for name in "${MODEL_ORDER[@]}"; do
  path="${MODELS[$name]}"
  if [[ ! -f "$path/config.json" ]]; then
    echo "[ERROR] Model not found: $path/config.json"
    exit 1
  fi
  echo "[OK] $name → $path"
done

# Check judge server
if ! curl -s --connect-timeout 3 http://localhost:8202/v1/models > /dev/null 2>&1; then
  echo "[ERROR] Judge server not reachable at localhost:8202"
  exit 1
fi
echo "[OK] Judge server at localhost:8202"
echo ""

# ── Run benchmarks sequentially ───────────────────────────────
for name in "${MODEL_ORDER[@]}"; do
  path="${MODELS[$name]}"
  out_dir="$BENCH_BASE_DIR/$name"
  log_file="$LOG_DIR/bench_${name}.log"

  # Check if already done (all result files present)
  EXPECTED_FILES=()
  for b in $BENCHMARKS; do
    case $b in
      rewardbench)       EXPECTED_FILES+=("reward_bench_results.json") ;;
      ppe)               EXPECTED_FILES+=("ppe_results.json") ;;
      rmb)               EXPECTED_FILES+=("rmb_results.json") ;;
      healthbench_rubric) EXPECTED_FILES+=("healthbench_rubric_quality.json") ;;
    esac
  done

  all_done=true
  for ef in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "$out_dir/$ef" ]]; then
      all_done=false
      break
    fi
  done

  if $all_done; then
    echo "[SKIP] $name — all results already exist at $out_dir"
    continue
  fi

  echo "[START] $name → $out_dir (log: $log_file)"
  mkdir -p "$out_dir"

  CUDA_VISIBLE_DEVICES=$GPU python -m src.evaluation.run_benchmark \
    --model_path "$path" \
    --benchmarks $BENCHMARKS \
    --num_samples "$NUM_SAMPLES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_workers "$EVAL_WORKERS" \
    --output_dir "$out_dir" \
    2>&1 | tee "$log_file"

  echo "[DONE] $name"
  echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " All benchmarks complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Collect and print comparison table ────────────────────────
python3 -c "
import json, os, sys

models = '${MODEL_ORDER[*]}'.split()
bench_dir = '$BENCH_BASE_DIR'

# Collect results
data = {}
for name in models:
    data[name] = {}
    d = os.path.join(bench_dir, name)
    
    # RewardBench
    f = os.path.join(d, 'reward_bench_results.json')
    if os.path.isfile(f):
        with open(f) as fh:
            r = json.load(fh)
        data[name]['rb_acc'] = r.get('accuracy', 0)
    
    # PPE
    f = os.path.join(d, 'ppe_results.json')
    if os.path.isfile(f):
        with open(f) as fh:
            r = json.load(fh)
        data[name]['ppe_acc'] = r.get('accuracy', 0)
    
    # RMB
    f = os.path.join(d, 'rmb_results.json')
    if os.path.isfile(f):
        with open(f) as fh:
            r = json.load(fh)
        data[name]['rmb_acc'] = r.get('accuracy', 0)
    
    # HealthBench Rubric Quality
    f = os.path.join(d, 'healthbench_rubric_quality.json')
    if os.path.isfile(f):
        with open(f) as fh:
            r = json.load(fh)
        s = r.get('summary', r)
        disc = s.get('discrimination', {})
        data[name]['hb_pair_acc'] = s.get('avg_pairwise_acc', 0)
        data[name]['hb_resolution'] = s.get('avg_resolution', 0)
        data[name]['hb_spearman'] = disc.get('avg_spearman', 0)
        data[name]['hb_top_bot'] = disc.get('avg_top_bottom_gap', 0)

# Print comparison
metrics = [
    ('rb_acc', 'RewardBench Acc'),
    ('ppe_acc', 'PPE Acc'),
    ('rmb_acc', 'RMB Acc'),
    ('hb_pair_acc', 'HB PairwiseAcc'),
    ('hb_resolution', 'HB Resolution'),
    ('hb_spearman', 'HB Spearman'),
    ('hb_top_bot', 'HB TopBotGap'),
]

# Header
header = f\"{'Metric':<20}\"
for name in models:
    header += f' | {name:>14}'
print(header)
print('-' * len(header))

for key, label in metrics:
    row = f'{label:<20}'
    vals = [data[name].get(key) for name in models]
    best = max((v for v in vals if v is not None), default=None)
    for name in models:
        v = data[name].get(key)
        if v is not None:
            marker = ' *' if v == best and best is not None else '  '
            row += f' | {v:>12.4f}{marker}'
        else:
            row += f' | {\"N/A\":>14}'
    print(row)

print()
print('* = best in row')
"
