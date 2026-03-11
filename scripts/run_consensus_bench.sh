#!/usr/bin/env bash
set -euo pipefail
# Run consensus benchmark on all 4 models and compare

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"
set -a; source "$ROOT_DIR/.env"; set +a

GPU="${BENCH_GPU:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-30}"
NUM_RUBRICS="${NUM_RUBRICS:-5}"
MAX_COMPLETIONS="${MAX_COMPLETIONS:-4}"
OUT_DIR="${OUT_DIR:-out/bench/consensus}"

declare -A MODELS
MODELS[base]="models/Qwen3-4B"
MODELS[sft]="out/sft_lora_a16/huggingface"
MODELS[rl_step_400]="out/rl/global_step_400/actor/huggingface"
MODELS[rl_step_460]="out/rl/global_step_460/actor/huggingface"
MODEL_ORDER=(base sft rl_step_400 rl_step_460)

echo "═══════════════════════════════════════════════════════════════"
echo " Cross-Rubric Consensus Benchmark"
echo " Models:       ${MODEL_ORDER[*]}"
echo " Prompts:      ${MAX_PROMPTS}"
echo " Rubrics/prompt: ${NUM_RUBRICS}"
echo " Completions:  ${MAX_COMPLETIONS}"
echo "═══════════════════════════════════════════════════════════════"

for name in "${MODEL_ORDER[@]}"; do
  path="${MODELS[$name]}"
  out="$OUT_DIR/$name"
  
  if [[ -f "$out/consensus_results.json" ]]; then
    echo "[SKIP] $name — results exist"
    continue
  fi
  
  echo "[START] $name"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/bench_consensus.py \
    --model_path "$path" \
    --output_dir "$out" \
    --max_prompts "$MAX_PROMPTS" \
    --num_rubrics "$NUM_RUBRICS" \
    --max_completions "$MAX_COMPLETIONS" \
    --eval_workers 8

  echo "[DONE] $name"
  echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Consensus Results Comparison"
echo "═══════════════════════════════════════════════════════════════"

python3 -c "
import json, os

models = '${MODEL_ORDER[*]}'.split()
base = '$OUT_DIR'

metrics = [
    ('avg_score_variance', 'Score Variance', True),   # lower is better
    ('avg_rank_agreement', 'Rank Agreement', False),   # higher is better
    ('ensemble_label_corr', 'Ensemble LabelCorr', False),
    ('avg_individual_label_corr', 'Indiv LabelCorr', False),
    ('collapse_rate', 'Collapse Rate', True),          # lower is better
    ('avg_rubric_length', 'Avg Rubric Len', False),
]

data = {}
for name in models:
    f = os.path.join(base, name, 'consensus_results.json')
    if os.path.isfile(f):
        with open(f) as fh:
            data[name] = json.load(fh)['summary']

header = f\"{'Metric':<22}\"
for name in models:
    header += f' | {name:>14}'
print(header)
print('-' * len(header))

for key, label, lower_better in metrics:
    row = f'{label:<22}'
    vals = [data.get(name, {}).get(key) for name in models]
    valid = [v for v in vals if v is not None]
    best = min(valid) if lower_better and valid else (max(valid) if valid else None)
    for v in vals:
        if v is not None:
            marker = ' *' if v == best else '  '
            row += f' | {v:>12.4f}{marker}'
        else:
            row += f' | {\"N/A\":>14}'
    print(row)

print()
dir_marker = '(lower is better)' 
print(f'* = best in row. Score Variance & Collapse Rate: lower is better. Others: higher is better.')
"
