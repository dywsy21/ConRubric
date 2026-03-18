#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Unified data preprocessing: generates SFT + RL training data.
#
# Steps:
#   1. Generate synthetic rubrics via Oracle  → data/synthetic_rubrics.jsonl
#   2. Prepare SFT weighted mix (HealthBench + synthetic) → data/sft_train.jsonl
#   3. Prepare RL parquet (HealthBench + synthetic)        → data/rl_train.parquet
#
# Usage:
#   ./run_data_preprocessing.sh --limit 100      # 100 samples per dataset
#   ./run_data_preprocessing.sh                   # use LIMIT_PER_DATASET from .env
#   ./run_data_preprocessing.sh --skip-synthetic  # skip Oracle generation (reuse existing)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── load .env ─────────────────────────────────────────────────────────────
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
else
  echo "[ERROR] .env not found at $ROOT_DIR/.env"
  exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"

# ── Apply proxy / CUDA from .env ──
if [[ -n "${PROXY_URL:-}" ]]; then
  export ALL_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" HTTP_PROXY="$PROXY_URL"
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
  export no_proxy="${NO_PROXY:-localhost,127.0.0.1}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

PYTHON="${PYTHON_BIN:-python}"
LIMIT="${LIMIT_PER_DATASET:-100}"
SKIP_SYNTHETIC=0
REBUILD_HB="${REBUILD_SPLIT:-0}"
PREVIEW=2
RUBRICHUB="${RUBRICHUB:-0}"
RUBRICHUB_OUT="${RUBRICHUB_SFT_JSONL:-${ROOT_DIR}/data/rubrichub_sft.jsonl}"

# ── parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"; shift 2 ;;
    --skip-synthetic)
      SKIP_SYNTHETIC=1; shift ;;
    --rebuild-split)
      REBUILD_HB=1; shift ;;
    --no-preview)
      PREVIEW=0; shift ;;
    --rubrichub)
      RUBRICHUB=1; shift ;;
    --skip-rubrichub)
      RUBRICHUB=0; shift ;;
    -h|--help)
      echo "Usage: $0 [--limit N] [--skip-synthetic] [--rubrichub] [--rebuild-split] [--no-preview]"
      echo ""
      echo "  --limit N          Samples per dataset (default: LIMIT_PER_DATASET from .env, or 100)"
      echo "  --skip-synthetic   Skip Oracle rubric generation, reuse existing synthetic_rubrics.jsonl"
      echo "  --rubrichub        Download & convert RubricHub data from HuggingFace"
      echo "  --skip-rubrichub   Skip RubricHub download (default)"
      echo "  --rebuild-split    Force rebuild HealthBench SFT/benchmark split"
      echo "  --no-preview       Don't print sample previews"
      exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

SFT_OUT="${ROOT_DIR}/data/sft_train.jsonl"
RL_OUT="${ROOT_DIR}/data/rl_train.parquet"
SYNTHETIC_JSONL="${ROOT_DIR}/data/synthetic_rubrics.jsonl"

echo "══════════════════════════════════════════════════════════"
echo " Data Preprocessing Pipeline"
echo "  limit per dataset : $LIMIT"
echo "  skip synthetic    : $SKIP_SYNTHETIC"
echo "  rubrichub         : $RUBRICHUB"
echo "  rebuild HB split  : $REBUILD_HB"
echo "══════════════════════════════════════════════════════════"

# ── Step 0 (optional): Download & convert RubricHub ──────────────────────
if [[ "$RUBRICHUB" == "1" ]]; then
  if [[ -f "$RUBRICHUB_OUT" ]]; then
    echo ""
    echo "[0/3] RubricHub already exists: $RUBRICHUB_OUT ($(wc -l < "$RUBRICHUB_OUT") rows)"
    echo "      Delete it to force re-download."
  else
    echo ""
    echo "[0/3] Downloading RubricHub from HuggingFace..."
    mkdir -p "$(dirname "$RUBRICHUB_OUT")"
    mkdir -p "${ROOT_DIR}/data/rubrichub_raw"
    "$PYTHON" -c "
from huggingface_hub import hf_hub_download, list_repo_tree
repo = 'sojuL/RubricHub_v1'
for subdir in ['sft_RuFT', 'RuRL']:
    for entry in list_repo_tree(repo, path_in_repo=subdir, repo_type='dataset'):
        if entry.path.endswith('.parquet'):
            print(f'Downloading {entry.path} ({entry.size/1e6:.0f} MB)...')
            hf_hub_download(repo, entry.path, repo_type='dataset',
                            local_dir='${ROOT_DIR}/data/rubrichub_raw')
print('Download complete')"
    echo "[0/3] Converting RubricHub to SFT JSONL..."
    "$PYTHON" -m src.data.prepare_rubrichub \
        --input-dir "${ROOT_DIR}/data/rubrichub_raw/RuRL" \
        --ruft-dir "${ROOT_DIR}/data/rubrichub_raw/sft_RuFT" \
        --output "$RUBRICHUB_OUT"
    echo "[0/3] Done → $RUBRICHUB_OUT ($(wc -l < "$RUBRICHUB_OUT") rows)"
  fi
fi

# ── Step 1: Generate synthetic rubrics ────────────────────────────────────
if [[ "$SKIP_SYNTHETIC" == "0" ]]; then
  echo ""
  echo "[1/3] Generating synthetic rubrics (Oracle reverse-engineering)..."
  "$PYTHON" -m src.data.generate_synthetic --limit "$LIMIT"
  echo "[1/3] Done → $SYNTHETIC_JSONL"
else
  echo ""
  if [[ -f "$SYNTHETIC_JSONL" ]]; then
    echo "[1/3] Skipping synthetic generation (reusing $SYNTHETIC_JSONL)"
  else
    echo "[1/3] Skipping synthetic generation (no existing file — HealthBench only)"
  fi
fi

# If synthetic data exists, pass the limit; otherwise tell downstream to skip it
SYNTHETIC_FLAG=()
if [[ -f "$SYNTHETIC_JSONL" ]]; then
  SYNTHETIC_FLAG=(--synthetic-limit "$LIMIT")
else
  SYNTHETIC_FLAG=(--no-synthetic)
fi

# ── Step 2: Prepare SFT data ─────────────────────────────────────────────
echo ""
echo "[2/3] Preparing SFT training data (HealthBench + synthetic)..."

SFT_ARGS=(
  -m src.training.sft_trainer
  --prepare-only
  --healthbench-limit "$LIMIT"
  "${SYNTHETIC_FLAG[@]}"
  --output-jsonl "$SFT_OUT"
  --preview-samples "$PREVIEW"
)
if [[ "$REBUILD_HB" == "1" ]]; then
  SFT_ARGS+=(--rebuild-healthbench-split)
fi
"$PYTHON" "${SFT_ARGS[@]}"
echo "[2/3] Done → $SFT_OUT"

# ── Step 3: Prepare RL data ──────────────────────────────────────────────
echo ""
echo "[3/4] Preparing RL training parquet (HealthBench + synthetic)..."

"$PYTHON" -m src.scripts.prepare_rl_data \
  --healthbench-limit "$LIMIT" \
  "${SYNTHETIC_FLAG[@]}" \
  --output "$RL_OUT"
echo "[3/4] Done → $RL_OUT"

# ── Step 4: Prepare RL validation parquet ────────────────────────────────
echo ""
echo "[4/4] Preparing RL validation parquet (HealthBench benchmark split)..."
"$PYTHON" -m src.scripts.prepare_rl_data --prepare-val --val-max-prompts 50
echo "[4/4] Done → data/rl_val_healthbench.parquet"

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════"
echo " All data ready!"
if [[ "$RUBRICHUB" == "1" ]]; then
  echo "  RubricHub : $RUBRICHUB_OUT"
fi
echo "  Synthetic : $SYNTHETIC_JSONL"
echo "  SFT data  : $SFT_OUT"
echo "  RL data   : $RL_OUT"
echo "  RL val    : data/rl_val_healthbench.parquet"
echo ""
echo " Next steps:"
echo "  SFT  → ./run_sft.sh"
echo "  RL   → ./run_rl.sh"
echo "══════════════════════════════════════════════════════════"

# ── Random sample preview ────────────────────────────────────────────────
if [[ "$PREVIEW" -gt 0 ]]; then
  echo ""
  "$PYTHON" -c "
import json, random, textwrap, sys
try:
    import pandas as pd
except ImportError:
    print('[preview] pandas not installed, skipping RL preview')
    pd = None

N = $PREVIEW

# --- SFT samples ---
sft_path = '$SFT_OUT'
try:
    with open(sft_path) as f:
        sft_rows = [json.loads(l) for l in f if l.strip()]
    samples = random.sample(sft_rows, min(N, len(sft_rows)))
    print('─' * 70)
    print(f'  Random SFT samples ({len(samples)} of {len(sft_rows)})')
    print('─' * 70)
    for i, r in enumerate(samples, 1):
        q = r.get('question', '')[:200]
        rubrics = r.get('rubrics', [])
        src = r.get('source', 'unknown')
        print(f'  [{i}] source={src}')
        print(f'      Q: {q}')
        for c in rubrics[:3]:
            pts = int(c.get('points', 0))
            print(f'        [{pts:+d}] {str(c.get(\"criterion\",\"\"))[:100]}')
        if len(rubrics) > 3:
            print(f'        ... +{len(rubrics)-3} more criteria')
        print()
except Exception as e:
    print(f'[preview] SFT preview failed: {e}')

# --- RL samples ---
rl_path = '$RL_OUT'
if pd is not None:
    try:
        df = pd.read_parquet(rl_path)
        idxs = random.sample(range(len(df)), min(N, len(df)))
        print('─' * 70)
        print(f'  Random RL samples ({len(idxs)} of {len(df)})')
        print('─' * 70)
        for i, idx in enumerate(idxs, 1):
            row = df.iloc[idx]
            src = row.get('data_source', 'unknown')
            prompt = row.get('prompt', [])
            if not isinstance(prompt, list):
                prompt = list(prompt) if hasattr(prompt, '__iter__') else []
            q_text = ''
            if len(prompt) > 0:
                q_text = str(prompt[0].get('content', '') if isinstance(prompt[0], dict) else prompt[0])[:200]
            resp = str(row.get('response', ''))[:200]
            print(f'  [{i}] source={src}')
            print(f'      Prompt: {q_text}')
            print(f'      Response: {resp}')
            print()
    except Exception as e:
        print(f'[preview] RL preview failed: {e}')
"
fi
