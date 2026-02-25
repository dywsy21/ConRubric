#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run_env_preparing.sh — Universal environment setup for GRM
#
# Creates Python venv (via uv), installs all dependencies, and verifies
# GPU access.  Idempotent: safe to re-run on any server.
#
# Usage:
#   ./run_env_preparing.sh                     # Full setup
#   ./run_env_preparing.sh --check-only        # Only verify environment
#   ./run_env_preparing.sh --python-version 3.11  # Use specific Python
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CHECK_ONLY=0

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)       CHECK_ONLY=1; shift ;;
    --python-version)   PYTHON_VERSION="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--check-only] [--python-version X.Y]"
      echo ""
      echo "  --check-only       Only verify the existing environment"
      echo "  --python-version   Python version to use (default: 3.12)"
      exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

# ── Load .env for proxy / CUDA settings ───────────────────────────────────
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

# ── Apply proxy if configured ─────────────────────────────────────────────
if [[ -n "${PROXY_URL:-}" ]]; then
  export ALL_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL" HTTP_PROXY="$PROXY_URL"
  export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
  export no_proxy="${NO_PROXY:-localhost,127.0.0.1}"
  echo "[env] Proxy configured: $PROXY_URL"
fi

# ── Detect CUDA ───────────────────────────────────────────────────────────
if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "[env] CUDA_HOME=$CUDA_HOME"
elif [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "[env] Auto-detected CUDA_HOME=/usr/local/cuda"
fi

if command -v nvidia-smi &>/dev/null; then
  echo "[env] GPU info:"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
else
  echo "[warn] nvidia-smi not found — no GPU detected"
fi

# ── Check-only mode ───────────────────────────────────────────────────────
if [[ "$CHECK_ONLY" == "1" ]]; then
  echo ""
  echo "── Verifying existing environment ──"
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[FAIL] venv not found at $VENV_DIR"
    exit 1
  fi
  source "$VENV_DIR/bin/activate"
  python -c "
import torch
print(f'  torch        : {torch.__version__}')
print(f'  CUDA         : {torch.version.cuda}')
print(f'  GPU count    : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {p.name} ({p.total_mem / 1e9:.1f} GB)')
"
  python -c "
import flash_attn; print(f'  flash_attn   : {flash_attn.__version__}')
import vllm;      print(f'  vllm         : {vllm.__version__}')
import verl;      print(f'  verl         : OK')
import transformers; print(f'  transformers : {transformers.__version__}')
import peft;      print(f'  peft         : {peft.__version__}')
"
  echo ""
  echo "All checks passed ✓"
  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
#  Step 1/4: Install uv package manager
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " [1/4] Checking uv package manager"
echo "══════════════════════════════════════════════════════════"
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  echo "uv installed: $(uv --version)"
else
  echo "uv already installed: $(uv --version)"
fi

# ══════════════════════════════════════════════════════════════════════════
#  Step 2/4: Create Python virtual environment
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " [2/4] Creating Python $PYTHON_VERSION virtual environment"
echo "══════════════════════════════════════════════════════════"
if [[ -d "$VENV_DIR" ]]; then
  EXISTING_VER=$("$VENV_DIR/bin/python" --version 2>/dev/null | awk '{print $2}' || echo "unknown")
  echo "venv already exists at $VENV_DIR (Python $EXISTING_VER)"
else
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
  echo "Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ══════════════════════════════════════════════════════════════════════════
#  Step 3/4: Install project dependencies
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " [3/4] Installing project dependencies"
echo "══════════════════════════════════════════════════════════"
cd "$ROOT_DIR"
uv pip install -e .

# ══════════════════════════════════════════════════════════════════════════
#  Step 4/4: Verify installation
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
echo " [4/4] Verifying installation"
echo "══════════════════════════════════════════════════════════"
python -c "
import torch
print(f'  torch        : {torch.__version__}')
print(f'  CUDA         : {torch.version.cuda}')
print(f'  GPU count    : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {p.name} ({p.total_mem / 1e9:.1f} GB)')
"
python -c "
try:
    import flash_attn; print(f'  flash_attn   : {flash_attn.__version__}')
except ImportError:
    print('  flash_attn   : NOT INSTALLED (optional)')
import vllm;         print(f'  vllm         : {vllm.__version__}')
import verl;         print(f'  verl         : OK')
import transformers; print(f'  transformers : {transformers.__version__}')
import peft;         print(f'  peft         : {peft.__version__}')
"

echo ""
echo "══════════════════════════════════════════════════════════"
echo " Environment ready!"
echo "  Activate : source $VENV_DIR/bin/activate"
echo "  Verify   : $0 --check-only"
echo "  Next     : ./run_data_preprocessing.sh"
echo "══════════════════════════════════════════════════════════"
