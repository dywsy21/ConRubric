#!/bin/bash
# Training launcher script
# Fixes GLIBCXX version mismatch between conda and system libraries

set -e

# Use system libstdc++ to fix GLIBCXX_3.4.32 requirement
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Disable uvloop for Ray to ensure nest_asyncio works correctly
export RAY_DISABLE_UVLOOP=1

# Activate virtual environment
source "$(dirname "$0")/.venv/bin/activate"

# Load .env variables
set -a
source "$(dirname "$0")/.env"
set +a

# Run training
python -m src.training.verl_main "$@"
