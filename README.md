# Meta-Reward GRM

This project implements the training of a General Reward Model (GRM) using Meta-Reward and Bi-level optimization.

`verl` is vendored in-repo at `./verl`. It is effectively a forked and modified version.

## Setup

```bash
# install cuda 12.9 first, make sure nvcc, etc. are on the PATH
uv venv
uv pip install .

cp .env.example .env
# And then fill in the .env file

# Optional: download all relevant things first (models, datasets)
python -m src.utils.download

```

## Usage

Please `source .venv/bin/activate` first. 

### Data Preparation

Generate all training data (synthetic rubrics + SFT mix + RL parquet) in one command:

```bash
./run_data_preprocessing.sh --limit 100   # 100 samples per dataset
./run_data_preprocessing.sh --skip-synthetic  # reuse existing synthetic_rubrics.jsonl
```

This produces:
- `data/synthetic_rubrics.jsonl` — raw Oracle-generated rubrics
- `data/sft_train.jsonl` — weighted SFT mix (HealthBench + synthetic)
- `data/rl_train.parquet` — verl-compatible RL training data

### Phase 1: SFT (Cold Start)

```bash
./run_sft.sh
```

Fine-tunes the GRM on the weighted SFT mix with pre/post benchmarking.

### Phase 2: Reinforcement Learning

```bash
./run_rl.sh
```

Runs the DAPO RL loop using the vendored `verl` under `./verl`.
