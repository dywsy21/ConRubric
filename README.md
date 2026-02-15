# Meta-Reward GRM

This project implements the training of a General Reward Model (GRM) using Meta-Reward and Bi-level optimization.

`verl` is vendored in-repo at `./verl` for direct modification and git tracking.

## Setup

1.  **Install Dependencies**:

```bash
# install cuda 12.9 first, make sure nvcc, etc. are on the PATH
uv venv
uv pip install .
```

2.  **Environment Variables**:
    
```bash
cp .env.example .env
# And then fill in the .env file
```

3.  **Download Resources**:

Download all required models and datasets to the local cache.

```bash
python -m src.utils.download
```

## Usage

Do a `source .venv/bin/activate` first. 

### Phase 1: Data Preparation & Cold Start

1.  **Generate Synthetic Data (Reverse Engineering)**:

```bash
python -m src.data.generate_synthetic --limit <N>  # Limit to N samples per dataset
```

This will generate `data/synthetic_rubrics.jsonl`.

2.  **SFT Training**:

```bash
python -m src.training.sft_trainer
```

This will fine-tune the GRM on the synthetic rubrics.

### Phase 2: Reinforcement Learning

1.  **Run RL Training**:

```bash
./run_training.sh --limit 999999999
```

This will start the RL loop using the vendored `verl` under `./verl`.

## Benchmark

```bash
python -m src.evaluation.run_benchmark --model_path <path_to_grm> --benchmarks rewardbench ppe rmb
```
