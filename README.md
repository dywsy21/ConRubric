# Meta-Reward GRM

This project implements the training of a General Reward Model (GRM) using Meta-Reward and Bi-level optimization.

## Setup

1.  **Install Dependencies**:

```bash
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
uv python -m src.utils.download
```

## Usage

### Phase 1: Data Preparation & Cold Start

1.  **Generate Synthetic Data (Reverse Engineering)**:

```bash
uv python -m src.data.generate_synthetic --limit 100
```

This will generate `data/synthetic_rubrics.jsonl`.

2.  **SFT Training**:

```bash
uv python -m src.training.sft_trainer
```

This will fine-tune the GRM on the synthetic rubrics.

### Phase 2: Reinforcement Learning

1.  **Run RL Training**:

```bash
uv python -m src.training.rl_trainer
```

This will start the RL loop using `verl` (requires `verl` to be installed).

## Project Structure

*   `src/data`: Data processing and generation.
*   `src/evaluation`: Oracle and Judge implementations.
*   `src/training`: SFT and RL training scripts.
*   `src/utils`: Utility functions and prompts.
