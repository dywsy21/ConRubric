# Project Plan: Meta-Reward GRM Training (Consensus-Based)

## 1. Overview

**Goal**: Train a specialized **General Reward Model (GRM)** that generates high-quality, robust evaluation rubrics (Principles) for open-ended and complex reasoning tasks.

**Core Hypothesis (Consensus/Robustness)**:
A "good" rubric is one that guides a solver to produce an answer that is universally high-quality. "Universally high-quality" is defined as scoring well not just against the generating rubric, but also against other plausible rubrics for the same question.
If a rubric $r_i$ leads to an answer $a_i$ that satisfies a diverse set of other valid rubrics $\{r_j\}$, then $r_i$ captures the essential, robust criteria for the task.

**Methodology**: Reinforcement Learning (RL) with a **Cross-Rubric Evaluation** reward signal.

## 2. Architecture & Components

### 2.1. Models

*   **General Reward Model (GRM) / Rubric Generator (Policy $\pi_\theta$)**:
    *   **Role**: Generates a Rubric $r$ given a Question $q$.
    *   **Base Model**: Qwen/Qwen3-4B-Instruct-2507.
    *   **Status**: **Trainable**.
*   **Solver (Environment)**:
    *   **Role**: Generates Answer $a$ given Question $q$ and Rubric $r$.
    *   **Model**: A pool of capable instruction-following models (e.g., Qwen-2.5-7B, Llama-3, etc.) to ensure generalization.
    *   **Status**: **Frozen**.
*   **Evaluator / Judge**:
    *   **Role**: Assigns a scalar score $S(a, r)$ to an answer $a$ based on rubric $r$.
    *   **Model**: Can be a strong frozen LLM (e.g., GPT-4o-mini, or a frozen copy of the Solver/GRM) prompted to act as a judge.
    *   **Status**: **Frozen**.

### 2.2. The Meta-Reward Mechanism (Cross-Rubric Evaluation)

For a given question $q$:

1.  **Rollout (Rubric Generation)**:
    *   The GRM generates $N$ candidate rubrics $\{r_1, r_2, ..., r_N\}$.
    *   *Assumption*: These rubrics represent different "views" or "complete scoring points" for the problem.

2.  **Execution (Answer Generation)**:
    *   For each rubric $r_i$, the Solver generates a corresponding answer $a_i$.
    *   $a_i \sim \pi_{solver}(\cdot | q, r_i)$.

3.  **Cross-Evaluation (The "Arena")**:
    *   We evaluate every answer $a_i$ against every *other* rubric $r_j$ (where $j \neq i$).
    *   Calculate Score Matrix $M_{ij} = \text{Evaluator}(q, a_i, r_j)$.
    *   $M_{ij}$ represents how well the answer generated from rubric $i$ satisfies the criteria of rubric $j$.

4.  **Reward Calculation**:
    *   The reward for rubric $r_i$ is the aggregate performance of its answer $a_i$ across the "consensus" of other rubrics.
    *   $R(r_i) = \text{Mean}(\{M_{ij} \mid j \neq i\})$.
    *   *Intuition*: If $r_i$ is a "bad" or "biased" rubric (e.g., "Answer must be in French"), $a_i$ will likely fail the criteria of other normal rubrics $r_j$ (which expect English/Correctness), resulting in a low Meta-Reward. If $r_i$ is "good", $a_i$ will be robust and score well everywhere.

## 3. Implementation Roadmap

### Phase 1: Data Preparation & Cold Start (SFT)

**Objective**: Initialize the GRM to generate coherent, valid rubrics.

1.  **Dataset**: Mix of Open-Ended (No Robots, Roleplay) and Deterministic (Math, Code).
2.  **Synthetic Data Generation**:
    *   Use Oracle (GPT-4o) to "reverse engineer" rubrics from `(Question, Gold Answer)` pairs.
    *   Create dataset: `(Question, Gold Rubric)`.
3.  **Supervised Fine-Tuning (SFT)**:
    *   Train GRM on `(Question, Gold Rubric)`.
    *   This ensures the model understands the format and concept of a "Rubric".

### Phase 2: Reinforcement Learning (Cross-Rubric Optimization)

**Objective**: Optimize GRM using the Cross-Rubric Meta-Reward.

1.  **Algorithm**: **DAPO** (Direct Alignment Policy Optimization).
    *   We use DAPO as a drop-in natural improvement over GRPO, leveraging its stability and efficiency for the $N$-output setup.
2.  **Training Loop**:
    *   **Sample**: Batch of questions $Q$.
    *   **Generate**: For each $q$, sample $N$ rubrics $\{r_1...r_N\}$.
    *   **Simulate**: Generate $N$ answers $\{a_1...a_N\}$ using the Frozen Solver.
    *   **Score**: Compute $N \times N$ score matrix using the Frozen Evaluator.
    *   **Reward**: Compute $R_i$ for each $r_i$.
    *   **Update**: Update GRM policy to maximize $R_i$.

### Phase 3: Evaluation & Analysis

1.  **Baselines**:
    *   Zero-shot, Generic Rubric, Self-Refine.
2.  **Metrics**:
    *   **Downstream Accuracy**: Pass@1 on MATH/HumanEval.
    *   **RM Benchmarks**: Evaluate the GRM's scoring capability on **RewardBench**, **PPE Preference**, **PPE Correctness**, and **RMB**.
    *   **Inference-time Scaling**: Test if Voting with Generated Rewards (using GRM scores) improves performance.
    *   **Rubric Quality**: Human evaluation. (maybe not needed)


## 4. Key Technical Considerations

*   **Computational Cost**:
    *   Old Plan: Inner Loop Training (Very Expensive).
    *   New Plan: $N$ generations + $N^2$ evaluations per question.
    *   *Optimization*: Keep $N$ small (e.g., $N=4$ or $N=5$). $N=4$ means 4 rubric gens, 4 answer gens, 12 evaluations (excluding diagonal). This is manageable.
*   **Evaluator Quality**:
    *   The signal depends on the Evaluator's ability to faithfully apply rubric $r_j$ to answer $a_i$.
    *   We need a robust prompt for the Evaluator.
*   **Rubric Diversity**:
    *   If the GRM collapses to generating $N$ identical rubrics, the cross-evaluation becomes meaningless (Self-Consistency = 100%).
    *   *Mitigation*: Add a KL-divergence penalty or an explicit diversity reward term (e.g., cosine distance between rubric embeddings).

## 5. Frameworks

*   **Training**: `verl` (VolcEngine RL) or `trl`.
*   **Inference**: `vLLM` for fast generation of rubrics and answers.
*   **Orchestration**: `Ray` to manage the parallel generation and evaluation tasks.
