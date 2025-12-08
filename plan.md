# Project Plan: Meta-Reward Rubric Generator

## 1. Overview

**Goal**: Train a specialized Large Language Model (Rubric Generator) to generate high-quality evaluation rubrics that guide downstream models (Solvers) to produce robust, high-quality answers.

**Core Hypothesis**: A "good" rubric is one that, when used to guide a solver, results in an answer that scores highly across *all possible* valid rubrics (approximated by a set of orthogonal "Anchor Rubrics").

**Methodology**: Bi-level optimization using Reinforcement Learning (RL), where the Rubric Generator is the policy and the Frozen Solver is part of the environment.

## 2. Architecture & Components (during RL-ing of the Rubric Generator)

### 2.1. Models

*   **Rubric Generator (Policy)**:
    *   Role: Generates a rubric $r$ given a question $q$.
    *   Base Model: Llama-3-8B-Instruct or similar mid-sized model.
    *   Status: Trainable (SFT -> RL).
*   **Solver (Environment)**:
    *   Role: Generates answer $a$ given question $q$ and rubric $r$.
    *   Base Model: Qwen-2.5-7B-Instruct or similar capable model. (maybe alternate between a set of different models?)
    *   Status: **Frozen** (Parameters fixed).
*   **Oracle / Judge (Reward Model)**:
    *   Role: Evaluates answer $a$ against a set of Anchor Rubrics.
    *   Model: The current most powerful model.

### 2.2. The Meta-Reward Function

To approximate "all possible rubrics", we define a set of **Anchor Rubrics** ($\mathcal{R}_{anchor}$) covering orthogonal dimensions:
1.  **Correctness**: Is the reasoning and final answer factually/logically correct?
2.  **Clarity/Conciseness**: Is the answer easy to understand and free of fluff?
3.  **Completeness**: Does it address all parts of the question?
4.  **Robustness/Safety**: Is the answer safe and robust to edge cases?
5.  ... (To be supplemented)


**Reward Calculation (Meta-Learning Objective)**:
For a generated rubric $r$:
1.  **Simulate RL Step (Inner Loop with DAPO)**:
    *   Generate rollouts $a \sim \pi_{solver}(\cdot|q)$.
    *   Score rollouts using rubric $r$ to obtain proxy rewards $R_{proxy}$.
    *   **Simulate DAPO Update**: Perform a simulated policy update $\pi_{solver} \to \pi'_{solver}$ using the **DAPO** algorithm (utilizing Decoupled Clipping and Dynamic Sampling) to maximize $R_{proxy}$.
2.  **Evaluate Improvement**:
    *   Measure the performance of the updated policy $\pi'_{solver}$ using the **Anchor Rubrics** (Ground Truth).
3.  **Meta-Reward**:
    *   $Reward(r) = \mathbb{E}_{a' \sim \pi'_{solver}}[\text{Score}_{anchor}(a')] - \mathbb{E}_{a \sim \pi_{solver}}[\text{Score}_{anchor}(a)]$.
    *   This measures the *alignment* of the generated rubric with the true objective by quantifying how much optimizing for $r$ improves the true score.

### 2.3 Justification for Meta-Reward Strategy

Why measure *Solver Improvement* instead of directly evaluating the generated rubric against the Anchor Rubrics?

1.  **Utility over Semantics**: A rubric might be semantically similar to an "Anchor Rubric" (e.g., "Be correct") but fail to provide the specific, actionable guidance a weaker solver needs to actually *be* correct. The Meta-Reward measures functional utility, not just surface-level quality.
2.  **Solver-Specific Optimization**: Different solvers have different weaknesses. The Meta-Reward encourages the generator to produce rubrics that patch the *specific* holes in the frozen solver's capabilities (e.g., "Pay extra attention to negative signs" for a model bad at arithmetic), which a generic "good rubric" might miss.
3.  **Avoiding Goodhart's Law**: If we reward the rubric solely on how well it matches a "Gold Standard Rubric", the generator might learn to mimic the style or keywords of the gold standard without capturing the substance needed for the specific problem.
4.  **End-to-End Alignment**: The ultimate goal is to generate high-quality answers. By optimizing the proxy metric (rubric quality) directly on the end metric (answer quality), we ensure perfect alignment between the training objective and the desired outcome.

### 2.4 Some thoughts

1. Maybe we can self-iterate? (use RL-ed Rubric Generator as the meta Rubric Generator for the next round of RL, include this in experiments section)
2. ...

## 3. Implementation Roadmap

### Phase 1: Data Preparation & Cold Start (SFT): Distilling + Existing High Quality (Question, Rubric) Pairs

**Objective**: Create a high-quality dataset to initialize the Rubric Generator.
1.  **Dataset Selection**:
    *   Focus on objective reasoning tasks: MATH (Mathematics), MBPP/HumanEval (Coding), ...
    *   Select ~5k-10k high-quality samples `(Question, Gold Answer)`.
2.  **Reverse Engineering (Synthetic Data)**:
    *   Use Oracle to "reverse engineer" the rubric that would lead to the Gold Answer.
    *   Prompt: "Given this Question and this perfect Answer, write the detailed Rubric that would guide a model to produce exactly this output."
    *   Output: `(Question, Gold Rubric)` pairs.
3.  **SFT Training**:
    *   Fine-tune the Rubric Generator on `(Question, Gold Rubric)`.
    *   Goal: Ensure the model understands the format and style of rubrics.

### Phase 2: Reinforcement Learning (RL)

**Objective**: Optimize the Generator to maximize the Meta-Reward.

1.  **Framework**: Use **verl** (VolcEngine Reinforcement Learning), a scalable RL framework for LLMs.
2.  **Algorithm**: **DAPO** (Direct Alignment Policy Optimization) is selected over GRPO for its superior sample efficiency and stability in reasoning tasks.
3.  **Key DAPO Configuration**:
    *   **Decoupled Clipping**: Use asymmetric clipping ratios (e.g., `clip_ratio_low: 0.2`, `clip_ratio_high: 0.28`) to stabilize updates.
    *   **Dynamic Sampling**: Enable `filter_groups` to ensure each training batch contains a mix of high and low-scoring samples, avoiding uninformative updates.
    *   **Token-level Loss**: Use `loss_agg_mode: "token-mean"` for more granular learning signals.
4.  **Training Loop (Bi-Level DAPO)**:
    *   **Outer Rollout (Generator)**:
        *   Sample batch of questions $q$.
        *   Generator produces rubrics $\{r_1, ..., r_k\}$.
    *   **Inner Loop Simulation (Meta-Reward Calculation)**:
        *   For each rubric $r_i$:
            *   **Inner Rollout**: Solver produces answers $A_{pre}$ guided by $r_i$.
            *   **Inner DAPO Update**: Simulate update $\pi_{solver} \to \pi'_{solver}$ using DAPO on $A_{pre}$.
            *   **Measure Improvement**: Evaluate $\pi'_{solver}$ vs $\pi_{solver}$ on Anchor Rubrics to get Meta-Reward $R_i$.
    *   **Outer Update (Generator)**:
        *   Update Generator parameters using **DAPO** (with Decoupled Clipping & Dynamic Sampling) based on Meta-Rewards $\{R_1, ..., R_k\}$.

### Phase 3: Evaluation & Analysis

1.  **Baselines**:
    *   **Zero-shot**: Solver prompted with "You are a helpful assistant."
    *   **Generic Rubric**: Solver prompted with a static, hand-crafted rubric.
    *   **Self-Refine**: Standard self-correction loop.
2.  **Metrics**:
    *   **Downstream Accuracy**: Pass@1 on MATH/HumanEval using the generated rubrics.
    *   **Rubric Quality**: Human evaluation of the generated rubrics (intelligibility, specificity).
    *   **Transferability**: Test if rubrics generated for one Solver work for another (e.g., Llama-generated rubric guiding a Mistral Solver).

## 4. Key Technical Challenges & Mitigations

*   **Reward Hacking**: The Generator might produce "jailbreak" rubrics that force the Solver to output specific keywords.
    *   *Mitigation*: Include a "Rubric Coherence" penalty or use a strong Judge that penalizes nonsensical rubrics.

## 5. Framework Selection

### 5.1. Training Framework: **verl**
We will use **verl** (VolcEngine Reinforcement Learning).
*   **Reasoning**: `verl` is designed for scalable RLHF with LLMs and natively supports the **DAPO** algorithm. It offers better performance and flexibility for complex RL loops compared to standard TRL.
*   **Key Feature**: It uses a modular architecture (Actor, Critic, Ref Model) built on **Ray**, allowing efficient distribution of the "Generator -> Solver -> Judge" pipeline.

### 5.2. Inference Engine: **vLLM** (via verl)
`verl` integrates with **vLLM** for efficient generation.
*   **Role**: Both the **Rubric Generator** (Actor) and the **Frozen Solver** (Environment) will utilize vLLM for high-throughput generation.
*   **Integration**: `verl` handles the vLLM backend configuration, allowing us to focus on the reward function logic.

### 5.3. Orchestration
*   **Ray**: `verl` uses Ray to manage the distributed components.
*   **Custom Environment/Reward**: We will implement a custom reward function in `verl` that encapsulates the **Inner Loop Simulation**:
    1.  Takes the generated rubric $r$.
    2.  **Simulate Inner DAPO**:
        *   Generate initial answers $A_{pre}$ from the Frozen Solver guided by $r$.
        *   Perform a simulated DAPO update to obtain $\pi'_{solver}$.
    3.  **Validation**: Generate answers $A_{post}$ from $\pi'_{solver}$ and score them against Anchor Rubrics.
    4.  Returns the performance improvement (Meta-Reward) to the Outer Generator.
