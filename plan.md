# Project Plan: Meta-Reward GRM Training

## 1. Overview

**Goal**: Train a specialized **General Reward Model (GRM)** that generates high-quality evaluation rubrics (Principles) to guide downstream models (Solvers) and provides accurate pointwise scoring.

**Core Hypothesis**: A "good" rubric is one that, when used to guide a solver, results in an answer that scores highly across *all possible* valid rubrics. By training a GRM to generate these "good" rubrics, we simultaneously enhance its reasoning and evaluation capabilities.

**Methodology**: Bi-level optimization using Reinforcement Learning (DAPO), where the GRM (Policy) generates rubrics and the Frozen Solver (Environment) generates answers.

## 2. Architecture & Components (during RL-ing of the GRM)

### 2.1. Models

*   **General Reward Model (GRM) / Rubric Generator**:
    *   **Type**: Pointwise, Generative Reward Model.
    *   **Architecture**: Single model that outputs `Principle (Rubric) -> Critique -> Score`.
    *   **Role**:
        1.  **Generator**: Generates `Principle` (Rubric) $r$ given question $q$.
        2.  **Evaluator**: Generates `Critique` and `Score` given $(q, a, r)$.
    *   **Base Model**: qwen3-4B-instruct
    *   **Status**: Trainable (SFT -> RFT -> RL).
*   **Solver (Environment)**:
    *   Role: Generates answer $a$ given question $q$ and rubric $r$.
    *   Base Model: Qwen-2.5-7B-Instruct or similar capable model. (maybe alternate between a set of different models?)
    *   Status: **Frozen** (Parameters fixed).
*   **Oracle / Judge (Ground Truth)**:
    *   Role: Evaluates answer $a$ against a set of Anchor Rubrics to provide the "Meta-Reward".
    *   Model: The current most powerful model (e.g., GPT-4o, DeepSeek-V3).

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
    *   Score rollouts using rubric $r$ and the current GRM to obtain proxy rewards $R_{proxy}$.
    *   **Simulate DAPO Update**: Perform $N_{step}$ simulated policy updates $\pi_{solver} \to \pi'_{solver}$ using the **DAPO** algorithm (utilizing Decoupled Clipping and Dynamic Sampling) to maximize $R_{proxy}$.
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

### Phase 1: Data Preparation & Cold Start (SFT + RFT)

**Objective**: Create a high-quality dataset to initialize the GRM using Distillation and Rejective Fine-Tuning (RFT).

1.  **Dataset Selection**:
    *   Focus on open tasks that essentially don't have "golden truths". (e.g., creative writing, role playing, etc.)
    *   Select ~5k-10k high-quality samples `(Question, Gold Answer)`.
2.  **Reverse Engineering (Synthetic Data)**:
    *   Use Oracle to "reverse engineer" the rubric that would lead to the Gold Answer.
    *   Output: `(Question, Gold Rubric)` pairs.
3.  **SFT Training**:
    *   Fine-tune the GRM on `(Question, Gold Rubric)`.
4.  **Rejective Fine-Tuning (RFT)**:
    *   Generate $N$ candidate rubrics for each question using the SFT-ed model.
    *   **Filter**:
        *   Use Solver to generate answers for each rubric.
        *   Score answers using Oracle.
        *   Keep rubrics that lead to high-quality answers (matching Gold Answer performance).
    *   **Train**: Fine-tune on the filtered high-quality `(Question, Rubric)` pairs.

### Phase 2: Reinforcement Learning (RL)

**Objective**: Optimize the GRM to maximize the Meta-Reward using **verl** and **DAPO**.

1.  **Framework**: **verl** (VolcEngine Reinforcement Learning).
2.  **Algorithm**: **DAPO** (Direct Alignment Policy Optimization).
3.  **Key DAPO Configuration**:
    *   **Decoupled Clipping**: Asymmetric clipping ratios.
    *   **Dynamic Sampling**: Enable `filter_groups` to ensure contrastive batches.
    *   **Token-level Loss**: `loss_agg_mode: "token-mean"`.
4.  **Training Loop (Bi-Level DAPO)**:
    *   **Outer Rollout (GRM Policy)**:
        *   Sample batch of questions $q$.
        *   GRM generates rubrics (Principles) $\{r_1, ..., r_k\}$ for each $q$.
    *   **Inner Loop DAPO Simulation (Meta-Reward Calculation)**:
        *   For each rubric $r_i$:
            *   **Data Collection**: Frozen Solver $\pi_{solver}$ generates a batch of answers $A$ for $q$.
            *   **Proxy Scoring**: GRM evaluates $A$ using rubric $r_i$, producing scores $S_{proxy}$.
            *   **Inner DAPO Step**: Simulate $N_{step}$ policy updates $\pi_{solver} \to \pi'_{solver}$ using **DAPO** to maximize $S_{proxy}$.
            *   **Meta-Evaluation**: Evaluate the performance of the updated policy $\pi'_{solver}$ (e.g., by scoring its outputs) against the **Anchor Rubrics** (Oracle).
            *   **Reward Assignment**: The improvement in Anchor Score (or absolute Anchor Score of $\pi'_{solver}$) becomes the Meta-Reward $R_i$ for rubric $r_i$.
    *   **Outer Update (GRM Policy)**:
        *   Update GRM parameters using **DAPO** (with Decoupled Clipping & Dynamic Sampling) to maximize the expected Meta-Reward, treating $\{r_1, ..., r_k\}$ as the rollout and $\{R_1, ..., R_k\}$ as the rewards.

### Phase 3: Evaluation & Analysis

1.  **Baselines**:
    *   Zero-shot, Generic Rubric, Self-Refine.
2.  **Metrics**:
    *   **Downstream Accuracy**: Pass@1 on MATH/HumanEval.
    *   **RM Benchmarks**: Evaluate the GRM's scoring capability on **RewardBench**, **PPE Preference**, **PPE Correctness**, and **RMB**.
    *   **Inference-time Scaling**: Test if Voting with Generated Rewards (using GRM scores) improves performance.
    *   **Rubric Quality**: Human evaluation. (maybe not needed)

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
