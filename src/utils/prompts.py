# ══════════════════════════════════════════════════════════════════════════
# Centralised prompt templates for the entire project.
#
# Every module that needs a prompt should import from here so that changes
# propagate everywhere and SFT / RL / inference stay consistent.
# ══════════════════════════════════════════════════════════════════════════

# ── GRM rubric‑generation instruction (used in SFT, RL, and inference) ───
RUBRIC_GENERATION_PROMPT = (
    "Generate an evaluation rubric of 6–12 criteria for the following question. "
    "Include BOTH positive criteria (what the answer should do) AND negative criteria "
    "(common mistakes or harmful behaviors to penalize). Do not repeat or paraphrase "
    "the same idea across multiple criteria.\n\n"
    "Output each criterion on its own line in this format:\n"
    "- [+/-points] criterion | tags: ...\n\n"
    "Question:\n{question}"
)

# ── Solver: answer a question guided by a rubric ─────────────────────────
SOLVER_PROMPT = """\
You are a helpful assistant. Please answer the following question.

Question:
{question}

Please ensure your answer follows these principles:
{rubric}

Answer:
"""

# ── Oracle: reverse‑engineer a rubric from (question, gold_answer) ───────
REVERSE_ENGINEER_RUBRIC_PROMPT = """
You are an expert evaluator and instruction designer.
Given the following Question and a high-quality Gold Answer, your task is to "reverse engineer" the evaluation rubric (Principles) that would guide a model to produce this specific answer.

Question:
{question}

Gold Answer:
{gold_answer}

Please generate a concise, actionable, and specific Rubric (set of Principles) that captures the key requirements, style, and constraints satisfied by the Gold Answer.

Scoring requirements:
1. Each criterion must include an integer `points` in range [-10, 10], excluding 0.
2. Include BOTH positive and negative criteria in the same rubric:
   - positive criteria (points > 0): what should be done
   - negative criteria (points < 0): common mistakes / harmful / missing behaviors to penalize
3. Prefer 6-12 total criteria.
4. `tags` is optional but recommended.

Output Format:
[
    {{
        "criterion": "...",
        "points": 8,
        "tags": ["accuracy", "completeness"]
    }},
    {{
        "criterion": "...",
        "points": -6,
        "tags": ["hallucination", "safety"]
    }}
]

Return JSON array only. Do not include markdown fences.
"""

DYNAMIC_RUBRIC_EVALUATION_PROMPT = """
Evaluate the Answer to the Question using the Rubric. Score 0-10.

Question:
{question}

Answer:
{answer}

Rubric:
{rubric}

End your response with exactly: Score: N
Where N is an integer from 0 to 10.
"""

# ── Oracle: batch-evaluate multiple answers against ONE rubric ────────────
# Used in RL meta-reward to get *relative* scores: seeing all answers at once
# lets the judge calibrate scores against each other, making the ranking signal
# much more meaningful than scoring each answer in isolation.
BATCH_RUBRIC_EVALUATION_PROMPT = """
Rate {n} answers 0-10 using the rubric. Compare them for calibration.

Question:
{question}

Rubric:
{rubric}

{answers_block}

Reply ONLY with a JSON array of exactly {n} integers.
Format example (THIS IS ONLY A FORMAT ILLUSTRATION, NOT ACTUAL SCORES — you must judge independently): {example_scores}
"""


# ANCHOR_EVALUATION_PROMPT = """
# You are an expert judge. Evaluate the following Answer to the Question based on the provided Anchor Rubrics.

# Question:
# {question}

# Answer:
# {answer}

# Anchor Rubrics:
# 1. Correctness: Is the reasoning and final answer factually/logically correct?
# 2. Clarity/Conciseness: Is the answer easy to understand and free of fluff?
# 3. Completeness: Does it address all parts of the question?
# 4. Robustness/Safety: Is the answer safe and robust to edge cases?

# Please provide a score from 0 to 10 for each dimension and an overall score.

# Output JSON:
# {{
#     "correctness": <score>,
#     "clarity": <score>,
#     "completeness": <score>,
#     "safety": <score>,
#     "overall": <score>,
#     "reasoning": "<brief explanation>"
# }}
# """
