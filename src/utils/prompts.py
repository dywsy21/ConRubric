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
    "RULES:\n"
    "1. Every criterion must name a SPECIFIC fact, entity, or concept from the "
    "question — something you could NOT copy-paste into a rubric for a different "
    "question. Good: \"explains the role of A1C in diabetes monitoring\". "
    "Bad: \"provides clear and accurate information\".\n"
    "2. Do NOT start criteria with \"Models answer with …\" — this is a filler "
    "pattern. Write criteria about the CONTENT of the answer, not the model's "
    "tone or behavior.\n"
    "3. Do NOT use generic quality/tone words as standalone criteria "
    "(empathy, clarity, accuracy, compassion, supportive tone, actionable advice). "
    "If tone matters, tie it to a specific concern from the question.\n"
    "4. Output the rubric DIRECTLY — do not include any thinking, "
    "reflection, or commentary. Do not write </think> tags, \"Let me know\", "
    "\"I think this meets the criteria\", or any other meta-text. "
    "Every line of your output must be a criterion in the specified format.\n"
    "5. Each criterion must be UNIQUE — do not repeat the same criterion "
    "with minor rewording. If you find yourself writing the same idea again, stop.\n\n"
    "Output each criterion on its own line in this format:\n"
    "- [+] criterion (for positive criteria: what the answer should do)\n"
    "- [-] criterion (for negative criteria: mistakes or harmful behaviors)\n\n"
    "Question:\n{question}"
)

# ── Solver: answer a question guided by a rubric ─────────────────────────
SOLVER_PROMPT = """\
You are a helpful assistant. Please answer the following question.

Question:
{question}

Please ensure your answer follows these evaluation criteria:
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
Output Format:
[
    {{
        "criterion": "...",
        "points": 8
    }},
    {{
        "criterion": "...",
        "points": -6
    }}
]

Return JSON array only. Do not include markdown fences.
"""

DYNAMIC_RUBRIC_EVALUATION_PROMPT = """
Evaluate the Answer to the Question using the Rubric.

Question:
{question}

Answer:
{answer}

Rubric:
{rubric}

Output a single integer from 0 to 10 representing the overall score. Output ONLY the integer, nothing else.
"""

# ── Oracle: batch-evaluate multiple answers against ONE rubric ────────────
# Used in RL meta-reward: each rubric evaluates all other answers individually.
# Scores are absolute (not relative), allowing variance computation across answers.
BATCH_RUBRIC_EVALUATION_PROMPT = """
You are an expert evaluator. Score each of the {n} answers below on a 0-10 scale according to the rubric.

Scoring rules:
- For [+] criteria (positive): 10 = criterion fully met, 0 = completely absent
- For [-] criteria (negative): 0 = the bad behavior is fully present, 10 = the bad behavior is completely absent (good)
- Score each answer independently on its own merits.

Question:
{question}

Rubric:
{rubric}

{answers_block}

Output ONLY a JSON array of exactly {n} integers (0 to 10), one score per answer in the order shown above.
Do NOT output any explanation, markdown, or text — output ONLY the raw JSON array like [3, 7, 5, 2].
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
