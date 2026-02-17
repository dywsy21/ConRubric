# Prompts for the Oracle / Judge

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
    {
        "criterion": "...",
        "points": 8,
        "tags": ["accuracy", "completeness"]
    },
    {
        "criterion": "...",
        "points": -6,
        "tags": ["hallucination", "safety"]
    }
]

Return JSON array only. Do not include markdown fences.
"""

ANCHOR_EVALUATION_PROMPT = """
You are an expert judge. Evaluate the following Answer to the Question based on the provided Anchor Rubrics.

Question:
{question}

Answer:
{answer}

Anchor Rubrics:
1. Correctness: Is the reasoning and final answer factually/logically correct?
2. Clarity/Conciseness: Is the answer easy to understand and free of fluff?
3. Completeness: Does it address all parts of the question?
4. Robustness/Safety: Is the answer safe and robust to edge cases?

Please provide a score from 0 to 10 for each dimension and an overall score.

Output JSON:
{{
    "correctness": <score>,
    "clarity": <score>,
    "completeness": <score>,
    "safety": <score>,
    "overall": <score>,
    "reasoning": "<brief explanation>"
}}
"""

DYNAMIC_RUBRIC_EVALUATION_PROMPT = """
You are an expert judge. Evaluate the following Answer to the Question based strictly on the provided Rubric.

Question:
{question}

Answer:
{answer}

Rubric:
{rubric}

Please evaluate how well the answer satisfies the criteria in the Rubric.
Provide a score from 0 to 10, where 0 is complete failure and 10 is perfect adherence.

Output JSON:
{{
    "score": <score>,
    "reasoning": "<brief explanation>"
}}
"""
