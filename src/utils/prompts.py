# Prompts for the Oracle / Judge

REVERSE_ENGINEER_RUBRIC_PROMPT = """
You are an expert evaluator and instruction designer.
Given the following Question and a high-quality Gold Answer, your task is to "reverse engineer" the evaluation rubric (Principles) that would guide a model to produce this specific answer.

Question:
{question}

Gold Answer:
{gold_answer}

Please generate a concise, actionable, and specific Rubric (set of Principles) that captures the key requirements, style, and constraints satisfied by the Gold Answer.
The Rubric should be formatted as a list of principles.

Output Format:
[
    "Principle 1...",
    "Principle 2...",
    ...
]
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
