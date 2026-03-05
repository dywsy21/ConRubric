from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import torch

from src.training.meta_reward import MetaRewardFunction
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY

print("Imported meta_reward_manager.py")


def _is_garbage_rubric(text: str, debug: bool = False) -> bool:
    """Detect garbage / degenerate rubrics that should receive reward=0.

    Criteria (any → garbage):
      1. Near-empty: fewer than 50 non-whitespace characters.
      2. Repetitive character spam: a single character repeated 10+ times in a
         row dominates > 60 % of the text (e.g. "||||||||||||" or "----------").
      3. No parseable criteria: no bullet points (*/-), numbered lists, or
         point markers like [+10], [-5], [criterion], etc.
      4. Extremely low character diversity: unique chars / len < 0.03 (after
         collapsing whitespace).
    """
    stripped = text.strip()

    # 1. Too short to be a real rubric
    non_ws = re.sub(r"\s", "", stripped)
    if len(non_ws) < 50:
        if debug:
            print(f"  [GARBRE] Too short: {len(non_ws)} chars")
        return True

    # 2. Repetitive single-char spam (very generous threshold)
    runs = re.findall(r"(.)\1{9,}", stripped)  # char repeated 10+ times
    total_run_len = sum(len(m) for m in runs)
    if total_run_len > 0.6 * len(stripped):
        if debug:
            print(f"  [GARBRE] Repetitive spam: {total_run_len}/{len(stripped)}")
        return True

    # 3. No parseable criterion markers - be very permissive
    # Accept: bullets, numbers, point markers like [+10], [-5], [criterion X], etc.
    has_criterion = bool(re.search(
        r"(?:[-*•]\s*\[?[+-]?\d|^\d+[\.\)]|\[criterion|points:|score:|rubric:)",
        stripped,
        re.IGNORECASE | re.MULTILINE,
    ))
    if not has_criterion:
        # Fallback: look for any bracketed point value
        has_criterion = bool(re.search(r"\[[+-]\d+\]", stripped))
    if not has_criterion:
        # Final fallback: any line with "criterion" or "points" keyword
        has_criterion = bool(re.search(
            r"(?:criterion|points|score|rubric)",
            stripped,
            re.IGNORECASE,
        ))
    if not has_criterion:
        if debug:
            print(f"  [GARBRE] No criterion markers found")
            print(f"    Text preview: {stripped[:100]!r}")
        return True

    # 4. Very low character diversity (extremely permissive)
    unique_chars = len(set(non_ws.lower()))
    if unique_chars / max(len(non_ws), 1) < 0.03:
        if debug:
            print(f"  [GARBRE] Low diversity: {unique_chars}/{len(non_ws)}={unique_chars/max(len(non_ws), 1):.3f}")
        return True

    return False


class MetaConsensusRewardManager(AbstractRewardManager):
    """Batch-level meta reward manager for cross-rubric consensus training."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **_: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.reward_fn = MetaRewardFunction.from_env()

    @staticmethod
    def _to_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and "question" in value:
            return str(value["question"])
        return ""

    def _extract_question(self, data_item) -> str:
        # Preferred path: reward_model.ground_truth.question
        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
            question = self._to_text(ground_truth)
            if question:
                return question

        # Fallback path: extra_info.question
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, dict):
            question = self._to_text(extra_info.get("question", ""))
            if question:
                return question

        return ""

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        batch_size = len(data)
        rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        questions: list[str] = []
        rubrics: list[str] = []
        valid_response_lengths: list[int] = []

        for i in range(batch_size):
            item = data[i]

            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(item.batch["attention_mask"][:prompt_len].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_response_len = int(item.batch["attention_mask"][prompt_len:].sum().item())
            valid_response_ids = response_ids[:valid_response_len]

            # Decode generated rubric
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # Extract source question from ground_truth / extra_info. Last fallback is decoded prompt.
            question = self._extract_question(item)
            if not question:
                question = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            questions.append(question)
            rubrics.append(response_str)
            valid_response_lengths.append(max(valid_response_len, 1))

        # ── Garbage rubric pre-filter ──────────────────────────────────
        # Fast pre-filter catches only truly degenerate rubrics (empty, spam).
        # Real garbage detection is delegated to the Solver model, which
        # outputs <GARBAGE_RUBRIC> when it receives a nonsensical rubric.
        garbage_mask = [_is_garbage_rubric(r) for r in rubrics]
        n_garbage = sum(garbage_mask)
        if n_garbage:
            print(f"[RewardManager] Pre-filter: {n_garbage}/{batch_size} degenerate rubrics → reward=0")
            for i, is_garb in enumerate(garbage_mask):
                if is_garb:
                    print(f"  [pre-filter {i}] len={len(rubrics[i].strip())} {rubrics[i][:80]!r}...")

        # Build filtered lists for non-garbage rubrics
        valid_indices = [i for i, g in enumerate(garbage_mask) if not g]

        if valid_indices:
            valid_questions = [questions[i] for i in valid_indices]
            valid_rubrics = [rubrics[i] for i in valid_indices]
            valid_scalar_rewards = self.reward_fn.compute_reward(valid_questions, valid_rubrics)
        else:
            valid_scalar_rewards = torch.tensor([])

        # Merge back: garbage → 0, valid → computed reward
        scalar_rewards = torch.zeros(batch_size, dtype=torch.float32)
        for out_idx, orig_idx in enumerate(valid_indices):
            scalar_rewards[orig_idx] = valid_scalar_rewards[out_idx]

        for i, score in enumerate(scalar_rewards):
            rewards[i, valid_response_lengths[i] - 1] = float(score)
            reward_extra_info["score"].append(float(score))

        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": reward_extra_info}
        return rewards


# Idempotent registration – the module may be exec'd more than once by verl's
# importlib-based loader, so we guard against duplicate-registration errors.
if "meta_consensus" not in REWARD_MANAGER_REGISTRY:
    REWARD_MANAGER_REGISTRY["meta_consensus"] = MetaConsensusRewardManager
