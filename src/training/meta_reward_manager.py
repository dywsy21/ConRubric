from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from src.training.meta_reward import MetaRewardFunction
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY

print("Imported meta_reward_manager.py")


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

        global_step = None
        if hasattr(data, "meta_info") and data.meta_info:
            global_step = data.meta_info.get("global_steps", None)
        if global_step is None:
            # Try batch-level non_tensor_batch
            for i in range(min(1, batch_size)):
                item = data[i]
                if hasattr(item, "meta_info") and item.meta_info:
                    global_step = item.meta_info.get("global_steps", None)
                    if global_step is not None:
                        break
        print(f"[RewardManager] global_step={global_step}, meta_info keys={list(data.meta_info.keys()) if hasattr(data, 'meta_info') and data.meta_info else 'N/A'}")
        scalar_rewards = self.reward_fn.compute_reward(
            questions, rubrics,
            global_step=global_step,
            response_token_counts=valid_response_lengths,
        )

        for i, score in enumerate(scalar_rewards):
            # Per-token reward: apply the same reward uniformly across
            # all valid response tokens.  This gives every token the
            # discrimination signal, making training more stable than
            # concentrating the entire reward at the last token.
            resp_len = valid_response_lengths[i]
            rewards[i, :resp_len] = float(score)
            reward_extra_info["score"].append(float(score))

        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": reward_extra_info}
        return rewards


# Idempotent registration – the module may be exec'd more than once by verl's
# importlib-based loader, so we guard against duplicate-registration errors.
if "meta_consensus" not in REWARD_MANAGER_REGISTRY:
    REWARD_MANAGER_REGISTRY["meta_consensus"] = MetaConsensusRewardManager
