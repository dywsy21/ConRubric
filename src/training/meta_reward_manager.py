from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np
import torch

from src.training.meta_reward import MetaRewardFunction
from src.training.rubric_quality import _CRITERION_RE, parse_rubric_text
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

    @staticmethod
    def _extract_gold_rubric(data_item) -> str:
        """Extract gold_rubric from reward_model.ground_truth."""
        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_model, dict):
            gt = reward_model.get("ground_truth")
            if isinstance(gt, dict):
                return str(gt.get("gold_rubric", "")).strip()
        return ""

    @staticmethod
    def _extract_gold_answer(data_item) -> str:
        """Extract gold_answer (ideal_completion) from reward_model.ground_truth."""
        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_model, dict):
            gt = reward_model.get("ground_truth")
            if isinstance(gt, dict):
                return str(gt.get("gold_answer", "")).strip()
        return ""

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        # Check if this is a validation call — use fast local metrics instead of Solver/Oracle
        is_validate = False
        if hasattr(data, "meta_info") and data.meta_info:
            is_validate = data.meta_info.get("validate", False)

        batch_size = len(data)
        rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        questions: list[str] = []
        rubrics: list[str] = []
        valid_response_lengths: list[int] = []
        gold_rubrics: list[str] = []
        gold_answers: list[str] = []

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
            gold_rubrics.append(self._extract_gold_rubric(item))
            gold_answers.append(self._extract_gold_answer(item))

        # ── Fast validation path (no Solver/Oracle API calls) ─────────
        if is_validate:
            return self._validate_fast(
                questions, rubrics, gold_rubrics, gold_answers, valid_response_lengths,
                rewards, batch_size, return_dict,
            )

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
            gold_rubrics=gold_rubrics,
            gold_answers=gold_answers,
            global_step=global_step,
            response_token_counts=valid_response_lengths,
        )

        for i, score in enumerate(scalar_rewards):
            resp_len = valid_response_lengths[i]
            dup_weights = self.reward_fn._dup_criterion_weights.get(i)
            has_dups = dup_weights and any(w < 1.0 - 1e-6 for w in dup_weights)

            if has_dups:
                # Token-level duplication penalty: tokens belonging to a
                # criterion repeated n times get reward / n.
                token_weights = self._compute_dup_token_weights(
                    rubrics[i], dup_weights, resp_len,
                )
                if token_weights is not None:
                    for j_tok in range(resp_len):
                        w = token_weights[j_tok] if j_tok < len(token_weights) else 1.0
                        rewards[i, j_tok] = float(score) * w
                    eff = sum(token_weights[:resp_len]) / max(resp_len, 1)
                    n_clusters = round(sum(dup_weights))
                    print(f"[RewardManager] rollout {i}: dup penalty applied, "
                          f"eff_weight={eff:.3f}, {n_clusters}/{len(dup_weights)} unique criteria")
                else:
                    rewards[i, :resp_len] = float(score)
            else:
                rewards[i, :resp_len] = float(score)

            reward_extra_info["score"].append(float(score))

            # GDPO: store per-component scalar rewards for decoupled normalization
            comp = self.reward_fn._component_rewards.get(i, (0.0, 0.0, 0.0, 0.0, 0.0))
            reward_extra_info["reward_consensus"].append(float(comp[0]))
            reward_extra_info["reward_disc"].append(float(comp[1]))
            reward_extra_info["reward_qa"].append(float(comp[2]))
            reward_extra_info["reward_gold_disc"].append(float(comp[3]))
            reward_extra_info["reward_calibration"].append(float(comp[4]))

        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": reward_extra_info}
        return rewards

    # ── Fast validation (no API calls) ─────────────────────────────────

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract lower-cased significant words (len>=4) from text."""
        words = re.findall(r"[a-zA-Z]{4,}", text.lower())
        # Exclude very common English stop words
        stop = {"this", "that", "with", "from", "have", "will", "been", "should",
                "could", "would", "which", "their", "there", "about", "into",
                "also", "does", "than", "when", "what", "were", "they", "some",
                "more", "most", "each", "only", "such", "very", "just", "like"}
        return {w for w in words if w not in stop}

    def _validate_fast(
        self,
        questions: list[str],
        rubrics: list[str],
        gold_rubrics: list[str],
        gold_answers: list[str],
        valid_response_lengths: list[int],
        rewards: torch.Tensor,
        batch_size: int,
        return_dict: bool,
    ):
        """Fast validation using local rubric quality metrics (no Solver/Oracle).

        Metrics computed per rubric:
        - n_criteria: number of parseable criteria
        - n_positive / n_negative: sign balance
        - avg_criterion_len: mean character length of criterion text
        - format_compliance: fraction of non-blank lines that parse as criteria
        - binary_sign_ratio: fraction of criteria with only [+] or [-] (no points)
        - gold_keyword_recall: fraction of gold rubric keywords covered
        - gold_keyword_precision: fraction of generated keywords that appear in gold
        - gold_keyword_f1: harmonic mean of precision and recall

        The composite score (used as reward for GDPO advantage) is gold_keyword_f1,
        which directly measures how well the generated rubric covers the gold rubric's
        content without needing Judge API calls.
        """
        reward_extra_info = defaultdict(list)

        for i in range(batch_size):
            rubric = rubrics[i]
            gold_rubric = gold_rubrics[i] if i < len(gold_rubrics) else ""
            resp_len = valid_response_lengths[i]

            criteria = parse_rubric_text(rubric)
            n_criteria = len(criteria)
            n_positive = sum(1 for c in criteria if c.sign == "+")
            n_negative = sum(1 for c in criteria if c.sign == "-")

            # Average criterion text length
            crit_lengths = [len(c.text) for c in criteria]
            avg_crit_len = float(np.mean(crit_lengths)) if crit_lengths else 0.0

            # Format compliance: how many non-blank lines are valid criteria
            non_blank = [ln for ln in rubric.splitlines() if ln.strip()]
            n_parseable = sum(1 for ln in non_blank if _CRITERION_RE.match(ln))
            format_compliance = n_parseable / max(len(non_blank), 1)

            # Gold keyword coverage
            gen_kw = self._extract_keywords(rubric)
            gold_kw = self._extract_keywords(gold_rubric) if gold_rubric else set()

            if gold_kw and gen_kw:
                overlap = gen_kw & gold_kw
                recall = len(overlap) / len(gold_kw)
                precision = len(overlap) / len(gen_kw)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            elif not gold_kw:
                recall, precision, f1 = 0.0, 0.0, 0.0
            else:
                recall, precision, f1 = 0.0, 0.0, 0.0

            # Composite score: weighted combination
            # f1 rewards content coverage; n_criteria in [3,15] is a soft indicator
            crit_count_ok = 1.0 if 3 <= n_criteria <= 15 else 0.0
            has_both_signs = 1.0 if n_positive > 0 and n_negative > 0 else 0.0
            composite = f1 * 5.0 + format_compliance + crit_count_ok + has_both_signs

            rewards[i, :resp_len] = composite

            reward_extra_info["score"].append(composite)
            reward_extra_info["val_n_criteria"].append(float(n_criteria))
            reward_extra_info["val_n_positive"].append(float(n_positive))
            reward_extra_info["val_n_negative"].append(float(n_negative))
            reward_extra_info["val_avg_crit_len"].append(avg_crit_len)
            reward_extra_info["val_format_compliance"].append(format_compliance)
            reward_extra_info["val_gold_kw_recall"].append(recall)
            reward_extra_info["val_gold_kw_precision"].append(precision)
            reward_extra_info["val_gold_kw_f1"].append(f1)
            reward_extra_info["val_crit_count_ok"].append(crit_count_ok)
            reward_extra_info["val_has_both_signs"].append(has_both_signs)

        print(f"[ValReward] Fast validation: {batch_size} rubrics, "
              f"mean_f1={np.mean(reward_extra_info['val_gold_kw_f1']):.3f}, "
              f"mean_recall={np.mean(reward_extra_info['val_gold_kw_recall']):.3f}, "
              f"mean_n_criteria={np.mean(reward_extra_info['val_n_criteria']):.1f}, "
              f"mean_format={np.mean(reward_extra_info['val_format_compliance']):.3f}")

        # ── Oracle-based rubric quality metrics (small subset) ─────────
        oracle_metrics = self._validate_oracle_subset(
            questions, rubrics, gold_answers, batch_size,
        )
        if oracle_metrics:
            for key, values in oracle_metrics.items():
                reward_extra_info[key] = values

        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": dict(reward_extra_info)}
        return rewards

    # ── Oracle-based validation metrics ─────────────────────────────────

    VAL_ORACLE_SUBSET = 10  # number of prompts to evaluate with Oracle

    def _validate_oracle_subset(
        self,
        questions: list[str],
        rubrics: list[str],
        gold_answers: list[str],
        batch_size: int,
    ) -> dict[str, list[float]]:
        """Score gold answer + mismatched answer against generated rubrics using Oracle.

        Uses a small subset of prompts with available gold answers.
        For each, scores the gold answer and a mismatched answer (from another question)
        to measure the rubric's discriminative power.

        Returns dict of metric_name -> list of batch_size values (padded with 0 for non-evaluated).
        """
        # Collect indices with both a non-empty rubric and gold answer
        eligible = [
            i for i in range(batch_size)
            if i < len(gold_answers) and gold_answers[i].strip()
            and rubrics[i].strip()
        ]
        if len(eligible) < 2:
            print("[ValReward] Oracle subset: <2 eligible prompts, skipping")
            return {}

        rng = np.random.default_rng(seed=42)
        subset = list(rng.choice(eligible, size=min(self.VAL_ORACLE_SUBSET, len(eligible)), replace=False))

        oracle = self.reward_fn.oracle
        gold_scores = [0.0] * batch_size
        mismatch_scores = [0.0] * batch_size
        top1_hits = [0.0] * batch_size
        resolutions = [0.0] * batch_size

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _score_pair(idx: int) -> tuple[int, float, float]:
            """Score gold answer and mismatched answer for prompt idx."""
            q = questions[idx]
            rubric = rubrics[idx]
            gold_ans = gold_answers[idx]

            # Pick a mismatched answer: gold answer from a different eligible prompt
            others = [j for j in eligible if j != idx and gold_answers[j].strip()]
            if not others:
                return idx, 0.0, 0.0
            mismatch_idx = others[rng.integers(len(others))]
            mismatch_ans = gold_answers[mismatch_idx]

            try:
                scores = oracle.evaluate_answers_by_rubric(q, [gold_ans, mismatch_ans], rubric)
                return idx, scores[0], scores[1]
            except Exception as e:
                print(f"[ValReward] Oracle error for prompt {idx}: {e}")
                return idx, 0.0, 0.0

        n_workers = min(len(subset), 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_score_pair, idx): idx for idx in subset}
            for fut in as_completed(futures):
                idx, gs, ms = fut.result()
                gold_scores[idx] = gs
                mismatch_scores[idx] = ms
                top1_hits[idx] = 1.0 if gs > ms else 0.0
                resolutions[idx] = gs - ms

        # Compute aggregate stats for the subset only
        subset_gold = [gold_scores[i] for i in subset]
        subset_top1 = [top1_hits[i] for i in subset]
        subset_res = [resolutions[i] for i in subset]

        mean_top1 = float(np.mean(subset_top1))
        mean_res = float(np.mean(subset_res))
        mean_gold = float(np.mean(subset_gold))

        print(f"[ValReward] Oracle subset ({len(subset)} prompts): "
              f"top1_acc={mean_top1:.3f}, "
              f"resolution={mean_res:.2f}, "
              f"gold_score={mean_gold:.2f}")

        # Broadcast aggregate values to all samples so process_validation_metrics
        # computes the correct mean (each uid sees the same aggregate value).
        return {
            "val_oracle_top1_acc": [mean_top1] * batch_size,
            "val_oracle_resolution": [mean_res] * batch_size,
            "val_oracle_gold_score": [mean_gold] * batch_size,
        }

    # ── Token-level duplication penalty ────────────────────────────────

    def _compute_dup_token_weights(
        self,
        response_str: str,
        crit_dup_weights: List[float],
        resp_token_len: int,
    ) -> Optional[List[float]]:
        """Map per-criterion duplication weights to per-token weights.

        For each criterion line in *response_str*, tokens belonging to a
        criterion in a Spearman-dedup cluster of size *c* get weight 1/c.
        Non-criterion tokens (preamble, blank lines, etc.) keep weight 1.0.

        Returns None if mapping fails (fallback to uniform reward).
        """
        # Step 1: build char-level weights ──────────────────────────────
        lines = response_str.split("\n")
        char_weights = [1.0] * (len(response_str) + 1)  # +1 safety
        crit_idx = 0
        char_offset = 0
        has_penalty = False
        for line in lines:
            line_start = char_offset
            line_end = char_offset + len(line)
            if _CRITERION_RE.match(line):
                if crit_idx < len(crit_dup_weights):
                    w = crit_dup_weights[crit_idx]
                    if w < 1.0 - 1e-6:
                        has_penalty = True
                        for c in range(line_start, min(line_end, len(response_str))):
                            char_weights[c] = w
                crit_idx += 1
            char_offset = line_end + 1  # +1 for '\n'

        if not has_penalty:
            return None  # no duplicates → uniform reward

        # Step 2: map chars → tokens via offset_mapping ─────────────────
        try:
            encoding = self.tokenizer(
                response_str,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = encoding["offset_mapping"]
            token_weights: List[float] = []
            for s, e in offsets:
                if s >= e:
                    token_weights.append(1.0)
                else:
                    # Use weight at the start of this token's char span
                    token_weights.append(char_weights[s])
            return token_weights
        except Exception:
            # Fallback: compute scalar discount from char fractions
            total = len(response_str) or 1
            weighted = sum(char_weights[:len(response_str)]) / total
            return [weighted] * resp_token_len


# Idempotent registration – the module may be exec'd more than once by verl's
# importlib-based loader, so we guard against duplicate-registration errors.
if "meta_consensus" not in REWARD_MANAGER_REGISTRY:
    REWARD_MANAGER_REGISTRY["meta_consensus"] = MetaConsensusRewardManager
