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
                questions, rubrics, gold_rubrics, valid_response_lengths,
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

        # ── Oracle-based pairwise rubric quality metrics (meta_eval subset) ──
        oracle_metrics = self._validate_pairwise_oracle(
            questions, rubrics, batch_size,
        )
        if oracle_metrics:
            for key, values in oracle_metrics.items():
                reward_extra_info[key] = values

        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": dict(reward_extra_info)}
        return rewards

    # ── Oracle-based pairwise validation (HealthBench meta_eval) ────────

    VAL_ORACLE_PROMPTS = 8  # number of prompts to evaluate with Oracle
    _meta_eval_index = None  # lazily loaded: question_prefix -> list[{completion, label_score}]

    @classmethod
    def _load_meta_eval_index(cls) -> dict[str, list[dict]]:
        """Load benchmark_meta_eval.jsonl, grouped by question text prefix."""
        if cls._meta_eval_index is not None:
            return cls._meta_eval_index
        import json
        meta_path = "data/healthbench_splits/benchmark_meta_eval.jsonl"
        index: dict[str, list[dict]] = {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    prompt = row.get("prompt", [])
                    user_content = ""
                    if isinstance(prompt, list):
                        for msg in prompt:
                            if msg.get("role") == "user":
                                user_content = msg.get("content", "")
                                break
                    else:
                        user_content = str(prompt)
                    key = user_content[:150]
                    binary_labels = row.get("binary_labels", [])
                    label_score = float(sum(1 for x in binary_labels if x) / len(binary_labels)) if binary_labels else 0.0
                    if key not in index:
                        index[key] = []
                    index[key].append({
                        "completion": row.get("completion", ""),
                        "label_score": label_score,
                    })
            print(f"[ValReward] Loaded meta_eval index: {len(index)} prompts")
        except FileNotFoundError:
            print(f"[ValReward] WARNING: {meta_path} not found, Oracle pairwise validation disabled")
            index = {}
        cls._meta_eval_index = index
        return index

    def _validate_pairwise_oracle(
        self,
        questions: list[str],
        rubrics: list[str],
        batch_size: int,
    ) -> dict[str, list[float]]:
        """Evaluate generated rubrics using real HealthBench multi-grade completions.

        For a subset of val prompts that have meta_eval completions (multiple
        completions with physician-annotated label_scores), we use the Judge
        to score all completions against the generated rubric, then compute
        real pairwise_acc, resolution, and top_bottom_gap — the same metrics
        used in the full HealthBench rubric quality benchmark.
        """
        meta_index = self._load_meta_eval_index()
        if not meta_index:
            return {}

        # Match val questions to meta_eval by stripping "User: " prefix
        eligible = []  # list of (batch_idx, question_text, meta_key)
        for i in range(batch_size):
            q = questions[i]
            if not rubrics[i].strip():
                continue
            content = q[len("User: "):] if q.startswith("User: ") else q
            key = content[:150]
            if key in meta_index:
                completions = meta_index[key]
                labels = [c["label_score"] for c in completions]
                # Need at least 2 distinct label values for pairwise metrics
                if len(set(labels)) >= 2:
                    eligible.append((i, q, key))

        if not eligible:
            print("[ValReward] Oracle pairwise: no eligible prompts with meta_eval completions")
            return {}

        rng = np.random.default_rng(seed=42)
        n_select = min(self.VAL_ORACLE_PROMPTS, len(eligible))
        selected_indices = rng.choice(len(eligible), size=n_select, replace=False)
        selected = [eligible[j] for j in selected_indices]

        oracle = self.reward_fn.oracle
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from scipy.stats import spearmanr
        import math

        per_prompt_pairwise_acc = []
        per_prompt_resolution = []
        per_prompt_spearman = []
        per_prompt_top_bottom = []

        def _eval_prompt(batch_idx: int, question: str, meta_key: str):
            """Score all meta_eval completions for one prompt against the generated rubric."""
            completions_data = meta_index[meta_key][:8]  # cap at 8
            rubric = rubrics[batch_idx]
            completions = [c["completion"] for c in completions_data]
            label_scores = [c["label_score"] for c in completions_data]

            try:
                pred_scores = oracle.evaluate_batch(
                    questions=[question] * len(completions),
                    answers=completions,
                    rubrics=[rubric] * len(completions),
                    max_workers=1,
                )
                pred_scores = [float(s) for s in pred_scores]
            except Exception as e:
                print(f"[ValReward] Oracle pairwise error for prompt {batch_idx}: {e}")
                return None

            # Pairwise accuracy and resolution (same as benchmark)
            correct = 0.0
            total = 0
            gap_sum = 0.0
            for ii in range(len(pred_scores)):
                for jj in range(ii + 1, len(pred_scores)):
                    if label_scores[ii] == label_scores[jj]:
                        continue
                    total += 1
                    gap_sum += abs(pred_scores[ii] - pred_scores[jj])
                    label_order = label_scores[ii] > label_scores[jj]
                    score_order = pred_scores[ii] > pred_scores[jj]
                    if pred_scores[ii] == pred_scores[jj]:
                        correct += 0.5
                    elif label_order == score_order:
                        correct += 1.0

            pairwise_acc = correct / total if total > 0 else None
            resolution = gap_sum / total if total > 0 else None

            # Spearman correlation
            unique_labels = set(label_scores)
            spearman_rho = None
            if len(unique_labels) >= 2 and len(pred_scores) >= 2:
                rho, _ = spearmanr(pred_scores, label_scores)
                if not math.isnan(rho):
                    spearman_rho = rho

            # Top-bottom gap
            top_bottom = None
            if len(unique_labels) >= 2:
                best_label = max(unique_labels)
                worst_label = min(unique_labels)
                top_preds = [s for s, y in zip(pred_scores, label_scores) if y == best_label]
                bot_preds = [s for s, y in zip(pred_scores, label_scores) if y == worst_label]
                if top_preds and bot_preds:
                    top_bottom = (sum(top_preds) / len(top_preds)) - (sum(bot_preds) / len(bot_preds))

            return pairwise_acc, resolution, spearman_rho, top_bottom

        # Run in parallel (4 workers, each prompt has ~8 sequential Judge calls)
        n_workers = min(len(selected), 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_eval_prompt, bi, q, mk): bi
                for bi, q, mk in selected
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                pa, res, sp, tb = result
                if pa is not None:
                    per_prompt_pairwise_acc.append(pa)
                if res is not None:
                    per_prompt_resolution.append(res)
                if sp is not None:
                    per_prompt_spearman.append(sp)
                if tb is not None:
                    per_prompt_top_bottom.append(tb)

        if not per_prompt_pairwise_acc:
            print("[ValReward] Oracle pairwise: all prompts failed")
            return {}

        mean_pa = float(np.mean(per_prompt_pairwise_acc))
        mean_res = float(np.mean(per_prompt_resolution)) if per_prompt_resolution else 0.0
        mean_sp = float(np.mean(per_prompt_spearman)) if per_prompt_spearman else 0.0
        mean_tb = float(np.mean(per_prompt_top_bottom)) if per_prompt_top_bottom else 0.0

        print(f"[ValReward] Oracle pairwise ({len(per_prompt_pairwise_acc)} prompts): "
              f"pairwise_acc={mean_pa:.3f}, resolution={mean_res:.2f}, "
              f"spearman={mean_sp:.3f}, top_bottom_gap={mean_tb:.2f}")

        # Broadcast aggregate values to all samples
        return {
            "val_oracle_pairwise_acc": [mean_pa] * batch_size,
            "val_oracle_resolution": [mean_res] * batch_size,
            "val_oracle_spearman": [mean_sp] * batch_size,
            "val_oracle_top_bottom_gap": [mean_tb] * batch_size,
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
