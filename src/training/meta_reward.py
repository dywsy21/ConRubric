"""
Meta-reward function for RL training of the GRM (rubric generator).

Redesigned reward system with two core signals:
  1. Variance-based discrimination reward: rubrics that produce varied scores
     across different answers are more discriminative and useful.
  2. Spearman redundancy penalty: rubrics whose score rankings highly correlate
     with another rubric's rankings are redundant and get penalized.

Plus a quality adjustment for rubric length (criteria count + token length).

All rewards are applied per-token (uniform across all response tokens).
"""

import json
import os
import re
import random
import threading
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from scipy import stats as scipy_stats
from src.evaluation.judge import Oracle
from src.models.solver import Solver
from src.training.rubric_quality import (
    RubricQualityConfig,
    score_rubric_quality,
)

# Per-service parallel settings
DEFAULT_SOLVER_WORKERS = int(os.environ.get("GRM_SOLVER_WORKERS", 4))
DEFAULT_ORACLE_WORKERS = int(os.environ.get("GRM_ORACLE_WORKERS", 4))

# Maximum number of solver answers per question group (0 = same as N)
SOLVER_N = int(os.environ.get("GRM_SOLVER_N", "0"))

# Rubric quality scoring
RUBRIC_QUALITY_CONFIG = RubricQualityConfig()

# Rollout logging directory
ROLLOUT_LOG_DIR = os.environ.get("GRM_ROLLOUT_LOG_DIR", "out/rl/rollout_logs")

# ── Discrimination reward parameters ──────────────────────────────────────
# Scale factor for variance-based reward.
# reward_disc = DISC_SCALE * sqrt(variance_of_scores)
# Using sqrt to prevent reward from exploding with very high variance.
DISC_SCALE = float(os.environ.get("GRM_DISC_SCALE", "1.0"))

# ── Spearman redundancy parameters ────────────────────────────────────────
# Threshold for Spearman correlation above which two rubrics are considered
# redundant.  Each redundant rubric's reward is divided by dup_count.
SPEARMAN_THRESHOLD = float(os.environ.get("GRM_SPEARMAN_THRESHOLD", "0.8"))


class MetaRewardFunction:
    def __init__(self, solver_model_name: str, oracle_model_name: str,
                 oracle_api_key: str = None, oracle_api_base: str = None,
                 solver_remote: bool = False, solver_api_key: str = None, solver_api_base: str = None):
        self.config = {
            "solver_model_name": solver_model_name,
            "oracle_model_name": oracle_model_name,
            "oracle_api_key": oracle_api_key,
            "oracle_api_base": oracle_api_base,
            "solver_remote": solver_remote,
            "solver_api_key": solver_api_key,
            "solver_api_base": solver_api_base
        }
        self._oracle = None
        self._solver = None
        self._step_counter = 0

    @property
    def oracle(self):
        if self._oracle is None:
            self._oracle = Oracle(
                model_name=self.config["oracle_model_name"],
                api_key=self.config["oracle_api_key"],
                api_base=self.config["oracle_api_base"]
            )
        return self._oracle

    @property
    def solver(self):
        if self._solver is None:
            self._solver = Solver(
                model_name=self.config["solver_model_name"],
                is_remote=self.config["solver_remote"],
                api_key=self.config["solver_api_key"],
                api_base=self.config["solver_api_base"]
            )
        return self._solver

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_oracle"] = None
        state["_solver"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_env(cls) -> "MetaRewardFunction":
        from src.config import ProjectConfig

        cfg = ProjectConfig()
        return cls(
            solver_model_name=cfg.solver_model_name,
            oracle_model_name=cfg.oracle_model_name,
            oracle_api_key=cfg.oracle_api_key,
            oracle_api_base=cfg.oracle_api_base,
            solver_remote=cfg.solver_remote,
            solver_api_key=cfg.solver_api_key,
            solver_api_base=cfg.solver_api_base,
        )

    def compute_reward(self, questions: List[str], rubrics: List[str],
                       solver_workers: int = None,
                       oracle_workers: int = None,
                       global_step: int = None,
                       response_token_counts: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computes per-rubric reward using variance-based discrimination
        and Spearman redundancy penalty.

        Pipeline per question (N rollouts → N rubrics):
            1. Generate N answers via Solver (one per rubric)
            2. Each rubric j evaluates all N-1 other answers → N-1 scores
            3. Discrimination reward = DISC_SCALE * sqrt(var(scores_j))
            4. Spearman redundancy: for each pair (j, k), if Spearman > threshold,
               both are redundant.  dup_count = number of correlated rubrics.
            5. Final reward = disc_reward / dup_count + quality_adjustment
            6. Reward applied uniformly to all response tokens (per-token)
        """
        if solver_workers is None:
            solver_workers = DEFAULT_SOLVER_WORKERS
        if oracle_workers is None:
            oracle_workers = DEFAULT_ORACLE_WORKERS

        solver_n = SOLVER_N

        # Use global_step if provided
        if global_step is not None:
            self._step_counter = global_step

        print(f"[MetaReward] Computing rewards for {len(questions)} samples "
              f"(global_step={self._step_counter}), "
              f"solver_workers={solver_workers}, oracle_workers={oracle_workers}, "
              f"solver_n={solver_n or 'all'}, "
              f"disc_scale={DISC_SCALE}, spearman_threshold={SPEARMAN_THRESHOLD}, "
              f"rubric_quality={RUBRIC_QUALITY_CONFIG.enabled}")

        # ── Group by question ──────────────────────────────────────────
        q_to_rubrics: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for idx, (q, r) in enumerate(zip(questions, rubrics)):
            q_to_rubrics[q].append((idx, r))

        q_items = list(q_to_rubrics.items())
        print(f"[MetaReward] Processing {len(q_items)} unique questions")

        rewards = torch.zeros(len(questions), dtype=torch.float32)

        # ── Pre-compute rubric quality adjustments ─────────────────────
        quality_adjustments = np.zeros(len(questions), dtype=np.float32)
        if RUBRIC_QUALITY_CONFIG.enabled:
            _quality_details = []
            for idx, rubric in enumerate(rubrics):
                tc = response_token_counts[idx] if response_token_counts else 0
                result = score_rubric_quality(rubric, RUBRIC_QUALITY_CONFIG, token_count=tc)
                quality_adjustments[idx] = result.total_adjustment
                _quality_details.append(result.detail)
                if result.detail.get("token_length_penalty", 0.0) < -0.01:
                    print(f"[MetaReward]   rubric {idx}: {tc} tokens, "
                          f"token_len_penalty={result.detail['token_length_penalty']:.3f}")
            _mean_adj = float(np.mean(quality_adjustments))
            _mean_tpen = np.mean([d.get("token_length_penalty", 0.0) for d in _quality_details])
            _mean_lpen = np.mean([d.get("length_penalty", 0.0) for d in _quality_details])
            print(f"[MetaReward] quality_adj_mean={_mean_adj:.3f}, "
                  f"length_penalty_mean={_mean_lpen:.3f}, "
                  f"token_len_penalty_mean={_mean_tpen:.3f}")

        # Per-question answer storage for rollout logging
        question_answers: Dict[int, Dict[int, str]] = {}

        # Shared thread-pools
        solver_pool = ThreadPoolExecutor(max_workers=max(1, solver_workers))
        oracle_pool = ThreadPoolExecutor(max_workers=max(1, oracle_workers))

        # Track progress
        progress_lock = threading.Lock()
        progress = {"solver_done": 0, "solver_total": 0,
                     "eval_done": 0, "eval_total": 0,
                     "q_done": 0, "q_total": len(q_items)}

        def _log_progress(kind: str):
            with progress_lock:
                progress[f"{kind}_done"] += 1
                sd, st = progress["solver_done"], progress["solver_total"]
                ed, et = progress["eval_done"], progress["eval_total"]
                qd, qt = progress["q_done"], progress["q_total"]
            total_done = sd + ed
            if total_done % 4 == 0 or (sd == st and ed == et):
                print(f"[MetaReward]   solver={sd}/{st}  eval={ed}/{et}  questions={qd}/{qt}")

        def _gen_answer(q: str, rubric: str) -> str:
            ans = self.solver.generate_answer(q, rubric)
            _log_progress("solver")
            return ans

        def _eval_batch_by_rubric(q: str, answers: List[str], rubric: str) -> List[float]:
            """Returns len(answers) scores using the oracle."""
            scores = self.oracle.evaluate_answers_by_rubric(q, answers, rubric)
            with progress_lock:
                progress["eval_done"] += len(answers)
            return scores

        # ── Phase 1: Submit solver tasks ───────────────────────────────
        solver_futures: Dict[int, List[Tuple[int, Future]]] = {}
        solver_answer_indices: Dict[int, List[int]] = {}

        for q_idx, (q, items) in enumerate(q_items):
            n = len(items)
            if solver_n > 0 and n > solver_n:
                selected = sorted(random.sample(range(n), solver_n))
            else:
                selected = list(range(n))
            solver_answer_indices[q_idx] = selected

            with progress_lock:
                progress["solver_total"] += len(selected)

            futs = []
            for local_i in selected:
                _, rubric = items[local_i]
                fut = solver_pool.submit(_gen_answer, q, rubric)
                futs.append((local_i, fut))
            solver_futures[q_idx] = futs

        # ── Phase 2: Coordinate evaluation per question ────────────────
        eval_futures: Dict[int, List] = {}  # q_idx -> [(j, answer_indices, future)]
        coordinator_futures: List[Future] = []
        coordinator_pool = ThreadPoolExecutor(max_workers=len(q_items))

        def _coordinate_question(q_idx: int, q: str, items: List[Tuple[int, str]], n: int):
            """Wait for solver results, then submit oracle evals."""
            # Collect answers
            answers = {}  # local_i -> answer text
            for local_i, fut in solver_futures[q_idx]:
                try:
                    answers[local_i] = fut.result(timeout=900)
                except Exception as e:
                    print(f"[MetaReward] Solver error Q{q_idx} rubric {local_i}: {e}")
                    answers[local_i] = ""

            # Store answers for logging
            question_answers[q_idx] = dict(answers)

            if n < 2:
                # Single rubric — no cross-evaluation possible
                if answers:
                    with progress_lock:
                        progress["eval_total"] += 1
                    fut = oracle_pool.submit(
                        _eval_batch_by_rubric, q, [answers.get(0, "")], items[0][1]
                    )
                    eval_futures[q_idx] = [(0, [0], fut)]
                else:
                    eval_futures[q_idx] = []
                return

            answer_keys = sorted(answers.keys())
            if not answer_keys:
                eval_futures[q_idx] = []
                return

            # Each rubric j evaluates all other available answers
            total_evals = 0
            batch_futs = []
            for j in range(n):
                available = [ai for ai in answer_keys if ai != j]
                if not available:
                    available = answer_keys[:]
                selected_answers = [answers[ai] for ai in available]
                total_evals += len(selected_answers)
                rubric_j = items[j][1]
                fut = oracle_pool.submit(
                    _eval_batch_by_rubric, q, selected_answers, rubric_j
                )
                batch_futs.append((j, available, fut))

            with progress_lock:
                progress["eval_total"] += total_evals
            eval_futures[q_idx] = batch_futs

        for q_idx, (q, items) in enumerate(q_items):
            n = len(items)
            cfut = coordinator_pool.submit(_coordinate_question, q_idx, q, items, n)
            coordinator_futures.append(cfut)

        # ── Phase 3: Collect results, compute discrimination + redundancy ──
        for cfut in coordinator_futures:
            cfut.result()

        for q_idx, (q, items) in enumerate(q_items):
            indices = [it[0] for it in items]  # global indices
            n = len(items)

            if n < 2:
                # Single rubric: use raw oracle score + quality adjustment
                if eval_futures.get(q_idx):
                    j, answer_indices, fut = eval_futures[q_idx][0]
                    try:
                        scores = fut.result(timeout=900)
                        raw_reward = scores[0] if scores else 0.0
                    except Exception as e:
                        print(f"[MetaReward] Eval error Q{q_idx}: {e}")
                        raw_reward = 0.0
                    qa = quality_adjustments[indices[0]] if RUBRIC_QUALITY_CONFIG.enabled else 0.0
                    rewards[indices[0]] = raw_reward + qa
            else:
                # Build score vectors: score_vectors[j][k] = score rubric j
                # gave to answer k (answers from other rubrics)
                score_vectors: Dict[int, Dict[int, float]] = {}

                for j, answer_indices, fut in eval_futures[q_idx]:
                    try:
                        scores = fut.result(timeout=900)
                        score_map = {}
                        for pos, ai in enumerate(answer_indices):
                            if pos < len(scores):
                                score_map[ai] = scores[pos]
                        score_vectors[j] = score_map
                    except Exception as e:
                        print(f"[MetaReward] Eval error Q{q_idx} rubric {j}: {e}")
                        score_vectors[j] = {}

                # ── Discrimination reward: variance of each rubric's scores ──
                disc_rewards = np.zeros(n, dtype=np.float64)
                for j in range(n):
                    sv = score_vectors.get(j, {})
                    score_list = list(sv.values())
                    if len(score_list) >= 2:
                        var = np.var(score_list)
                        disc_rewards[j] = DISC_SCALE * np.sqrt(var)
                    else:
                        disc_rewards[j] = 0.0

                # ── Spearman redundancy penalty ──────────────────────────
                # For each pair (j, k), compute Spearman on common answers.
                # If rho > threshold, increment dup_count for both.
                dup_counts = np.ones(n, dtype=np.int32)  # start at 1 (self)

                if n >= 3:  # need ≥3 rubrics for meaningful redundancy
                    for j in range(n):
                        for k in range(j + 1, n):
                            sv_j = score_vectors.get(j, {})
                            sv_k = score_vectors.get(k, {})
                            common = sorted(set(sv_j.keys()) & set(sv_k.keys()))
                            if len(common) < 3:
                                continue
                            vals_j = [sv_j[ai] for ai in common]
                            vals_k = [sv_k[ai] for ai in common]
                            try:
                                rho, _ = scipy_stats.spearmanr(vals_j, vals_k)
                                if np.isnan(rho):
                                    continue
                                if rho > SPEARMAN_THRESHOLD:
                                    dup_counts[j] += 1
                                    dup_counts[k] += 1
                            except Exception:
                                continue

                # ── Final reward per rubric ───────────────────────────────
                for i in range(n):
                    disc_r = disc_rewards[i] / dup_counts[i]
                    qa = quality_adjustments[indices[i]] if RUBRIC_QUALITY_CONFIG.enabled else 0.0
                    rewards[indices[i]] = disc_r + qa

                # Log details
                _disc_mean = np.mean(disc_rewards)
                _dup_mean = np.mean(dup_counts.astype(float))
                _reward_vals = [rewards[idx].item() for idx in indices]
                print(f"[MetaReward]   Q{q_idx}: disc_mean={_disc_mean:.3f}, "
                      f"dup_count_mean={_dup_mean:.1f}, "
                      f"rewards={[f'{r:.2f}' for r in _reward_vals]}")

            with progress_lock:
                progress["q_done"] += 1

            print(f"[MetaReward]   Q{q_idx+1}/{len(q_items)} done")

        # ── Cleanup ────────────────────────────────────────────────────
        solver_pool.shutdown(wait=False)
        oracle_pool.shutdown(wait=False)
        coordinator_pool.shutdown(wait=False)

        # ── Log sample rubrics ─────────────────────────────────────────
        self._log_sample_rubrics(q_items, rewards)

        # ── Rollout logging ────────────────────────────────────────────
        log_step = global_step if global_step is not None else self._step_counter
        if global_step is None:
            self._step_counter += 1
        self._save_rollout_log(
            step=log_step,
            q_items=q_items,
            question_answers=question_answers,
            rewards=rewards,
        )

        print(f"[MetaReward] All rewards computed, mean={rewards.mean():.3f}")
        return rewards

    def _log_sample_rubrics(
        self,
        q_items: List[Tuple[str, List[Tuple[int, str]]]],
        rewards: torch.Tensor,
    ):
        """Print sample (question, rubrics) pairs to stdout for qualitative monitoring."""
        n_to_show = min(1, len(q_items))
        print(f"\n{'='*80}")
        print(f"[RubricSample] Step {self._step_counter} — "
              f"{len(q_items)} questions, showing {n_to_show} full example(s)")
        print(f"{'='*80}")

        for q_idx in range(n_to_show):
            q, items = q_items[q_idx]
            print(f"\n[Q{q_idx+1}] {q[:300]}")
            print(f"  ({len(items)} rubrics generated)")
            for local_i, (global_idx, rubric) in enumerate(items):
                r = rewards[global_idx].item()
                rubric_preview = rubric[:500].replace('\n', '\n    ')
                print(f"  --- Rubric {local_i+1} (reward={r:.3f}) ---")
                print(f"    {rubric_preview}")
                if len(rubric) > 500:
                    print(f"    ... ({len(rubric)} chars total)")

        if len(q_items) > n_to_show:
            print(f"\n[RubricSample] Other questions summary:")
            for q_idx in range(n_to_show, len(q_items)):
                q, items = q_items[q_idx]
                rews = [rewards[it[0]].item() for it in items]
                lens = [len(it[1]) for it in items]
                print(f"  Q{q_idx+1}: \"{q[:80]}...\" "
                      f"| {len(items)} rubrics | rewards={[f'{r:.2f}' for r in rews]} "
                      f"| lengths={lens}")

        print(f"{'='*80}\n")

    def _save_rollout_log(
        self,
        step: int,
        q_items: List[Tuple[str, List[Tuple[int, str]]]],
        question_answers: Dict[int, Dict[int, str]],
        rewards: torch.Tensor,
    ):
        """Save per-step rollout log: question, rubrics, answers, rewards."""
        log_dir = Path(ROLLOUT_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"step_{step:04d}.jsonl"

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                for q_idx, (q, items) in enumerate(q_items):
                    answers_dict = question_answers.get(q_idx, {})
                    rollouts = []
                    for local_i, (global_idx, rubric) in enumerate(items):
                        rollouts.append({
                            "rubric": rubric[:2000],
                            "answer": answers_dict.get(local_i, "")[:2000],
                            "reward": round(rewards[global_idx].item(), 4),
                        })
                    record = {
                        "question": q[:500],
                        "n_rollouts": len(items),
                        "rollouts": rollouts,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[MetaReward] Rollout log saved: {log_file}")
        except Exception as e:
            print(f"[MetaReward] Failed to save rollout log: {e}")

    def _generate_answer(self, question: str, rubric: str) -> str:
        return self.solver.generate_answer(question, rubric)
