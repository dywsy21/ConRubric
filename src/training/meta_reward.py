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
    parse_rubric_text,
    score_rubric_quality,
)

# Strip legacy "| tags: ..." suffixes from generated rubric text
_TAGS_SUFFIX_RE = re.compile(r'\s*\|\s*tags?\s*:.*$', re.IGNORECASE | re.MULTILINE)

# Strip <think>...</think> blocks from instruct/thinking model output
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)

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

# ── Semantic diversity penalty ────────────────────────────────────────────
# Penalizes rubrics with low semantic uniqueness (e.g., mirrored [+]/[-] pairs).
# penalty = -LAMBDA_DIVERSITY * max(0, 1 - unique_ratio / diversity_target)
# where unique_ratio = n_unique_clusters / n_total_criteria.
# Default target: 60% unique.  Rubrics below this threshold are penalized.
LAMBDA_DIVERSITY = float(os.environ.get("GRM_LAMBDA_DIVERSITY", "1.5"))
DIVERSITY_TARGET = float(os.environ.get("GRM_DIVERSITY_TARGET", "0.6"))

# ── Spearman redundancy parameters ────────────────────────────────────────
# Threshold for Spearman correlation above which two criteria within the same
# rollout are considered redundant.  Redundant criteria are counted only once
# when computing the consensus score.
SPEARMAN_THRESHOLD = float(os.environ.get("GRM_SPEARMAN_THRESHOLD", "0.8"))


def _find_unique_criteria(
    crit_scores: Dict[int, Dict[int, float]],
    threshold: float,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Find non-redundant criteria via Spearman correlation (union-find).

    Within a single rollout, criteria whose score vectors are highly
    correlated (rho > threshold) are merged into one cluster.  Returns one
    representative criterion index per cluster.

    Args:
        crit_scores: {criterion_idx: {answer_idx: score}}
        threshold: Spearman rho above which two criteria are redundant

    Returns:
        Tuple of:
          - Sorted list of representative criterion indices.
          - Dict mapping each representative to its full cluster members.
    """
    all_k = sorted(crit_scores.keys())
    if len(all_k) <= 1:
        clusters = {k: [k] for k in all_k}
        return all_k, clusters

    # Union-Find
    parent = {k: k for k in all_k}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i_idx, ki in enumerate(all_k):
        for kj in all_k[i_idx + 1:]:
            sv_i = crit_scores.get(ki, {})
            sv_j = crit_scores.get(kj, {})
            common = sorted(set(sv_i.keys()) & set(sv_j.keys()))
            if len(common) < 3:
                continue
            vals_i = [sv_i[a] for a in common]
            vals_j = [sv_j[a] for a in common]
            # Skip if either vector is constant (correlation undefined)
            if len(set(vals_i)) < 2 or len(set(vals_j)) < 2:
                continue
            try:
                rho, _ = scipy_stats.spearmanr(vals_i, vals_j)
                if not np.isnan(rho) and rho > threshold:
                    union(ki, kj)
            except Exception:
                continue

    # One representative per component (smallest index)
    components = defaultdict(list)
    for k in all_k:
        components[find(k)].append(k)
    reps = sorted(min(comp) for comp in components.values())
    clusters = {min(comp): sorted(comp) for comp in components.values()}
    return reps, clusters


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
        # Per-criterion duplication weights for token-level reward scaling.
        # Populated by compute_reward: {global_idx: [1/cluster_size_k, ...]}
        self._dup_criterion_weights: Dict[int, List[float]] = {}

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
        Computes per-rollout reward using per-criterion evaluation.

        Pipeline per question (N rollouts → N rubric sets):
            1. Generate N answers via Solver (one per rollout/rubric-set)
            2. Parse each rollout's rubric text into M individual criteria
            3. Each criterion c_k of rollout j evaluates all N-1 other answers
               → score vector of length N-1
            4. Spearman dedup within rollout: criteria with rho > threshold
               are clustered; only one representative per cluster counts
            5. Consensus: for each other answer, mean of unique criteria scores
               → consensus_j = mean across other answers
            6. Discrimination: for each unique criterion,
               disc_k = sqrt(var(scores_k)) → disc_j = mean(disc_k) * DISC_SCALE
            7. Final reward = consensus + disc + quality_adjustment
            8. Reward applied uniformly to all response tokens (per-token)
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
            # Strip <think>...</think> blocks (instruct/thinking models)
            r_clean = _THINK_BLOCK_RE.sub('', r)
            # Strip legacy "| tags: ..." suffixes from generated rubric text
            r_clean = _TAGS_SUFFIX_RE.sub('', r_clean)
            q_to_rubrics[q].append((idx, r_clean))

        q_items = list(q_to_rubrics.items())
        print(f"[MetaReward] Processing {len(q_items)} unique questions")

        rewards = torch.zeros(len(questions), dtype=torch.float32)
        self._dup_criterion_weights = {}  # reset per batch

        # ── Pre-compute rubric quality adjustments ─────────────────────
        quality_adjustments = np.zeros(len(questions), dtype=np.float32)
        if RUBRIC_QUALITY_CONFIG.enabled:
            _quality_details = []
            for idx, rubric in enumerate(rubrics):
                # Strip <think> blocks before quality scoring
                rubric_clean = _THINK_BLOCK_RE.sub('', rubric)
                tc = response_token_counts[idx] if response_token_counts else 0
                result = score_rubric_quality(rubric_clean, RUBRIC_QUALITY_CONFIG, token_count=tc)
                quality_adjustments[idx] = result.total_adjustment
                _quality_details.append(result.detail)
                if result.detail.get("token_length_penalty", 0.0) < -0.01:
                    print(f"[MetaReward]   rubric {idx}: {tc} tokens, "
                          f"token_len_penalty={result.detail['token_length_penalty']:.3f}")
            _mean_adj = float(np.mean(quality_adjustments))
            _mean_tpen = np.mean([d.get("token_length_penalty", 0.0) for d in _quality_details])
            _mean_lpen = np.mean([d.get("length_penalty", 0.0) for d in _quality_details])
            _mean_mpen = np.mean([d.get("malformed_penalty", 0.0) for d in _quality_details])
            print(f"[MetaReward] quality_adj_mean={_mean_adj:.3f}, "
                  f"length_penalty_mean={_mean_lpen:.3f}, "
                  f"token_len_penalty_mean={_mean_tpen:.3f}, "
                  f"malformed_penalty_mean={_mean_mpen:.3f}")

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
        # eval_futures: q_idx -> [(j, k, answer_indices, future)]
        #   j = rollout index, k = criterion index within rollout
        eval_futures: Dict[int, List] = {}
        # Store parsed criterion texts per rollout for logging
        rollout_criterion_texts: Dict[int, Dict[int, List[str]]] = {}  # q_idx -> {j -> [texts]}
        coordinator_futures: List[Future] = []
        coordinator_pool = ThreadPoolExecutor(max_workers=len(q_items))

        def _coordinate_question(q_idx: int, q: str, items: List[Tuple[int, str]], n: int):
            """Wait for solver results, then submit per-criterion oracle evals."""
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
                # Single rollout — no cross-evaluation possible
                eval_futures[q_idx] = []
                return

            answer_keys = sorted(answers.keys())
            if not answer_keys:
                eval_futures[q_idx] = []
                return

            # Parse each rollout's rubric into individual criteria,
            # then submit one oracle call per criterion
            total_evals = 0
            batch_futs = []
            if q_idx not in rollout_criterion_texts:
                rollout_criterion_texts[q_idx] = {}
            for j in range(n):
                _, rubric_text = items[j]
                criteria = parse_rubric_text(rubric_text)
                if criteria:
                    criterion_texts = [f"- [{c.sign}] {c.text}" for c in criteria]
                else:
                    # NO FALLBACK: if no valid criteria parsed, skip oracle evals.
                    # The reward for this rollout will be quality_adjustment only
                    # (which is strongly negative due to 0-criteria penalty).
                    criterion_texts = []
                rollout_criterion_texts[q_idx][j] = criterion_texts

                # Other answers (exclude answer j)
                other_keys = [ai for ai in answer_keys if ai != j]
                if not other_keys:
                    other_keys = answer_keys[:]
                other_answers = [answers[ai] for ai in other_keys]

                for k, crit_text in enumerate(criterion_texts):
                    total_evals += len(other_answers)
                    fut = oracle_pool.submit(
                        _eval_batch_by_rubric, q, other_answers, crit_text
                    )
                    batch_futs.append((j, k, other_keys, fut))

            with progress_lock:
                progress["eval_total"] += total_evals
            eval_futures[q_idx] = batch_futs

        for q_idx, (q, items) in enumerate(q_items):
            n = len(items)
            cfut = coordinator_pool.submit(_coordinate_question, q_idx, q, items, n)
            coordinator_futures.append(cfut)

        # ── Phase 3: Collect per-criterion results, compute rewards ────
        for cfut in coordinator_futures:
            cfut.result()

        all_rollout_details: Dict[int, Dict[int, Dict]] = {}  # q_idx -> {j -> detail}

        for q_idx, (q, items) in enumerate(q_items):
            indices = [it[0] for it in items]  # global indices
            n = len(items)

            if n < 2:
                # Single rollout: quality adjustment only (no cross-evaluation)
                qa = float(quality_adjustments[indices[0]]) if RUBRIC_QUALITY_CONFIG.enabled else 0.0
                rewards[indices[0]] = qa
                all_rollout_details[q_idx] = {}
            else:
                # Build per-criterion score matrix:
                # crit_scores[j][k] = {answer_idx: score}
                crit_scores: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(dict)

                for j, k, other_keys, fut in eval_futures[q_idx]:
                    try:
                        scores = fut.result(timeout=900)
                        score_map = {}
                        for pos, ai in enumerate(other_keys):
                            if pos < len(scores):
                                score_map[ai] = scores[pos]
                        crit_scores[j][k] = score_map
                    except Exception as e:
                        print(f"[MetaReward] Eval error Q{q_idx} R{j} C{k}: {e}")
                        crit_scores[j][k] = {}

                # Per-rollout: Spearman dedup → consensus + discrimination
                # Store per-criterion detail for logging
                rollout_details: Dict[int, Dict] = {}

                for j in range(n):
                    j_crit = crit_scores.get(j, {})
                    n_total_crit = len(j_crit)

                    # Find unique (non-redundant) criteria within this rollout
                    unique_reps, clusters = _find_unique_criteria(j_crit, SPEARMAN_THRESHOLD)
                    n_unique = len(unique_reps)

                    # ── Per-criterion duplication weights ─────────────
                    # For token-level reward scaling: tokens of a criterion
                    # in a cluster of size c get reward / c.
                    crit_cluster_sizes: Dict[int, int] = {}
                    for _rep, _members in clusters.items():
                        for _m in _members:
                            crit_cluster_sizes[_m] = len(_members)
                    self._dup_criterion_weights[indices[j]] = [
                        1.0 / crit_cluster_sizes.get(k, 1)
                        for k in range(n_total_crit)
                    ]

                    # ── Consensus: when rollout j evaluates answer i, the
                    # score = mean of unique criteria scores for answer i.
                    # Consensus reward = mean across all other answers.
                    answer_keys_for_j = set()
                    for k_data in j_crit.values():
                        answer_keys_for_j.update(k_data.keys())
                    other_answer_keys = sorted(answer_keys_for_j)

                    consensus_per_answer = []
                    for ai in other_answer_keys:
                        crit_vals = []
                        for uk in unique_reps:
                            s = j_crit.get(uk, {}).get(ai)
                            if s is not None:
                                crit_vals.append(s)
                        if crit_vals:
                            consensus_per_answer.append(np.mean(crit_vals))
                    consensus = float(np.mean(consensus_per_answer)) if consensus_per_answer else 0.0

                    # ── Discrimination: for each unique criterion,
                    # disc_k = sqrt(var(scores_k across answers)).
                    # Rollout disc = mean(disc_k) * DISC_SCALE.
                    disc_values = []
                    for uk in unique_reps:
                        score_list = list(j_crit.get(uk, {}).values())
                        if len(score_list) >= 2:
                            disc_values.append(np.sqrt(np.var(score_list)))
                    disc_reward = float(DISC_SCALE * np.mean(disc_values)) if disc_values else 0.0

                    # ── Per-criterion detail (for logging) ────────────────
                    crit_detail = []
                    crit_texts = rollout_criterion_texts.get(q_idx, {}).get(j, [])
                    for k_idx in range(n_total_crit):
                        scores_for_k = j_crit.get(k_idx, {})
                        mean_score = float(np.mean(list(scores_for_k.values()))) if scores_for_k else 0.0
                        is_dup = k_idx not in unique_reps
                        dup_of = None
                        if is_dup:
                            for rep, members in clusters.items():
                                if k_idx in members and rep != k_idx:
                                    dup_of = rep
                                    break
                        crit_detail.append({
                            "idx": k_idx,
                            "text": crit_texts[k_idx] if k_idx < len(crit_texts) else f"criterion_{k_idx}",
                            "mean_score": mean_score,
                            "is_duplicate": is_dup,
                            "duplicate_of": dup_of,
                        })

                    # ── Semantic diversity penalty ─────────────────────────
                    # Penalize rubrics where n_unique/n_total is below target.
                    # This discourages mirrored [+]/[-] pairs that inflate
                    # criteria count without adding evaluation dimensions.
                    if n_total_crit > 0:
                        unique_ratio = n_unique / n_total_crit
                        if unique_ratio < DIVERSITY_TARGET:
                            diversity_penalty = -LAMBDA_DIVERSITY * (1.0 - unique_ratio / DIVERSITY_TARGET)
                        else:
                            diversity_penalty = 0.0
                    else:
                        diversity_penalty = 0.0

                    # ── Final reward for rollout j ────────────────────────
                    qa = float(quality_adjustments[indices[j]]) if RUBRIC_QUALITY_CONFIG.enabled else 0.0
                    rewards[indices[j]] = consensus + disc_reward + qa + diversity_penalty

                    rollout_details[j] = {
                        "n_total": n_total_crit,
                        "n_unique": n_unique,
                        "consensus": consensus,
                        "disc": disc_reward,
                        "qa": qa,
                        "diversity_penalty": diversity_penalty,
                        "reward": rewards[indices[j]].item(),
                        "criteria": crit_detail,
                        "clusters": {str(rep): members for rep, members in clusters.items()},
                    }

                    print(f"[MetaReward]   Q{q_idx} R{j}: {n_total_crit} criteria → "
                          f"{n_unique} unique, consensus={consensus:.2f}, "
                          f"disc={disc_reward:.2f}, qa={qa:.2f}, "
                          f"div={diversity_penalty:.2f}, "
                          f"reward={rewards[indices[j]].item():.2f}")

                all_rollout_details[q_idx] = rollout_details

            with progress_lock:
                progress["q_done"] += 1

            print(f"[MetaReward]   Q{q_idx+1}/{len(q_items)} done")

        # ── Cleanup ────────────────────────────────────────────────────
        solver_pool.shutdown(wait=False)
        oracle_pool.shutdown(wait=False)
        coordinator_pool.shutdown(wait=False)

        # ── Log sample rubrics ─────────────────────────────────────────
        self._log_sample_rubrics(q_items, rewards, all_rollout_details)

        # ── Rollout logging ────────────────────────────────────────────
        log_step = global_step if global_step is not None else self._step_counter
        if global_step is None:
            self._step_counter += 1
        self._save_rollout_log(
            step=log_step,
            q_items=q_items,
            question_answers=question_answers,
            rewards=rewards,
            all_rollout_details=all_rollout_details,
        )

        print(f"[MetaReward] All rewards computed, mean={rewards.mean():.3f}")
        return rewards

    def _log_sample_rubrics(
        self,
        q_items: List[Tuple[str, List[Tuple[int, str]]]],
        rewards: torch.Tensor,
        all_rollout_details: Dict[int, Dict[int, Dict]],
    ):
        """Print detailed (question, rubrics) with per-criterion scores for monitoring."""
        n_to_show = min(1, len(q_items))

        # Compute reward stats for normalization context
        all_rewards = [rewards[it[0]].item() for _, items in q_items for it in items]
        r_mean = float(np.mean(all_rewards)) if all_rewards else 0.0
        r_std = float(np.std(all_rewards)) if len(all_rewards) > 1 else 1.0

        print(f"\n{'='*80}")
        print(f"[RubricSample] Step {self._step_counter} — "
              f"{len(q_items)} questions, showing {n_to_show} full example(s)")
        print(f"[RubricSample] Reward stats: mean={r_mean:.3f}, std={r_std:.3f}")
        print(f"{'='*80}")

        for q_idx in range(n_to_show):
            q, items = q_items[q_idx]
            details = all_rollout_details.get(q_idx, {})
            print(f"\n[Q{q_idx+1}] {q}")
            print(f"  ({len(items)} rollouts)")

            for local_i, (global_idx, rubric) in enumerate(items):
                r = rewards[global_idx].item()
                norm_r = (r - r_mean) / r_std if r_std > 1e-8 else 0.0
                detail = details.get(local_i, {})

                print(f"\n  {'─'*70}")
                print(f"  Rubric {local_i+1}  reward={r:.3f}  "
                      f"normalized={norm_r:+.3f}  "
                      f"(consensus={detail.get('consensus', 0):.2f} + "
                      f"disc={detail.get('disc', 0):.2f} + "
                      f"qa={detail.get('qa', 0):.2f})")
                print(f"  Criteria: {detail.get('n_total', '?')} total → "
                      f"{detail.get('n_unique', '?')} unique")

                # Full rubric text
                for line in rubric.split('\n'):
                    print(f"    {line}")

                # Per-criterion scores
                crit_list = detail.get("criteria", [])
                if crit_list:
                    print(f"  Per-criterion scores:")
                    for c in crit_list:
                        dup_tag = ""
                        if c["is_duplicate"]:
                            dup_tag = f"  [DUP of C{c['duplicate_of']}]"
                        print(f"    C{c['idx']}: mean={c['mean_score']:.2f}{dup_tag}  {c['text']}")

        if len(q_items) > n_to_show:
            print(f"\n[RubricSample] Other questions summary:")
            for q_idx in range(n_to_show, len(q_items)):
                q, items = q_items[q_idx]
                rews = [rewards[it[0]].item() for it in items]
                details = all_rollout_details.get(q_idx, {})
                n_unique_list = [details.get(j, {}).get("n_unique", "?") for j in range(len(items))]
                print(f"  Q{q_idx+1}: \"{q[:100]}\" "
                      f"| {len(items)} rollouts | rewards={[f'{r:.2f}' for r in rews]} "
                      f"| unique_criteria={n_unique_list}")

        print(f"{'='*80}\n")

    @staticmethod
    def _json_default(obj):
        """Handle numpy/torch types for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Catch-all for any numeric type with .item() (torch scalars, etc.)
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def _save_rollout_log(
        self,
        step: int,
        q_items: List[Tuple[str, List[Tuple[int, str]]]],
        question_answers: Dict[int, Dict[int, str]],
        rewards: torch.Tensor,
        all_rollout_details: Dict[int, Dict[int, Dict]],
    ):
        """Save per-step rollout log: question, rubrics, answers, rewards, criteria details."""
        log_dir = Path(ROLLOUT_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"step_{step:04d}.jsonl"

        try:
            with open(log_file, "w", encoding="utf-8") as f:
                for q_idx, (q, items) in enumerate(q_items):
                    answers_dict = question_answers.get(q_idx, {})
                    details = all_rollout_details.get(q_idx, {})
                    rollouts = []
                    for local_i, (global_idx, rubric) in enumerate(items):
                        detail = details.get(local_i, {})
                        rollouts.append({
                            "rubric": rubric,
                            "answer": answers_dict.get(local_i, ""),
                            "reward": round(rewards[global_idx].item(), 4),
                            "consensus": round(detail.get("consensus", 0), 4),
                            "disc": round(detail.get("disc", 0), 4),
                            "qa": round(detail.get("qa", 0), 4),
                            "n_criteria_total": detail.get("n_total", 0),
                            "n_criteria_unique": detail.get("n_unique", 0),
                            "criteria": detail.get("criteria", []),
                            "clusters": detail.get("clusters", {}),
                        })
                    record = {
                        "question": q,
                        "n_rollouts": len(items),
                        "rollouts": rollouts,
                    }
                    f.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")
            print(f"[MetaReward] Rollout log saved: {log_file}")
        except Exception as e:
            print(f"[MetaReward] Failed to save rollout log: {e}")

    def _generate_answer(self, question: str, rubric: str) -> str:
        return self.solver.generate_answer(question, rubric)
