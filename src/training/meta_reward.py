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

# Regex to extract **bolded** concepts from rubric text
_BOLD_CONCEPT_RE = re.compile(r'\*\*([^*]+)\*\*')

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
# Thresholded power-variance discrimination: criteria with std < DISC_STD_FLOOR
# are considered non-discriminative and get disc_k = 0 (hard zero floor).
# For discriminative criteria: disc_k = (std_k - DISC_STD_FLOOR) ^ DISC_POWER.
# This creates strong separation: generic criteria → 0, specific criteria → high.
# Oracle scores are [0,10], so std range is [0, ~5].
DISC_SCALE = float(os.environ.get("GRM_DISC_SCALE", "1.0"))
DISC_STD_FLOOR = float(os.environ.get("GRM_DISC_STD_FLOOR", "0.5"))  # std below this → disc=0
DISC_POWER = float(os.environ.get("GRM_DISC_POWER", "2.0"))  # power > 1 amplifies separation
# ── Spearman redundancy parameters ────────────────────────────────────────
# Threshold for Spearman correlation above which two criteria within the same
# rollout are considered redundant.  Redundant criteria are counted only once
# when computing the consensus score.
SPEARMAN_THRESHOLD = float(os.environ.get("GRM_SPEARMAN_THRESHOLD", "0.8"))

# ── Anti-collapse: criterion length floor ────────────────────────────────
# Criteria shorter than this (chars) get a per-criterion penalty added to QA.
CRITERION_MIN_CHARS = int(os.environ.get("GRM_CRITERION_MIN_CHARS", "200"))
CRITERION_SHORT_PENALTY = float(os.environ.get("GRM_CRITERION_SHORT_PENALTY", "-0.3"))

# ── Anti-collapse: response length soft floor ────────────────────────────
# Rollouts shorter than this (tokens) get a linear penalty added to QA.
RESPONSE_MIN_TOKENS = int(os.environ.get("GRM_RESPONSE_MIN_TOKENS", "400"))
RESPONSE_SHORT_PENALTY = float(os.environ.get("GRM_RESPONSE_SHORT_PENALTY", "-3.0"))

# ── Gold-disc: reward rubrics that score gold rubric's answer higher ─────
# Enable/disable gold-disc reward component.
GOLD_DISC_ENABLED = os.environ.get("GRM_GOLD_DISC_ENABLED", "false").lower() == "true"
# Scale factor for gold_disc component (applied before GDPO normalization).
GOLD_DISC_SCALE = float(os.environ.get("GRM_GOLD_DISC_SCALE", "1.0"))

# ── Calibration: penalise rubrics that produce binary (0/10) scores ──────
# Binary ratio above this floor triggers a linear penalty.
BINARY_RATIO_FLOOR = float(os.environ.get("GRM_BINARY_RATIO_FLOOR", "0.3"))
BINARY_PENALTY_SCALE = float(os.environ.get("GRM_BINARY_PENALTY_SCALE", "3.0"))

# ── Anti-parroting: penalise criteria that quote phrases from the question ──
# Measures what fraction of each criterion's content words come from the question.
# A criterion that merely restates question phrases adds no evaluation value.
PARROT_PENALTY_SCALE = float(os.environ.get("GRM_PARROT_PENALTY_SCALE", "10.0"))
PARROT_RATIO_FLOOR = float(os.environ.get("GRM_PARROT_RATIO_FLOOR", "0.25"))

_PARROT_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "or", "but", "if", "not", "no", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "about", "between", "after", "before",
    "that", "this", "these", "those", "it", "its", "my", "your", "his",
    "her", "our", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "any", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just",
    "answer", "must", "should", "criterion", "criteria", "response",
    "explicitly", "specifically", "directly", "correctly", "clearly",
    "mention", "include", "identify", "reference", "state", "provide",
    "ensure", "address", "explain", "describe", "discuss", "note",
})


def _content_words(text: str) -> set:
    """Extract lowercased content words (len >= 3, not in stop list)."""
    return {w for w in re.findall(r'[a-z0-9]+', text.lower())
            if w not in _PARROT_STOP_WORDS and len(w) >= 3}


def _compute_parrot_penalty(question: str, criteria_texts: List[str]) -> float:
    """Compute penalty for rubric criteria that parrot the question.

    For each criterion, measures what fraction of its content words also appear
    in the question.  Criteria dominated by question vocabulary are parroting.

    Returns a non-positive float (penalty).
    """
    # Strip common prefixes
    q = re.sub(r'^(User|Human|Assistant):\s*', '', question, flags=re.IGNORECASE)
    q_words = _content_words(q)
    if len(q_words) < 3:
        return 0.0

    crit_parrot_scores = []
    for crit in criteria_texts:
        c_words = _content_words(crit)
        if len(c_words) < 3:
            crit_parrot_scores.append(0.0)
            continue
        # What fraction of criterion words come from the question?
        overlap = len(q_words & c_words) / len(c_words)
        crit_parrot_scores.append(overlap)

    if not crit_parrot_scores:
        return 0.0
    mean_parrot = float(np.mean(crit_parrot_scores))
    excess = mean_parrot - PARROT_RATIO_FLOOR
    if excess > 0:
        return -PARROT_PENALTY_SCALE * excess
    return 0.0


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


def _extract_concepts(rubric_text: str) -> Set[str]:
    """Extract bolded concepts from rubric text, lowercased."""
    return {m.lower().strip() for m in _BOLD_CONCEPT_RE.findall(rubric_text)}


def _compute_group_diversity(rubric_texts: List[str]) -> List[float]:
    """Compute per-rollout concept diversity within a GRPO group.

    For each rollout i, diversity_i = 1 - mean_concept_jaccard(i, j) for j != i.
    Returns a list of diversity scores in [0, 1].
    Higher = rollout covers more unique concepts relative to siblings.
    """
    n = len(rubric_texts)
    if n < 2:
        return [0.0] * n

    concept_sets = [_extract_concepts(r) for r in rubric_texts]

    diversity_scores = []
    for i in range(n):
        if not concept_sets[i]:
            diversity_scores.append(0.0)
            continue
        pairwise_jaccards = []
        for j in range(n):
            if j == i:
                continue
            if not concept_sets[j]:
                pairwise_jaccards.append(0.0)
                continue
            inter = len(concept_sets[i] & concept_sets[j])
            union = len(concept_sets[i] | concept_sets[j])
            pairwise_jaccards.append(inter / union if union > 0 else 0.0)
        diversity_scores.append(1.0 - float(np.mean(pairwise_jaccards)))
    return diversity_scores


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
                       gold_rubrics: Optional[List[str]] = None,
                       gold_answers: Optional[List[str]] = None,
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
               std_k < DISC_STD_FLOOR → disc_k = 0 (non-discriminative)
               else → disc_k = (std_k - DISC_STD_FLOOR) ^ DISC_POWER
               disc_j = mean(disc_k) * DISC_SCALE (zeros included in mean)
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
              f"disc_scale={DISC_SCALE}, disc_std_floor={DISC_STD_FLOOR}, "
              f"disc_power={DISC_POWER}, spearman_threshold={SPEARMAN_THRESHOLD}, "
              f"rubric_quality={RUBRIC_QUALITY_CONFIG.enabled}, "
              f"gold_disc={GOLD_DISC_ENABLED}, gold_disc_scale={GOLD_DISC_SCALE}, "
              f"binary_penalty_scale={BINARY_PENALTY_SCALE}, binary_ratio_floor={BINARY_RATIO_FLOOR}")

        # ── Group by question ──────────────────────────────────────────
        q_to_rubrics: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        q_to_gold_rubric: Dict[str, str] = {}  # question -> gold rubric text
        q_to_gold_answer: Dict[str, str] = {}  # question -> ideal_completion (gold answer)
        for idx, (q, r) in enumerate(zip(questions, rubrics)):
            # Strip <think>...</think> blocks (instruct/thinking models)
            r_clean = _THINK_BLOCK_RE.sub('', r)
            # Strip legacy "| tags: ..." suffixes from generated rubric text
            r_clean = _TAGS_SUFFIX_RE.sub('', r_clean)
            q_to_rubrics[q].append((idx, r_clean))
            # Associate gold rubric with question (same for all rollouts of the question)
            if GOLD_DISC_ENABLED and gold_rubrics and idx < len(gold_rubrics):
                gr = gold_rubrics[idx].strip()
                if gr and q not in q_to_gold_rubric:
                    q_to_gold_rubric[q] = gr
            # Associate gold answer with question
            if gold_answers and idx < len(gold_answers):
                ga = gold_answers[idx].strip()
                if ga and q not in q_to_gold_answer:
                    q_to_gold_answer[q] = ga

        q_items = list(q_to_rubrics.items())
        n_with_gold = sum(1 for q, _ in q_items if q in q_to_gold_rubric)
        n_with_gold_ans = sum(1 for q, _ in q_items if q in q_to_gold_answer)
        print(f"[MetaReward] Processing {len(q_items)} unique questions "
              f"({n_with_gold} with gold rubric, {n_with_gold_ans} with gold answer)")

        rewards = torch.zeros(len(questions), dtype=torch.float32)
        self._dup_criterion_weights = {}  # reset per batch
        # GDPO: per-component rewards for decoupled normalization
        self._component_rewards: Dict[int, Tuple[float, float, float, float, float]] = {}  # idx -> (consensus, disc, qa, gold_disc, calibration)

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
        # Gold rubric solver futures: q_idx -> Future (one per question)
        gold_solver_futures: Dict[int, Future] = {}

        for q_idx, (q, items) in enumerate(q_items):
            n = len(items)
            if solver_n > 0 and n > solver_n:
                selected = sorted(random.sample(range(n), solver_n))
            else:
                selected = list(range(n))
            solver_answer_indices[q_idx] = selected

            n_solver_tasks = len(selected)
            # Submit gold rubric solver task only if no pre-computed gold answer
            gold_rubric_text = q_to_gold_rubric.get(q)
            precomputed_gold = q_to_gold_answer.get(q)
            if gold_rubric_text and not precomputed_gold:
                gold_solver_futures[q_idx] = solver_pool.submit(_gen_answer, q, gold_rubric_text)
                n_solver_tasks += 1

            with progress_lock:
                progress["solver_total"] += n_solver_tasks

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

        # Gold answer storage: q_idx -> answer text (from gold rubric solver)
        gold_answers: Dict[int, str] = {}
        # Gold answer index within the answer pool: q_idx -> int
        gold_answer_indices: Dict[int, int] = {}

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

            # Collect gold answer: prefer precomputed ideal_completion, fall back to gold solver
            gold_ans = None
            precomputed_gold = q_to_gold_answer.get(q)
            if precomputed_gold:
                gold_ans = precomputed_gold
                gold_answers[q_idx] = gold_ans
            elif q_idx in gold_solver_futures:
                try:
                    gold_ans = gold_solver_futures[q_idx].result(timeout=900)
                    gold_answers[q_idx] = gold_ans
                except Exception as e:
                    print(f"[MetaReward] Gold solver error Q{q_idx}: {e}")

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

            # Add gold answer to the pool with a special index (n = beyond rollout indices)
            gold_idx = None
            if gold_ans is not None:
                gold_idx = n  # index beyond the rollout indices 0..n-1
                answers[gold_idx] = gold_ans
                gold_answer_indices[q_idx] = gold_idx

            all_answer_keys = sorted(answers.keys())

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
                    criterion_texts = []
                rollout_criterion_texts[q_idx][j] = criterion_texts

                # Other answers (exclude rollout j's own answer, include gold)
                other_keys = [ai for ai in all_answer_keys if ai != j]
                if not other_keys:
                    other_keys = all_answer_keys[:]
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
                self._component_rewards[indices[0]] = (0.0, 0.0, qa, 0.0, 0.0, 0.0)
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
                    # Anti-collapse: only include discriminative criteria
                    # (std >= DISC_STD_FLOOR) in consensus. This prevents
                    # tautological criteria from inflating consensus scores.
                    answer_keys_for_j = set()
                    for k_data in j_crit.values():
                        answer_keys_for_j.update(k_data.keys())
                    other_answer_keys = sorted(answer_keys_for_j)
                    # Exclude gold answer from consensus/disc (used only for gold_disc)
                    gold_idx = gold_answer_indices.get(q_idx)
                    solver_answer_keys = [ai for ai in other_answer_keys if ai != gold_idx]

                    # Pre-compute which unique criteria are discriminative
                    discriminative_reps = set()
                    for uk in unique_reps:
                        # Use only solver answers for std computation (exclude gold)
                        score_list = [j_crit.get(uk, {}).get(ai) for ai in solver_answer_keys]
                        score_list = [s for s in score_list if s is not None]
                        if len(score_list) >= 2:
                            std_k = float(np.std(score_list))
                            if std_k >= DISC_STD_FLOOR:
                                discriminative_reps.add(uk)
                    # Fall back to all criteria if none are discriminative
                    consensus_reps = discriminative_reps if discriminative_reps else set(unique_reps)

                    consensus_per_answer = []
                    for ai in solver_answer_keys:
                        crit_vals = []
                        for uk in consensus_reps:
                            s = j_crit.get(uk, {}).get(ai)
                            if s is not None:
                                crit_vals.append(s)
                        if crit_vals:
                            consensus_per_answer.append(np.mean(crit_vals))
                    consensus = float(np.mean(consensus_per_answer)) if consensus_per_answer else 0.0

                    # ── Discrimination: thresholded power-variance.
                    # For each unique criterion k:
                    #   std_k < DISC_STD_FLOOR → disc_k = 0 (non-discriminative)
                    #   else → disc_k = (std_k - DISC_STD_FLOOR) ^ DISC_POWER
                    # Rollout disc = mean(disc_k) * DISC_SCALE (zeros included).
                    disc_values = []
                    n_discriminative = 0
                    for uk in unique_reps:
                        # Use only solver answers for disc (exclude gold)
                        score_list = [j_crit.get(uk, {}).get(ai) for ai in solver_answer_keys]
                        score_list = [s for s in score_list if s is not None]
                        if len(score_list) >= 2:
                            std_k = float(np.std(score_list))
                            if std_k < DISC_STD_FLOOR:
                                disc_values.append(0.0)
                            else:
                                disc_values.append((std_k - DISC_STD_FLOOR) ** DISC_POWER)
                                n_discriminative += 1
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

                    # ── Final reward for rollout j ────────────────────────
                    qa = float(quality_adjustments[indices[j]]) if RUBRIC_QUALITY_CONFIG.enabled else 0.0

                    # Anti-collapse: criterion length floor penalty
                    n_short_crit = 0
                    for ct in crit_texts:
                        if len(ct) < CRITERION_MIN_CHARS:
                            n_short_crit += 1
                    crit_len_penalty = n_short_crit * CRITERION_SHORT_PENALTY

                    # Anti-collapse: response length soft floor penalty
                    resp_len_penalty = 0.0
                    tc_j = response_token_counts[indices[j]] if response_token_counts else 0
                    if tc_j > 0 and tc_j < RESPONSE_MIN_TOKENS:
                        # Linear ramp: full penalty at 0 tokens, zero at RESPONSE_MIN_TOKENS
                        resp_len_penalty = RESPONSE_SHORT_PENALTY * (1.0 - tc_j / RESPONSE_MIN_TOKENS)

                    qa += crit_len_penalty + resp_len_penalty

                    # ── Gold-disc: how well rubric_j scores gold answer vs solver answers ──
                    gold_disc = 0.0
                    gold_idx = gold_answer_indices.get(q_idx)
                    if gold_idx is not None and GOLD_DISC_ENABLED:
                        gold_disc_per_crit = []
                        for uk in unique_reps:
                            gold_s = j_crit.get(uk, {}).get(gold_idx)
                            solver_scores = [
                                j_crit.get(uk, {}).get(ai)
                                for ai in solver_answer_keys
                            ]
                            solver_scores = [s for s in solver_scores if s is not None]
                            if gold_s is not None and solver_scores:
                                gold_disc_per_crit.append(gold_s - float(np.mean(solver_scores)))
                        if gold_disc_per_crit:
                            gold_disc = float(np.mean(gold_disc_per_crit)) * GOLD_DISC_SCALE

                    # ── Calibration penalty: penalise binary (0/10) score distributions ──
                    calibration_penalty = 0.0
                    all_eval_scores = []
                    for uk in unique_reps:
                        all_eval_scores.extend(j_crit.get(uk, {}).values())
                    if all_eval_scores:
                        binary_count = sum(1 for s in all_eval_scores if s <= 0.5 or s >= 9.5)
                        binary_ratio = binary_count / len(all_eval_scores)
                        excess = binary_ratio - BINARY_RATIO_FLOOR
                        if excess > 0:
                            calibration_penalty = -BINARY_PENALTY_SCALE * excess

                    # ── Anti-parroting: penalise criteria that quote the question ──
                    parrot_penalty = _compute_parrot_penalty(q, crit_texts)

                    rewards[indices[j]] = consensus + disc_reward + qa + gold_disc + calibration_penalty + parrot_penalty
                    # GDPO: store per-component rewards for decoupled normalization
                    self._component_rewards[indices[j]] = (consensus, disc_reward, qa, gold_disc, calibration_penalty, parrot_penalty)

                    rollout_details[j] = {
                        "n_total": n_total_crit,
                        "n_unique": n_unique,
                        "n_discriminative": len(discriminative_reps),
                        "n_short_crit": n_short_crit,
                        "consensus": consensus,
                        "disc": disc_reward,
                        "qa": qa,
                        "gold_disc": gold_disc,
                        "calibration": calibration_penalty,
                        "parrot": parrot_penalty,
                        "binary_ratio": binary_ratio if all_eval_scores else 0.0,
                        "crit_len_penalty": crit_len_penalty,
                        "resp_len_penalty": resp_len_penalty,
                        "reward": rewards[indices[j]].item(),
                        "criteria": crit_detail,
                        "clusters": {str(rep): members for rep, members in clusters.items()},
                    }

                all_rollout_details[q_idx] = rollout_details

                # ── GDPO-annotated per-rollout summary for this question ──
                w = self.GDPO_WEIGHTS
                g_cons = [rollout_details[j].get("consensus", 0.0) for j in range(n)]
                g_disc = [rollout_details[j].get("disc", 0.0) for j in range(n)]
                g_qa = [rollout_details[j].get("qa", 0.0) for j in range(n)]
                g_gold = [rollout_details[j].get("gold_disc", 0.0) for j in range(n)]
                g_cal = [rollout_details[j].get("calibration", 0.0) for j in range(n)]
                g_parrot = [rollout_details[j].get("parrot", 0.0) for j in range(n)]
                n_cons = self._gdpo_normalize_group(g_cons, weight=w["consensus"])
                n_disc = self._gdpo_normalize_group(g_disc, weight=w["disc"])
                n_qa = self._gdpo_normalize_group(g_qa, weight=w["qa"])
                n_gold = self._gdpo_normalize_group(g_gold, weight=w["gold_disc"])
                n_cal = self._gdpo_normalize_group(g_cal, weight=w["calibration"])
                n_parrot = self._gdpo_normalize_group(g_parrot, weight=w["parrot"])
                gold_rubric_avail = q_to_gold_rubric.get(q, "") != ""
                for j in range(n):
                    rd = rollout_details[j]
                    gdpo_adv = n_cons[j] + n_disc[j] + n_qa[j] + n_gold[j] + n_cal[j] + n_parrot[j]
                    gold_str = f"gold_disc={rd.get('gold_disc', 0):.2f}(×{w['gold_disc']:.1f}→{n_gold[j]:+.3f})  " if gold_rubric_avail else ""
                    cal_str = f"cal={rd.get('calibration', 0):.2f}(bin={rd.get('binary_ratio', 0):.0%})(×{w['calibration']:.1f}→{n_cal[j]:+.3f})"
                    parrot_str = f"  parrot={rd.get('parrot', 0):.2f}(×{w['parrot']:.1f}→{n_parrot[j]:+.3f})" if rd.get('parrot', 0) != 0 else ""
                    print(f"[MetaReward]   Q{q_idx} R{j}: {rd['n_total']} criteria → "
                          f"{rd['n_unique']} unique, reward={rd['reward']:.2f}  "
                          f"gdpo={gdpo_adv:+.3f}  "
                          f"cons={rd['consensus']:.2f}(×{w['consensus']:.1f}→{n_cons[j]:+.3f})  "
                          f"disc={rd['disc']:.2f}(×{w['disc']:.1f}→{n_disc[j]:+.3f})  "
                          f"qa={rd['qa']:.2f}(×{w['qa']:.1f}→{n_qa[j]:+.3f})  "
                          f"{gold_str}{cal_str}{parrot_str}")

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

    # GDPO per-component weights (must match core_algos.GDPO_COMPONENT_WEIGHTS)
    GDPO_WEIGHTS = {"consensus": 1.0, "disc": 1.0, "qa": 0.5, "gold_disc": 2.0, "calibration": 0.5, "parrot": 3.0}

    @staticmethod
    def _gdpo_normalize_group(values: List[float], weight: float = 1.0, epsilon: float = 1e-6) -> List[float]:
        """Replicate GDPO within-group normalization for display (with weight)."""
        if len(values) <= 1:
            return [0.0] * len(values)
        arr = np.array(values, dtype=np.float64)
        m, s = float(arr.mean()), float(arr.std())
        return [float(weight * (v - m) / (s + epsilon)) for v in values]

    def _log_sample_rubrics(
        self,
        q_items: List[Tuple[str, List[Tuple[int, str]]]],
        rewards: torch.Tensor,
        all_rollout_details: Dict[int, Dict[int, Dict]],
    ):
        """Print detailed (question, rubrics) with per-criterion scores for monitoring."""
        n_to_show = min(1, len(q_items))

        # Compute reward stats for context
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

            # Compute GDPO per-component within-group normalization
            group_cons = [details.get(j, {}).get("consensus", 0.0) for j in range(len(items))]
            group_disc = [details.get(j, {}).get("disc", 0.0) for j in range(len(items))]
            group_qa = [details.get(j, {}).get("qa", 0.0) for j in range(len(items))]
            group_gold = [details.get(j, {}).get("gold_disc", 0.0) for j in range(len(items))]
            group_cal = [details.get(j, {}).get("calibration", 0.0) for j in range(len(items))]
            group_parrot = [details.get(j, {}).get("parrot", 0.0) for j in range(len(items))]
            w = self.GDPO_WEIGHTS
            norm_cons = self._gdpo_normalize_group(group_cons, weight=w["consensus"])
            norm_disc = self._gdpo_normalize_group(group_disc, weight=w["disc"])
            norm_qa = self._gdpo_normalize_group(group_qa, weight=w["qa"])
            norm_gold = self._gdpo_normalize_group(group_gold, weight=w["gold_disc"])
            norm_cal = self._gdpo_normalize_group(group_cal, weight=w["calibration"])
            norm_parrot = self._gdpo_normalize_group(group_parrot, weight=w["parrot"])

            print(f"\n[Q{q_idx+1}] {q}")
            print(f"  ({len(items)} rollouts)  GDPO weights: cons={w['consensus']:.1f}, disc={w['disc']:.1f}, qa={w['qa']:.1f}, gold={w['gold_disc']:.1f}, cal={w['calibration']:.1f}, parrot={w['parrot']:.1f}")
            print(f"  Group stats: consensus μ={np.mean(group_cons):.2f} σ={np.std(group_cons):.2f}, "
                  f"disc μ={np.mean(group_disc):.2f} σ={np.std(group_disc):.2f}, "
                  f"qa μ={np.mean(group_qa):.2f} σ={np.std(group_qa):.2f}, "
                  f"gold_disc μ={np.mean(group_gold):.2f} σ={np.std(group_gold):.2f}, "
                  f"cal μ={np.mean(group_cal):.2f} σ={np.std(group_cal):.2f}, "
                  f"parrot μ={np.mean(group_parrot):.2f} σ={np.std(group_parrot):.2f}")

            for local_i, (global_idx, rubric) in enumerate(items):
                r = rewards[global_idx].item()
                detail = details.get(local_i, {})
                cons_raw = detail.get('consensus', 0.0)
                disc_raw = detail.get('disc', 0.0)
                qa_raw = detail.get('qa', 0.0)
                gold_raw = detail.get('gold_disc', 0.0)
                cal_raw = detail.get('calibration', 0.0)
                parrot_raw = detail.get('parrot', 0.0)
                gdpo_adv = norm_cons[local_i] + norm_disc[local_i] + norm_qa[local_i] + norm_gold[local_i] + norm_cal[local_i] + norm_parrot[local_i]

                print(f"\n  {'─'*70}")
                print(f"  Rubric {local_i+1}  reward={r:.3f}  "
                      f"gdpo_adv={gdpo_adv:+.3f}")
                print(f"    cons={cons_raw:.2f}(×{w['consensus']:.1f}→{norm_cons[local_i]:+.3f})  |  "
                      f"disc={disc_raw:.2f}(×{w['disc']:.1f}→{norm_disc[local_i]:+.3f})  |  "
                      f"qa={qa_raw:.2f}(×{w['qa']:.1f}→{norm_qa[local_i]:+.3f})")
                parrot_str = f"  |  parrot={parrot_raw:.2f}(×{w['parrot']:.1f}→{norm_parrot[local_i]:+.3f})" if parrot_raw != 0 else ""
                print(f"    gold_disc={gold_raw:.2f}(×{w['gold_disc']:.1f}→{norm_gold[local_i]:+.3f})  |  "
                      f"cal={cal_raw:.2f}(bin={detail.get('binary_ratio', 0):.0%})(×{w['calibration']:.1f}→{norm_cal[local_i]:+.3f}){parrot_str}")
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
                details = all_rollout_details.get(q_idx, {})
                n = len(items)
                # Compute GDPO summary
                g_cons = [details.get(j, {}).get("consensus", 0.0) for j in range(n)]
                g_disc = [details.get(j, {}).get("disc", 0.0) for j in range(n)]
                g_qa = [details.get(j, {}).get("qa", 0.0) for j in range(n)]
                g_gold = [details.get(j, {}).get("gold_disc", 0.0) for j in range(n)]
                g_cal = [details.get(j, {}).get("calibration", 0.0) for j in range(n)]
                g_parrot = [details.get(j, {}).get("parrot", 0.0) for j in range(n)]
                n_cons = self._gdpo_normalize_group(g_cons, weight=self.GDPO_WEIGHTS["consensus"])
                n_disc = self._gdpo_normalize_group(g_disc, weight=self.GDPO_WEIGHTS["disc"])
                n_qa = self._gdpo_normalize_group(g_qa, weight=self.GDPO_WEIGHTS["qa"])
                n_gold = self._gdpo_normalize_group(g_gold, weight=self.GDPO_WEIGHTS["gold_disc"])
                n_cal = self._gdpo_normalize_group(g_cal, weight=self.GDPO_WEIGHTS["calibration"])
                n_parrot = self._gdpo_normalize_group(g_parrot, weight=self.GDPO_WEIGHTS["parrot"])
                gdpo_advs = [n_cons[j] + n_disc[j] + n_qa[j] + n_gold[j] + n_cal[j] + n_parrot[j] for j in range(n)]
                n_unique_list = [details.get(j, {}).get("n_unique", "?") for j in range(n)]
                rews = [rewards[it[0]].item() for it in items]
                gold_ds = [f'{details.get(j, {}).get("gold_disc", 0):.1f}' for j in range(n)]
                print(f"  Q{q_idx+1}: \"{q[:80]}\" | {n} rollouts "
                      f"| reward={[f'{r:.1f}' for r in rews]} "
                      f"| gdpo_adv={[f'{a:+.2f}' for a in gdpo_advs]} "
                      f"| unique={n_unique_list} "
                      f"| gold_disc={gold_ds}")

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
                            "gold_disc": round(detail.get("gold_disc", 0), 4),
                            "calibration": round(detail.get("calibration", 0), 4),
                            "binary_ratio": round(detail.get("binary_ratio", 0), 4),
                            "n_criteria_total": detail.get("n_total", 0),
                            "n_criteria_unique": detail.get("n_unique", 0),
                            "n_discriminative": detail.get("n_discriminative", 0),
                            "n_short_crit": detail.get("n_short_crit", 0),
                            "crit_len_penalty": round(detail.get("crit_len_penalty", 0), 4),
                            "resp_len_penalty": round(detail.get("resp_len_penalty", 0), 4),
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
