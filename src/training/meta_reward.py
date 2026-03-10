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
from tqdm import tqdm
from src.evaluation.judge import Oracle
from src.models.solver import Solver
from src.training.matrix_completion import als_matrix_completion
from src.training.rubric_quality import (
    RubricQualityConfig,
    score_rubric_quality,
)

# Per-service parallel settings
DEFAULT_SOLVER_WORKERS = int(os.environ.get("GRM_SOLVER_WORKERS", 4))
DEFAULT_ORACLE_WORKERS = int(os.environ.get("GRM_ORACLE_WORKERS", 4))

# K-sparse cross-evaluation: 0 means "use all N-1" (full matrix, backward compat)
K_SPARSE = int(os.environ.get("GRM_K_SPARSE", "0"))

# Maximum number of solver answers per question group (0 = same as N)
SOLVER_N = int(os.environ.get("GRM_SOLVER_N", "0"))

# Matrix completion after K-sparse observation
USE_MATRIX_COMPLETION = os.environ.get("GRM_USE_MATRIX_COMPLETION", "false").lower() in ("1", "true", "yes")
MC_RANK = int(os.environ.get("GRM_MC_RANK", "3"))
MC_MAX_ITER = int(os.environ.get("GRM_MC_MAX_ITER", "30"))
MC_REG = float(os.environ.get("GRM_MC_REG", "0.1"))

# Rubric quality scoring
RUBRIC_QUALITY_CONFIG = RubricQualityConfig()

# Marker the solver outputs when it detects a garbage rubric
GARBAGE_RUBRIC_MARKER = "<GARBAGE_RUBRIC>"

# Weight penalty for garbage rubrics:
# - Garbage rubric's own reward = normal_reward / w
# - When garbage rubric evaluates others, its scores get weight 1/w
# Increased from 3→10 to strongly discourage garbage generation.
# With weight=10, a garbage rubric scoring 8.0 consensus only gets 0.8 reward.
GARBAGE_RUBRIC_WEIGHT = float(os.environ.get("GRM_GARBAGE_RUBRIC_WEIGHT", "10"))

# Marker the solver outputs when it detects a generic (non-specific) rubric
GENERAL_RUBRIC_MARKER = "<GENERAL_RUBRIC>"

# Weight penalty for generic rubrics:
# - Generic rubric's own reward = normal_reward / w
# - Unlike garbage, generic rubrics' evaluations of others are NOT down-weighted
GENERAL_RUBRIC_WEIGHT = float(os.environ.get("GRM_GENERAL_RUBRIC_WEIGHT", "3"))

# Rollout logging directory
ROLLOUT_LOG_DIR = os.environ.get("GRM_ROLLOUT_LOG_DIR", "out/rl/rollout_logs")

# ── Generic-flag false-positive override ──────────────────────────────────
# Words that appear in almost any health-domain rubric — NOT topic-specific.
# Only words >= 5 chars matter (shorter words are excluded by the overlap check).
_GENERIC_STOP_WORDS: Set[str] = {
    # common English (5+ chars)
    "about", "after", "along", "among", "asked", "based", "being", "below",
    "between", "better", "could", "daily", "doing", "during", "every",
    "first", "function", "given", "going", "great", "might", "never",
    "other", "partial", "please", "point", "should", "since", "still",
    "their", "there", "these", "thing", "think", "those", "three", "times",
    "today", "total", "under", "until", "using", "where", "which", "while",
    "whole", "worse", "would", "years", "without",
    # generic medical / clinical terms
    "acute", "advice", "assess", "assessment", "avoid", "based", "cause",
    "causes", "changes", "check", "chronic", "clear", "clinic", "clinical",
    "common", "complete", "concern", "concerns", "condition", "conditions",
    "conflicting", "consider", "context", "correct", "critical", "current",
    "details", "diagnosis", "discuss", "domain", "emergency", "ensure",
    "evaluate", "evaluation", "evidence", "explain", "factor", "factors",
    "follow", "further", "general", "guidance", "guideline", "guidelines",
    "health", "important", "immediate", "include", "includes", "including",
    "information", "initial", "issue", "issues", "level", "levels",
    "management", "medical", "mention", "moderate", "monitor", "needs",
    "normal", "notes", "objective", "option", "options", "outcome",
    "outcomes", "overall", "patient", "patients", "physical", "potential",
    "practice", "practices", "present", "produce", "professional",
    "properly", "provide", "provides", "quality", "questions", "recommend",
    "recommendation", "recovery", "reduce", "refer", "referral", "relevant",
    "report", "response", "results", "review", "risks", "safety",
    "screening", "severe", "signs", "situation", "specific", "standard",
    "steps", "stress", "subjective", "suggest", "support", "symptoms",
    "system", "testing", "therapy", "times", "treatment", "understand",
    "visit",
    # rubric-meta / prompt terms
    "accuracy", "actionable", "address", "addresses", "alignment",
    "answer", "answers", "appropriate", "clarity", "communication",
    "comprehensive", "criteria", "criterion", "effective", "empathy",
    "model", "models", "points", "question", "rubric", "tags", "tailors",
    # short but common medical
    "care", "data", "diet", "drug", "exam", "labs", "life", "long",
    "meds", "pain", "plan", "risk", "safe", "test", "type",
}


def _has_topic_specificity(question: str, rubric: str, min_overlap: int = 2) -> list:
    """Check if rubric contains topic-specific words from the question.

    Returns the list of topic-overlap words (empty if no specificity found).
    Used to override false-positive <GENERAL_RUBRIC> flags from the solver.
    """
    # Extract content words (5+ chars) from the question
    q_words = {w.lower() for w in re.findall(r"[a-zA-Z]{5,}", question)}
    q_topic = q_words - _GENERIC_STOP_WORDS

    rubric_lower = rubric.lower()
    overlap = [w for w in q_topic if w in rubric_lower]

    return overlap if len(overlap) >= min_overlap else []


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
        # When pickling, don't include the client objects
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
        Computes the Consensus-Based Meta-Reward with pipelined execution,
        K-sparse cross-evaluation, optional matrix completion, and rubric
        quality adjustments.

        Pipeline per question (N rollouts → N rubrics):
            1. Generate N answers via Solver (parallel within solver_pool)
            2. K-sparse selection: for each rubric j, pick min(K, N-1) other
               rubrics to evaluate against (K=0 → full N-1, backward compat)
            3. Oracle scores only selected (answer, rubric) pairs → O(NK) calls
            4. Optionally apply ALS matrix completion on sparse score matrix
            5. Compute cross-consensus reward from (completed) matrix
            6. Add rubric quality adjustment (repetition/diversity/length)

        Every rollout gets its own reward for proper RL policy gradients.
        """
        if solver_workers is None:
            solver_workers = DEFAULT_SOLVER_WORKERS
        if oracle_workers is None:
            oracle_workers = DEFAULT_ORACLE_WORKERS

        k_sparse = K_SPARSE
        use_mc = USE_MATRIX_COMPLETION
        solver_n = SOLVER_N  # 0 = use all N rubrics

        # Use global_step if provided, otherwise fall back to internal counter
        if global_step is not None:
            self._step_counter = global_step

        print(f"[MetaReward] Computing rewards for {len(questions)} samples "
              f"(global_step={self._step_counter}), "
              f"solver_workers={solver_workers}, oracle_workers={oracle_workers}, "
              f"K_sparse={k_sparse or 'full'}, matrix_completion={use_mc}, "
              f"solver_n={solver_n or 'all'}, "
              f"rubric_quality={RUBRIC_QUALITY_CONFIG.enabled}")

        # ── Group by question ──────────────────────────────────────────
        q_to_rubrics: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for idx, (q, r) in enumerate(zip(questions, rubrics)):
            q_to_rubrics[q].append((idx, r))

        q_items = list(q_to_rubrics.items())
        print(f"[MetaReward] Processing {len(q_items)} unique questions (pipelined)")

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
                if result.detail.get("think_leak_penalty", 0.0) < -0.01:
                    print(f"[MetaReward]   rubric {idx}: think_leak_penalty="
                          f"{result.detail['think_leak_penalty']:.3f} "
                          f"(tags={result.detail.get('think_leak_count', 0)}, "
                          f"filler={result.detail.get('think_filler_count', 0)})")
            # Summary of quality adjustments
            _mean_adj = float(np.mean(quality_adjustments))
            _mean_pdiv = np.mean([d.get("point_diversity_bonus", 0.0) for d in _quality_details])
            _mean_tpen = np.mean([d.get("token_length_penalty", 0.0) for d in _quality_details])
            _mean_think = np.mean([d.get("think_leak_penalty", 0.0) for d in _quality_details])
            _mean_filler = np.mean([d.get("filler_pattern_penalty", 0.0) for d in _quality_details])
            _mean_tagrep = np.mean([d.get("tag_repetition_penalty", 0.0) for d in _quality_details])
            _n_think_any = sum(1 for d in _quality_details if d.get("think_leak_count", 0) > 0)
            _n_filler_any = sum(1 for d in _quality_details if d.get("filler_pattern_penalty", 0.0) < 0)
            print(f"[MetaReward] quality_adj_mean={_mean_adj:.3f}, "
                  f"point_div_bonus_mean={_mean_pdiv:.3f}, "
                  f"token_len_penalty_mean={_mean_tpen:.3f}, "
                  f"think_leak_penalty_mean={_mean_think:.3f}, "
                  f"think_leak_pct={100*_n_think_any/max(len(_quality_details),1):.0f}%, "
                  f"filler_penalty_mean={_mean_filler:.3f}, "
                  f"filler_pct={100*_n_filler_any/max(len(_quality_details),1):.0f}%, "
                  f"tag_rep_penalty_mean={_mean_tagrep:.3f}")

        # ── Batch-Level Similarity Penalty ──────────────────────────────
        # Detects template collapse: when multiple rubrics for the same
        # question are structurally near-identical despite cosmetic word changes.
        # Three signals combined: (A) point-value multiset Jaccard,
        # (B) content-word Jaccard, (C) content-word bigram Jaccard.
        _SIM_STOP = frozenset({
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'that', 'this', 'it', 'its', 'not', 'no', 'as', 'if', 'so',
            'such', 'very', 'too', 'just', 'also', 'more', 'most', 'some',
            'any', 'all', 'each', 'every', 'both', 'may', 'might', 'can',
            'should', 'would', 'could', 'will', 'shall', 'about', 'into',
            'than', 'then', 'when', 'where', 'how', 'what', 'which', 'who',
            'especially', 'since', 'user', 'using', 'used', 'ensure',
            'provide', 'offer', 'include', 'make', 'sure', 'given',
        })
        _CRIT_RE_SIM = re.compile(
            r"^\s*[-*]\s*\[([+-]?\d+)\]\s*(.+?)(?:\s*\|\s*tags?\s*:\s*(.*))?$",
            re.IGNORECASE,
        )
        from collections import Counter as _Counter
        batch_sim_penalties = np.zeros(len(questions), dtype=np.float32)
        sim_details = []

        for q, r_list in q_items:
            if len(r_list) < 2:
                continue

            # Per-rubric feature extraction
            point_sigs_sorted = []  # sorted point-value tuple (for multiset Jaccard)
            word_sets = []          # set of content words from all criteria
            bigram_sets = []        # set of content-word bigrams
            for _, rubric in r_list:
                pts = []
                words_list = []
                for line in rubric.splitlines():
                    m = _CRIT_RE_SIM.match(line)
                    if m:
                        pts.append(int(m.group(1)))
                        ws = [w for w in re.findall(r'[a-z]{4,}', m.group(2).lower())
                              if w not in _SIM_STOP]
                        words_list.extend(ws)
                point_sigs_sorted.append(tuple(sorted(pts)))
                word_sets.append(set(words_list))
                bgs = set()
                for k in range(len(words_list) - 1):
                    bgs.add((words_list[k], words_list[k + 1]))
                bigram_sets.append(bgs)

            # Compute per-rubric mean similarity vs rest of batch
            for i, (global_idx, _) in enumerate(r_list):
                pair_scores = []
                for j in range(len(r_list)):
                    if j == i:
                        continue
                    # Signal A: point-value multiset Jaccard
                    pi, pj = point_sigs_sorted[i], point_sigs_sorted[j]
                    if pi and pj:
                        ci, cj = _Counter(pi), _Counter(pj)
                        pts_sim = sum((ci & cj).values()) / max(sum((ci | cj).values()), 1)
                    else:
                        pts_sim = 0.0
                    # Signal B: content-word Jaccard
                    wi, wj = word_sets[i], word_sets[j]
                    word_j = len(wi & wj) / max(len(wi | wj), 1) if (wi and wj) else 0.0
                    # Signal C: bigram Jaccard
                    bi, bj = bigram_sets[i], bigram_sets[j]
                    bg_j = len(bi & bj) / max(len(bi | bj), 1) if (bi and bj) else 0.0
                    # Combined weighted score
                    combined = 0.25 * pts_sim + 0.35 * word_j + 0.40 * bg_j
                    pair_scores.append(combined)

                mean_sim = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
                sim_details.append(mean_sim)

                # Penalty: threshold 0.20, linearly to -8.0 at sim=1.0
                if mean_sim > 0.20:
                    penalty = -8.0 * (mean_sim - 0.20) / 0.80
                    batch_sim_penalties[global_idx] = penalty
                    quality_adjustments[global_idx] += penalty

        if sim_details:
            _mean_sim = np.mean(sim_details)
            _mean_pen = np.mean(batch_sim_penalties)
            _n_penalized = int(np.sum(batch_sim_penalties < -0.01))
            print(f"[MetaReward] batch_sim_score_mean={_mean_sim:.2%}, "
                  f"batch_sim_penalty_mean={_mean_pen:.3f}, "
                  f"batch_sim_penalized={_n_penalized}/{len(batch_sim_penalties)}")

        # Per-question answer storage for rollout logging
        question_answers: Dict[int, Dict[int, str]] = {}  # q_idx -> {local_i -> answer}

        # Shared thread-pools – one for each remote service.
        solver_pool = ThreadPoolExecutor(max_workers=max(1, solver_workers))
        oracle_pool = ThreadPoolExecutor(max_workers=max(1, oracle_workers))

        # Track progress with a simple counter protected by a lock.
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
            # Log every 4 events or on completion
            total_done = sd + ed
            if total_done % 4 == 0 or (sd == st and ed == et):
                print(f"[MetaReward]   solver={sd}/{st}  eval={ed}/{et}  questions={qd}/{qt}")

        # ── Helper: generate one answer ────────────────────────────────
        def _gen_answer(q: str, rubric: str) -> str:
            ans = self.solver.generate_answer(q, rubric)
            _log_progress("solver")
            return ans

        # ── Helper: evaluate selected answers against ONE rubric ───────
        def _eval_batch_by_rubric(q: str, answers: List[str], rubric: str) -> List[float]:
            """Returns len(answers) scores, all relative to each other."""
            scores = self.oracle.evaluate_answers_by_rubric(q, answers, rubric)
            with progress_lock:
                progress["eval_done"] += len(answers)
            return scores

        # ── K-sparse rubric selection ──────────────────────────────────
        def _select_rubric_indices(n: int, k: int) -> List[List[int]]:
            """For each of N rubrics, select K other rubric indices to evaluate.

            Returns eval_sets[j] = list of rubric indices that rubric j will
            evaluate (i.e., answers from those rubrics will be scored by rubric j).

            When K >= N-1, returns full cross-eval (all j != i).
            """
            if k <= 0 or k >= n - 1:
                # Full evaluation: each rubric evaluates all others
                return [list(range(n)) for _ in range(n)]

            # Random K-sparse: each answer i is evaluated by K randomly
            # chosen rubrics j.  We build this from the answer perspective
            # then invert to rubric perspective.
            #
            # For each answer i, sample K rubrics from {0..N-1}\{i}
            answer_to_rubrics: Dict[int, List[int]] = {}
            for i in range(n):
                candidates = [j for j in range(n) if j != i]
                chosen = sorted(random.sample(candidates, min(k, len(candidates))))
                answer_to_rubrics[i] = chosen

            # Invert: rubric_to_answers[j] = which answers rubric j needs to score
            rubric_to_answers: Dict[int, List[int]] = defaultdict(list)
            for i, rubs in answer_to_rubrics.items():
                for j in rubs:
                    rubric_to_answers[j].append(i)

            # Return as list-of-lists indexed by rubric j.
            # Include j itself so the oracle always sees a full ranking context
            # (diagonal will be excluded from reward later).
            eval_sets = []
            for j in range(n):
                answer_indices = sorted(set(rubric_to_answers.get(j, [])) | {j})
                eval_sets.append(answer_indices)

            return eval_sets

        # ── Phase 1: Submit solver tasks (optionally capped by solver_n) ──
        solver_futures: Dict[int, List[Tuple[int, Future]]] = {}
        # Track which local indices actually have solver answers
        solver_answer_indices: Dict[int, List[int]] = {}

        for q_idx, (q, items) in enumerate(q_items):
            n = len(items)

            # Determine which rubric indices get solver answers
            if solver_n > 0 and n > solver_n:
                # Randomly select solver_n out of n rubrics for answer generation
                selected = sorted(random.sample(range(n), solver_n))
            else:
                selected = list(range(n))
            solver_answer_indices[q_idx] = selected

            with progress_lock:
                progress["solver_total"] += len(selected)

            if n < 2:
                idx0, r0 = items[0]
                fut = solver_pool.submit(_gen_answer, q, r0)
                solver_futures[q_idx] = [(0, fut)]
            else:
                futs = []
                for local_i in selected:
                    _, rubric = items[local_i]
                    fut = solver_pool.submit(_gen_answer, q, rubric)
                    futs.append((local_i, fut))
                solver_futures[q_idx] = futs

        # ── Phase 2: Coordinate evaluation per question ────────────────
        eval_futures: Dict[int, List] = {}
        # Store the eval_sets for reward computation
        eval_sets_per_q: Dict[int, List[List[int]]] = {}
        coordinator_futures: List[Future] = []
        coordinator_pool = ThreadPoolExecutor(max_workers=len(q_items))

        # Track solver-detected garbage rubrics per question
        solver_garbage: Dict[int, set] = {}  # q_idx -> set of local indices flagged garbage
        # Track solver-detected generic (non-specific) rubrics per question
        solver_general: Dict[int, set] = {}  # q_idx -> set of local indices flagged generic

        def _coordinate_question(q_idx: int, q: str, items: List[Tuple[int, str]], n: int):
            """Wait for solver results, then submit K-sparse oracle evals."""
            # Collect answers for indices that have solver answers
            has_answer = set(solver_answer_indices[q_idx])
            answers = {}  # local_i -> answer text
            for local_i, fut in solver_futures[q_idx]:
                try:
                    answers[local_i] = fut.result(timeout=900)
                except Exception as e:
                    print(f"[MetaReward] Solver error Q{q_idx} rubric {local_i}: {e}")
                    answers[local_i] = ""

            # Store answers for logging before garbage filtering
            question_answers[q_idx] = dict(answers)

            # ── Solver-based garbage detection ─────────────────────────
            # If the solver output contains <GARBAGE_RUBRIC>, the rubric was
            # nonsensical.  Strip the marker but KEEP the answer for evaluation.
            # Garbage rubrics get penalised reward (÷w) and their evaluations
            # of others are down-weighted (weight 1/w).
            garbage_local = set()
            for local_i, ans in list(answers.items()):
                if GARBAGE_RUBRIC_MARKER in ans:
                    garbage_local.add(local_i)
                    # Strip the marker, keep the actual answer
                    cleaned = ans.replace(GARBAGE_RUBRIC_MARKER, "").strip()
                    if cleaned:
                        answers[local_i] = cleaned
                    else:
                        # Solver only output the marker with no answer — remove
                        del answers[local_i]
            if garbage_local:
                n_kept = len([i for i in garbage_local if i in answers])
                print(f"[MetaReward] Q{q_idx}: solver flagged {len(garbage_local)}/{n} "
                      f"rubrics as garbage (indices {sorted(garbage_local)}, "
                      f"{n_kept} kept with answer)")
            solver_garbage[q_idx] = garbage_local

            # ── Solver-based generic rubric detection ──────────────────
            # If the solver output contains <GENERAL_RUBRIC>, the rubric was
            # overly generic / not specific to the question.  Strip the marker
            # but KEEP the answer.  Generic rubrics get penalised own reward
            # (÷w) but their evaluations of others keep normal weight.
            general_local = set()
            for local_i, ans in list(answers.items()):
                if GENERAL_RUBRIC_MARKER in ans:
                    general_local.add(local_i)
                    answers[local_i] = ans.replace(GENERAL_RUBRIC_MARKER, "").strip()

            # ── Programmatic generic detection fallback ──────────
            # Even if the solver didn't flag a rubric, check topic
            # specificity programmatically.  If no question-specific terms
            # appear in the rubric text, mark it as generic.
            # Guard: skip if question has no extractable topic words (all
            # words <5 chars), to avoid false-flagging everything.
            q_topic_words = {w.lower() for w in re.findall(r"[a-zA-Z]{5,}", q)} - _GENERIC_STOP_WORDS
            if q_topic_words:  # only run when we have topic words to check
                for local_i in range(n):
                    if local_i in general_local or local_i in garbage_local:
                        continue
                    rubric_text_i = items[local_i][1]
                    overlap = _has_topic_specificity(q, rubric_text_i, min_overlap=1)
                    if not overlap:
                        general_local.add(local_i)
                        # Check for the notorious "Models answer with" filler
                        if re.search(r'(?i)models?\s+answer\s+with', rubric_text_i):
                            print(f"[MetaReward] Q{q_idx} rubric {local_i}: "
                                  f"programmatic generic flag (\"Models answer with\" pattern)")
                        else:
                            print(f"[MetaReward] Q{q_idx} rubric {local_i}: "
                                  f"programmatic generic flag (no topic-specific terms)")

            if general_local:
                print(f"[MetaReward] Q{q_idx}: {len(general_local)}/{n} "
                      f"rubrics flagged generic (indices {sorted(general_local)})")
            solver_general[q_idx] = general_local

            if n < 2:
                if 0 in garbage_local:
                    # Single rubric flagged garbage — skip oracle entirely
                    eval_futures[q_idx] = []
                    eval_sets_per_q[q_idx] = []
                    return
                with progress_lock:
                    progress["eval_total"] += 1
                fut = oracle_pool.submit(
                    _eval_batch_by_rubric, q, [answers.get(0, "")], items[0][1]
                )
                eval_futures[q_idx] = [("single", fut)]
                eval_sets_per_q[q_idx] = [[0]]
                return

            # Build ordered list of non-garbage answer indices
            answer_keys = sorted(answers.keys())
            if not answer_keys:
                # All answers were garbage — skip oracle entirely
                eval_futures[q_idx] = []
                eval_sets_per_q[q_idx] = []
                return
            n_answers = len(answer_keys)

            # For cross-evaluation: each rubric j evaluates all available answers
            # (K-sparse applies to which rubrics evaluate, not which answers)
            eval_sets = _select_rubric_indices(n, k_sparse)
            eval_sets_per_q[q_idx] = eval_sets

            # For each rubric j, evaluate available answers.
            # Garbage rubrics ARE used as evaluators (their scores are
            # down-weighted by 1/GARBAGE_RUBRIC_WEIGHT in Phase 3).
            total_evals = 0
            batch_futs = []
            for j in range(n):
                # Only use answer indices that exist (garbage w/o answer excluded)
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

        # ── Phase 3: Collect results and compute rewards ───────────────
        for cfut in coordinator_futures:
            cfut.result()

        for q_idx, (q, items) in enumerate(q_items):
            indices = [it[0] for it in items]  # global indices
            n = len(items)

            # Solver-detected garbage indices for this question
            q_garbage = solver_garbage.get(q_idx, set())
            # Solver-detected generic indices for this question
            q_general = solver_general.get(q_idx, set())

            if n < 2:
                if not eval_futures.get(q_idx):
                    # Garbage or no eval — reward stays 0
                    pass
                else:
                    tag, fut = eval_futures[q_idx][0]
                    try:
                        scores = fut.result(timeout=900)
                        consensus_reward = scores[0] if scores else 0.0
                    except Exception as e:
                        print(f"[MetaReward] Eval error Q{q_idx}: {e}")
                        consensus_reward = 0.0
                    qa = quality_adjustments[indices[0]] if RUBRIC_QUALITY_CONFIG.enabled else 0.0
                    rewards[indices[0]] = consensus_reward + qa
            else:
                # Build sparse score matrix
                score_matrix = np.full((n, n), np.nan)
                mask = np.zeros((n, n), dtype=np.float64)

                for j, answer_indices, fut in eval_futures[q_idx]:
                    try:
                        scores = fut.result(timeout=900)
                        for pos, ai in enumerate(answer_indices):
                            score_matrix[ai][j] = scores[pos]
                            mask[ai][j] = 1.0
                    except Exception as e:
                        print(f"[MetaReward] Eval error Q{q_idx} rubric {j}: {e}")

                # Replace NaN with 0 for matrix operations
                observed = np.nan_to_num(score_matrix, nan=0.0)

                # Optional matrix completion
                if use_mc and mask.sum() < n * n and n >= 3:
                    try:
                        completed = als_matrix_completion(
                            observed, mask,
                            rank=min(MC_RANK, n - 1),
                            max_iter=MC_MAX_ITER,
                            reg=MC_REG,
                        )
                        # Use completed matrix for reward, but mark
                        # which entries are real vs imputed
                        reward_matrix = completed
                        # Log completion stats
                        n_imputed = int((1 - mask).sum())
                        print(f"[MetaReward]   Q{q_idx}: matrix completion "
                              f"imputed {n_imputed}/{n*n} entries")
                    except Exception as e:
                        print(f"[MetaReward]   Q{q_idx}: MC failed ({e}), "
                              f"using sparse rewards")
                        reward_matrix = observed
                else:
                    reward_matrix = observed

                # Compute cross-consensus reward per rubric (weighted by garbage status)
                w = GARBAGE_RUBRIC_WEIGHT
                for i in range(n):
                    if use_mc and mask.sum() < n * n:
                        # With MC: use all off-diagonal entries from completed matrix
                        entries = [(reward_matrix[i][j], j) for j in range(n) if j != i]
                    else:
                        # Without MC: use only observed off-diagonal entries
                        entries = [(reward_matrix[i][j], j)
                                   for j in range(n)
                                   if j != i and mask[i][j] > 0]

                    if entries:
                        # Weighted average: garbage evaluators' scores get weight 1/w
                        weighted_sum = 0.0
                        weight_sum = 0.0
                        for score_val, j in entries:
                            wt = (1.0 / w) if j in q_garbage else 1.0
                            weighted_sum += wt * score_val
                            weight_sum += wt
                        consensus = weighted_sum / weight_sum if weight_sum > 0 else 0.0
                    else:
                        consensus = 0.0

                    # Garbage rubric's own reward is penalised by /w
                    if i in q_garbage:
                        consensus = consensus / w

                    # Generic rubric's own reward is penalised (but NOT its evaluator weight)
                    if i in q_general:
                        consensus = consensus / GENERAL_RUBRIC_WEIGHT

                    # Add rubric quality adjustment
                    qa = quality_adjustments[indices[i]] if RUBRIC_QUALITY_CONFIG.enabled else 0.0
                    rewards[indices[i]] = consensus + qa

            with progress_lock:
                progress["q_done"] += 1

            print(f"[MetaReward]   Q{q_idx+1}/{len(q_items)} done, "
                  f"rewards={[rewards[idx].item() for idx in indices]}")

        # ── Cleanup ────────────────────────────────────────────────────
        solver_pool.shutdown(wait=False)
        oracle_pool.shutdown(wait=False)
        coordinator_pool.shutdown(wait=False)

        # ── Log sample (question, rubrics) to stdout ──────────────
        self._log_sample_rubrics(q_items, rewards)

        # ── Rollout logging ────────────────────────────────────────
        # Use global_step directly for correct file naming across resumes.
        # Fallback to internal counter only when global_step is unavailable.
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
        # Log first question's rubrics in full, plus a brief summary of others
        n_to_show = min(1, len(q_items))  # show 1 full example per step
        print(f"\n{'='*80}")
        print(f"[RubricSample] Step {self._step_counter} — "
              f"{len(q_items)} questions, showing {n_to_show} full example(s)")
        print(f"{'='*80}")

        for q_idx in range(n_to_show):
            q, items = q_items[q_idx]
            indices = [it[0] for it in items]
            print(f"\n[Q{q_idx+1}] {q[:300]}")
            print(f"  ({len(items)} rubrics generated)")
            for local_i, (global_idx, rubric) in enumerate(items):
                r = rewards[global_idx].item()
                # Show first 500 chars of each rubric
                rubric_preview = rubric[:500].replace('\n', '\n    ')
                print(f"  --- Rubric {local_i+1} (reward={r:.3f}) ---")
                print(f"    {rubric_preview}")
                if len(rubric) > 500:
                    print(f"    ... ({len(rubric)} chars total)")

        # Summary for the rest
        if len(q_items) > n_to_show:
            print(f"\n[RubricSample] Other questions summary:")
            for q_idx in range(n_to_show, len(q_items)):
                q, items = q_items[q_idx]
                indices = [it[0] for it in items]
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
                    indices = [it[0] for it in items]
                    rubric_texts = [it[1] for it in items]
                    answers_dict = question_answers.get(q_idx, {})

                    rollouts = []
                    for local_i, (global_idx, rubric) in enumerate(items):
                        rollouts.append({
                            "rubric": rubric[:2000],  # truncate for readability
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
