import os
import random
import threading
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
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
                       oracle_workers: int = None) -> torch.Tensor:
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

        print(f"[MetaReward] Computing rewards for {len(questions)} samples, "
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
            for idx, rubric in enumerate(rubrics):
                result = score_rubric_quality(rubric, RUBRIC_QUALITY_CONFIG)
                quality_adjustments[idx] = result.total_adjustment

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

            if n < 2:
                with progress_lock:
                    progress["eval_total"] += 1
                fut = oracle_pool.submit(
                    _eval_batch_by_rubric, q, [answers.get(0, "")], items[0][1]
                )
                eval_futures[q_idx] = [("single", fut)]
                eval_sets_per_q[q_idx] = [[0]]
                return

            # Build ordered list of answer indices that exist
            answer_keys = sorted(answers.keys())
            n_answers = len(answer_keys)

            # For cross-evaluation: each rubric j evaluates all available answers
            # (K-sparse applies to which rubrics evaluate, not which answers)
            eval_sets = _select_rubric_indices(n, k_sparse)
            eval_sets_per_q[q_idx] = eval_sets

            # For each rubric j, only evaluate answers that actually exist
            total_evals = 0
            batch_futs = []
            for j in range(n):
                # Only use answer indices that exist (intersection with solver answers)
                available = [ai for ai in answer_keys if ai != j]
                if not available:
                    # If the only answer is from this rubric, include it anyway
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

            if n < 2:
                tag, fut = eval_futures[q_idx][0]
                try:
                    scores = fut.result(timeout=900)
                    consensus_reward = scores[0] if scores else 0.0
                except Exception as e:
                    print(f"[MetaReward] Eval error Q{q_idx}: {e}")
                    consensus_reward = 0.0
                # Combined reward
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

                # Compute cross-consensus reward per rubric
                for i in range(n):
                    if use_mc and mask.sum() < n * n:
                        # With MC: use all off-diagonal entries from completed matrix
                        off_diag = [reward_matrix[i][j] for j in range(n) if j != i]
                    else:
                        # Without MC: use only observed off-diagonal entries
                        off_diag = [reward_matrix[i][j]
                                    for j in range(n)
                                    if j != i and mask[i][j] > 0]
                    consensus = sum(off_diag) / len(off_diag) if off_diag else 0.0

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

        print(f"[MetaReward] All rewards computed, mean={rewards.mean():.3f}")
        return rewards

    def _generate_answer(self, question: str, rubric: str) -> str:
        return self.solver.generate_answer(question, rubric) 
