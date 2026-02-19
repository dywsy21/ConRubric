import os
import threading
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from tqdm import tqdm
from src.evaluation.judge import Oracle
from src.models.solver import Solver

# Per-service parallel settings
DEFAULT_SOLVER_WORKERS = int(os.environ.get("GRM_SOLVER_WORKERS", 4))
DEFAULT_ORACLE_WORKERS = int(os.environ.get("GRM_ORACLE_WORKERS", 4))

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
        Computes the Consensus-Based Meta-Reward with pipelined execution.

        All questions are processed concurrently: while remote Solver/Oracle
        calls are in-flight for question i, work for question i+1 is already
        being submitted.  Two shared thread-pools (solver_pool, oracle_pool)
        enforce per-service concurrency limits.

        Pipeline per question:
            1. Generate N answers via Solver (parallel within solver_pool)
            2. As soon as ALL answers for a question arrive, submit N*(N-1)
               cross-evaluation tasks to the oracle_pool.
            3. When all evals finish, compute per-rubric reward.

        Every rollout gets its own reward for proper RL policy gradients.
        """
        if solver_workers is None:
            solver_workers = DEFAULT_SOLVER_WORKERS
        if oracle_workers is None:
            oracle_workers = DEFAULT_ORACLE_WORKERS

        print(f"[MetaReward] Computing rewards for {len(questions)} samples, "
              f"solver_workers={solver_workers}, oracle_workers={oracle_workers}")

        # ── Group by question ──────────────────────────────────────────
        q_to_rubrics: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for idx, (q, r) in enumerate(zip(questions, rubrics)):
            q_to_rubrics[q].append((idx, r))

        q_items = list(q_to_rubrics.items())
        print(f"[MetaReward] Processing {len(q_items)} unique questions (pipelined)")

        rewards = torch.zeros(len(questions), dtype=torch.float32)

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

        # ── Helper: evaluate one (answer, rubric) pair ─────────────────
        def _eval_pair(q: str, answer: str, rubric: str) -> float:
            score = self.oracle.evaluate_answer(q, answer, rubric=rubric)
            _log_progress("eval")
            return score

        # ── Phase 1: Submit ALL solver tasks across ALL questions ──────
        # solver_futures[q_idx] = list of (rubric_local_idx, Future[str])
        solver_futures: Dict[int, List[Tuple[int, Future]]] = {}

        for q_idx, (q, items) in enumerate(q_items):
            n = len(items)
            with progress_lock:
                progress["solver_total"] += n

            if n < 2:
                # Degenerate case – still submit to solver_pool for uniformity
                idx0, r0 = items[0]
                fut = solver_pool.submit(_gen_answer, q, r0)
                solver_futures[q_idx] = [(0, fut)]
            else:
                futs = []
                for local_i, (_, rubric) in enumerate(items):
                    fut = solver_pool.submit(_gen_answer, q, rubric)
                    futs.append((local_i, fut))
                solver_futures[q_idx] = futs

        # ── Phase 2: For each question, wait for its answers then
        #    immediately submit cross-eval tasks.  A coordinator thread
        #    per question handles this so questions overlap. ─────────────
        # eval_futures[q_idx] = list of (i, j, Future[float])
        eval_futures: Dict[int, List[Tuple[int, int, Future]]] = {}
        coordinator_futures: List[Future] = []
        coordinator_pool = ThreadPoolExecutor(max_workers=len(q_items))

        def _coordinate_question(q_idx: int, q: str, items: List[Tuple[int, str]], n: int):
            """Wait for this question's solver results, then submit oracle evals."""
            # Collect answers in order
            answers = [""] * n
            for local_i, fut in solver_futures[q_idx]:
                try:
                    answers[local_i] = fut.result(timeout=900)  # 15 min max
                except Exception as e:
                    print(f"[MetaReward] Solver error Q{q_idx} rubric {local_i}: {e}")

            if n < 2:
                # Single rubric – direct eval, no cross-eval
                idx0 = items[0][0]
                fut = oracle_pool.submit(_eval_pair, q, answers[0], items[0][1])
                with progress_lock:
                    progress["eval_total"] += 1
                eval_futures[q_idx] = [(0, -1, fut)]
                return

            # Submit all cross-eval pairs to oracle_pool
            pairs = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        rubric_j = items[j][1]
                        fut = oracle_pool.submit(_eval_pair, q, answers[i], rubric_j)
                        pairs.append((i, j, fut))
            with progress_lock:
                progress["eval_total"] += len(pairs)
            eval_futures[q_idx] = pairs

        for q_idx, (q, items) in enumerate(q_items):
            n = len(items)
            cfut = coordinator_pool.submit(_coordinate_question, q_idx, q, items, n)
            coordinator_futures.append(cfut)

        # ── Phase 3: Wait for all coordinators (i.e. all oracle tasks
        #    have been submitted), then collect results. ─────────────────
        for cfut in coordinator_futures:
            cfut.result()  # raises on error

        # Now all oracle futures are in eval_futures. Collect them.
        for q_idx, (q, items) in enumerate(q_items):
            indices = [it[0] for it in items]
            n = len(items)

            if n < 2:
                # Single rubric
                for i, j, fut in eval_futures[q_idx]:
                    try:
                        score = fut.result(timeout=900)
                    except Exception as e:
                        print(f"[MetaReward] Eval error Q{q_idx}: {e}")
                        score = 0.0
                    rewards[indices[0]] = score
            else:
                score_matrix = np.zeros((n, n))
                for i, j, fut in eval_futures[q_idx]:
                    try:
                        score_matrix[i][j] = fut.result(timeout=900)
                    except Exception as e:
                        print(f"[MetaReward] Eval error Q{q_idx} ({i},{j}): {e}")

                for i in range(n):
                    avg_score = score_matrix[i].sum() / (n - 1)
                    rewards[indices[i]] = avg_score

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
