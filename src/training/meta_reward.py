import os
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.evaluation.judge import Oracle
from src.models.solver import Solver

# Default parallel settings
DEFAULT_MAX_WORKERS = int(os.environ.get("GRM_REWARD_WORKERS", 8))

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
        
    def compute_reward(self, questions: List[str], rubrics: List[str], 
                       max_workers: int = None) -> torch.Tensor:
        """
        Computes the Consensus-Based Meta-Reward with parallel execution.
        
        Logic:
        1. Group inputs by Question.
        2. For each Question, we have N rubrics.
        3. Generate N answers (one per rubric) - PARALLEL
        4. Cross-Evaluate: Evaluate answer A_i against Rubric R_j (for all j != i) - PARALLEL
        5. Reward R_i = Mean(Score(A_i, R_j))
        """
        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS
        
        print(f"[MetaReward] Computing rewards for {len(questions)} samples, max_workers={max_workers}")
        
        # Group by question to handle the N samples per prompt
        # Map question -> list of (index, rubric)
        q_to_rubrics = defaultdict(list)
        for idx, (q, r) in enumerate(zip(questions, rubrics)):
            q_to_rubrics[q].append((idx, r))
            
        rewards = torch.zeros(len(questions), dtype=torch.float32)
        
        # Progress bar for questions
        q_items = list(q_to_rubrics.items())
        print(f"[MetaReward] Processing {len(q_items)} unique questions")
        
        for q_idx, (q, items) in enumerate(q_items):
            print(f"[MetaReward] Question {q_idx+1}/{len(q_items)}: {len(items)} rubrics")
            indices = [item[0] for item in items]
            current_rubrics = [item[1] for item in items]
            n = len(current_rubrics)
            
            if n < 2:
                # Fallback if only 1 rubric per question (can't do cross-eval properly)
                print(f"Warning: Only {n} rubric(s) for question '{q[:20]}...'. Skipping cross-eval.")
                for idx, r in zip(indices, current_rubrics):
                    ans = self.solver.generate_answer(q, r)
                    score = self.oracle.evaluate_answer(q, ans) 
                    rewards[idx] = score
                continue

            # 1. Generate Answers in PARALLEL
            # answers[i] corresponds to rubric[i]
            print(f"[MetaReward]   Generating {n} answers...")
            answers = self.solver.generate_batch([q]*n, current_rubrics, 
                                                  show_progress=False, 
                                                  max_workers=max_workers)
            print(f"[MetaReward]   Generated {len(answers)} answers")
            
            # 2. Cross-Evaluation Matrix - PARALLEL
            # Build list of all (i, j) pairs to evaluate (excluding diagonal)
            eval_tasks = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        eval_tasks.append((i, j, q, answers[i], current_rubrics[j]))
            
            score_matrix = np.zeros((n, n))
            
            def evaluate_single(args: Tuple[int, int, str, str, str]) -> Tuple[int, int, float]:
                i, j, question, answer, rubric = args
                score = self.oracle.evaluate_answer(question, answer, rubric=rubric)
                return i, j, score
            
            # Execute evaluations in parallel (or sequentially if max_workers=1)
            print(f"[MetaReward]   Cross-evaluating {len(eval_tasks)} pairs...")
            completed = 0
            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
                futures = {executor.submit(evaluate_single, task): task[:2] for task in eval_tasks}
                
                for future in as_completed(futures):
                    try:
                        i, j, score = future.result(timeout=600)  # 10 min timeout per eval
                        score_matrix[i][j] = score
                        completed += 1
                        if completed % 4 == 0 or completed == len(eval_tasks):
                            print(f"[MetaReward]     Completed {completed}/{len(eval_tasks)} evals")
                    except Exception as e:
                        print(f"[MetaReward]     Error in eval: {e}")
                        i_idx, j_idx = futures[future]
                        score_matrix[i_idx][j_idx] = 0.0
            
            # 3. Compute Rewards
            # Reward for Rubric i is the average score of Answer i against all other Rubrics j!=i
            for i in range(n):
                avg_score = score_matrix[i].sum() / (n - 1)
                rewards[indices[i]] = avg_score
            
            print(f"[MetaReward]   Question {q_idx+1} done, avg rewards: {[rewards[idx].item() for idx in indices]}")
                
        print(f"[MetaReward] All rewards computed, mean={rewards.mean():.3f}")
        return rewards

    def _generate_answer(self, question: str, rubric: str) -> str:
        return self.solver.generate_answer(question, rubric) 

