import torch
import numpy as np
from typing import List, Dict
from collections import defaultdict
from src.evaluation.judge import Oracle
from src.models.solver import Solver

class MetaRewardFunction:
    def __init__(self, solver_model_name: str, oracle_model_name: str, 
                 oracle_api_key: str = None, oracle_api_base: str = None,
                 solver_remote: bool = False, solver_api_key: str = None, solver_api_base: str = None):
        self.solver_model_name = solver_model_name
        self.oracle = Oracle(
            model_name=oracle_model_name,
            api_key=oracle_api_key,
            api_base=oracle_api_base
        )
        # Initialize Solver
        # Note: In a distributed setting (Ray/Verl), this might need to be handled differently (e.g. as an Actor)
        self.solver = Solver(
            model_name=solver_model_name,
            is_remote=solver_remote,
            api_key=solver_api_key,
            api_base=solver_api_base
        )
        
    def compute_reward(self, questions: List[str], rubrics: List[str]) -> torch.Tensor:
        """
        Computes the Consensus-Based Meta-Reward.
        
        Logic:
        1. Group inputs by Question.
        2. For each Question, we have N rubrics.
        3. Generate N answers (one per rubric).
        4. Cross-Evaluate: Evaluate answer A_i against Rubric R_j (for all j != i).
        5. Reward R_i = Mean(Score(A_i, R_j))
        """
        
        # Group by question to handle the N samples per prompt
        # Map question -> list of (index, rubric)
        q_to_rubrics = defaultdict(list)
        for idx, (q, r) in enumerate(zip(questions, rubrics)):
            q_to_rubrics[q].append((idx, r))
            
        rewards = torch.zeros(len(questions), dtype=torch.float32)
        
        for q, items in q_to_rubrics.items():
            indices = [item[0] for item in items]
            current_rubrics = [item[1] for item in items]
            n = len(current_rubrics)
            
            if n < 2:
                # Fallback if only 1 rubric per question (can't do cross-eval properly)
                # Just evaluate against itself or generic anchor
                print(f"Warning: Only {n} rubric(s) for question '{q[:20]}...'. Skipping cross-eval.")
                for idx, r in zip(indices, current_rubrics):
                    # Fallback: Generate answer and eval against Anchor Rubrics (old method)
                    ans = self.solver.generate_answer(q, r)
                    # evaluate_answer now returns a float directly
                    score = self.oracle.evaluate_answer(q, ans) 
                    rewards[idx] = score
                continue

            # 1. Generate Answers
            # answers[i] corresponds to rubric[i]
            answers = self.solver.generate_batch([q]*n, current_rubrics)
            
            # 2. Cross-Evaluation Matrix
            # matrix[i][j] = Score of Answer i evaluated by Rubric j
            score_matrix = np.zeros((n, n))
            
            for i in range(n): # Answer i
                for j in range(n): # Rubric j
                    if i == j:
                        continue # Skip self-evaluation for consensus score (optional)
                    
                    # Evaluate Answer i against Rubric j
                    # evaluate_answer now returns a float directly
                    score = self.oracle.evaluate_answer(q, answers[i], rubric=current_rubrics[j])
                    score_matrix[i][j] = score
            
            # 3. Compute Rewards
            # Reward for Rubric i is the average score of Answer i against all other Rubrics j!=i
            for i in range(n):
                # Mean of row i, excluding diagonal (which is 0)
                # Sum / (N-1)
                avg_score = score_matrix[i].sum() / (n - 1)
                rewards[indices[i]] = avg_score
                
        return rewards

    def _generate_answer(self, question: str, rubric: str) -> str:
        return self.solver.generate_answer(question, rubric)
        # prompt = f"Rubric:\n{rubric}\n\nQuestion:\n{question}\n\nAnswer:"
        return "Placeholder Answer" 

