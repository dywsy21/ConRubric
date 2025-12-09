import os
import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

# Load env vars
load_dotenv()

from src.config import ProjectConfig
from src.models.grm import RubricGenerator
from src.evaluation.judge import Judge

def run_reward_bench(grm: RubricGenerator, judge: Judge, num_samples: int = None, output_dir: str = "results"):
    print(f"\nRunning RewardBench Evaluation...")
    try:
        # 'filtered' split is commonly used for reliable evaluation
        dataset = load_dataset("allenai/reward-bench", split="filtered")
    except Exception as e:
        print(f"Could not load RewardBench: {e}")
        return

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    
    results = []
    correct = 0
    total = 0
    
    # Group by subset for detailed reporting
    subset_stats = {}

    for row in tqdm(dataset, desc="RewardBench"):
        prompt = row['prompt']
        chosen = row['chosen']
        rejected = row['rejected']
        subset = row.get('subset', 'unknown')
        
        # 1. Generate Rubric
        rubric = grm.generate_rubric(prompt)
        
        # 2. Score Chosen and Rejected
        # Judge expects: evaluate_answer(question, answer, rubric) -> score
        score_chosen = judge.evaluate_answer(prompt, chosen, rubric)
        score_rejected = judge.evaluate_answer(prompt, rejected, rubric)
        
        is_correct = False
        val = 0.0
        if score_chosen > score_rejected:
            correct += 1
            val = 1.0
            is_correct = True
        elif score_chosen == score_rejected:
            correct += 0.5 # Tie
            val = 0.5
            
        total += 1
        
        # Update subset stats
        if subset not in subset_stats:
            subset_stats[subset] = {"correct": 0, "total": 0}
        subset_stats[subset]["correct"] += val
        subset_stats[subset]["total"] += 1
        
        results.append({
            "prompt": prompt,
            "rubric": rubric,
            "score_chosen": score_chosen,
            "score_rejected": score_rejected,
            "is_correct": is_correct,
            "subset": subset
        })
        
    accuracy = correct / total if total > 0 else 0
    print(f"RewardBench Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    print("\nSubset Breakdown:")
    for subset, stats in subset_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {subset}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "reward_bench_results.json"), "w") as f:
        json.dump({
            "accuracy": accuracy, 
            "subset_stats": subset_stats,
            "details": results
        }, f, indent=2)

def run_ppe_benchmark(grm: RubricGenerator, judge: Judge, output_dir: str):
    print("\nRunning PPE Benchmark...")
    print("PPE Benchmark dataset not found/implemented. Skipping.")
    # TODO: Implement PPE loading if dataset is available

def run_rmb_benchmark(grm: RubricGenerator, judge: Judge, output_dir: str):
    print("\nRunning RMB Benchmark...")
    print("RMB Benchmark dataset not found/implemented. Skipping.")
    # TODO: Implement RMB loading if dataset is available

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained GRM model")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["rewardbench"], help="Benchmarks to run: rewardbench, ppe, rmb")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to run (for debugging)")
    parser.add_argument("--output_dir", type=str, default="results/benchmarks", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize models
    grm = RubricGenerator(args.model_path)
    judge = Judge() # Uses env vars for API key
    
    if "rewardbench" in args.benchmarks:
        run_reward_bench(grm, judge, args.num_samples, args.output_dir)
        
    if "ppe" in args.benchmarks:
        run_ppe_benchmark(grm, judge, args.output_dir)
        
    if "rmb" in args.benchmarks:
        run_rmb_benchmark(grm, judge, args.output_dir)
