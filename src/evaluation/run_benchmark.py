import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load env vars BEFORE importing HuggingFace libraries
load_dotenv()

# Set HF_HOME to absolute path if relative (must be done before importing transformers/datasets)
hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")
print(f"HF_HOME set to: {hf_home}")

import torch
from tqdm import tqdm
from datasets import load_dataset

from src.config import ProjectConfig
from src.models.grm import RubricGenerator
from src.evaluation.judge import Judge
from src.evaluation.run_rubric_quality_benchmark import run_healthbench_rubric_quality


def _score_pairs_in_parallel(judge: Judge, pairs: List[Dict[str, Any]], max_workers: int) -> List[Dict[str, Any]]:
    """Score (chosen, rejected) pairs in one batched parallel call to fully use remote vLLM."""
    if not pairs:
        return []

    questions = []
    answers = []
    rubrics = []
    for p in pairs:
        questions.extend([p["prompt"], p["prompt"]])
        answers.extend([p["chosen"], p["rejected"]])
        rubrics.extend([p["rubric"], p["rubric"]])

    scores = judge.evaluate_batch(
        questions=questions,
        answers=answers,
        rubrics=rubrics,
        show_progress=False,
        max_workers=max_workers,
    )

    out = []
    for i, p in enumerate(pairs):
        out.append(
            {
                **p,
                "score_chosen": float(scores[2 * i]),
                "score_rejected": float(scores[2 * i + 1]),
            }
        )
    return out

def run_reward_bench(grm: RubricGenerator, judge: Judge, num_samples: int = None, output_dir: str = "results", eval_batch_size: int = 32, max_workers: int = 8):
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

    pending_pairs = []

    def flush_pending():
        nonlocal correct, total, results, pending_pairs, subset_stats
        if not pending_pairs:
            return
        scored = _score_pairs_in_parallel(judge, pending_pairs, max_workers=max_workers)
        for item in scored:
            score_chosen = item["score_chosen"]
            score_rejected = item["score_rejected"]
            subset = item["subset"]

            is_correct = False
            val = 0.0
            if score_chosen > score_rejected:
                correct += 1
                val = 1.0
                is_correct = True
            elif score_chosen == score_rejected:
                correct += 0.5
                val = 0.5

            total += 1
            if subset not in subset_stats:
                subset_stats[subset] = {"correct": 0, "total": 0}
            subset_stats[subset]["correct"] += val
            subset_stats[subset]["total"] += 1

            results.append({
                "prompt": item["prompt"],
                "rubric": item["rubric"],
                "score_chosen": score_chosen,
                "score_rejected": score_rejected,
                "is_correct": is_correct,
                "subset": subset,
            })
        pending_pairs = []

    for row in tqdm(dataset, desc="RewardBench"):
        prompt = row['prompt']
        rubric = grm.generate_rubric(prompt)
        pending_pairs.append({
            "prompt": prompt,
            "chosen": row['chosen'],
            "rejected": row['rejected'],
            "rubric": rubric,
            "subset": row.get('subset', 'unknown'),
        })
        if len(pending_pairs) >= eval_batch_size:
            flush_pending()

    flush_pending()
        
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

def run_ppe_benchmark(grm: RubricGenerator, judge: Judge, num_samples: int = None, output_dir: str = "results", eval_batch_size: int = 32, max_workers: int = 8):
    print("\nRunning PPE Benchmark (using lmarena-ai/PPE-Debug)...")
    try:
        dataset = load_dataset("lmarena-ai/PPE-Debug", split="test")
    except Exception as e:
        print(f"Could not load PPE dataset: {e}")
        return

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    
    results = []
    correct = 0
    total = 0
    
    pending_pairs = []

    def flush_pending():
        nonlocal correct, total, results, pending_pairs
        if not pending_pairs:
            return
        scored = _score_pairs_in_parallel(judge, pending_pairs, max_workers=max_workers)
        for item in scored:
            score_chosen = item["score_chosen"]
            score_rejected = item["score_rejected"]

            is_correct = False
            val = 0.0
            if score_chosen > score_rejected:
                correct += 1
                val = 1.0
                is_correct = True
            elif score_chosen == score_rejected:
                correct += 0.5
                val = 0.5

            total += 1
            results.append({
                "prompt": item["prompt"],
                "rubric": item["rubric"],
                "score_chosen": score_chosen,
                "score_rejected": score_rejected,
                "is_correct": is_correct,
                "winner": item["winner"],
            })
        pending_pairs = []

    for row in tqdm(dataset, desc="PPE"):
        prompt = row['prompt']
        response_1 = row['response_1']
        response_2 = row['response_2']
        winner = row['winner']
        
        if winner not in ['model_a', 'model_b']:
            continue # Skip ties or errors for now
            
        chosen = response_1 if winner == 'model_a' else response_2
        rejected = response_2 if winner == 'model_a' else response_1
        
        rubric = grm.generate_rubric(prompt)
        pending_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "rubric": rubric,
            "winner": winner,
        })
        if len(pending_pairs) >= eval_batch_size:
            flush_pending()

    flush_pending()
        
    accuracy = correct / total if total > 0 else 0
    print(f"PPE Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "ppe_results.json"), "w") as f:
        json.dump({
            "accuracy": accuracy, 
            "details": results
        }, f, indent=2)

def run_rmb_benchmark(grm: RubricGenerator, judge: Judge, num_samples: int = None, output_dir: str = "results", eval_batch_size: int = 32, max_workers: int = 8):
    print("\nRunning RMB Benchmark (using dd101bb/RMB_dataset)...")
    try:
        dataset = load_dataset("dd101bb/RMB_dataset", split="pairwise")
    except Exception as e:
        print(f"Could not load RMB dataset: {e}")
        return

    if num_samples:
        dataset = dataset.select(range(min(len(dataset), num_samples)))
    
    results = []
    correct = 0
    total = 0
    
    subset_stats = {}

    pending_pairs = []

    def flush_pending():
        nonlocal correct, total, results, pending_pairs, subset_stats
        if not pending_pairs:
            return
        scored = _score_pairs_in_parallel(judge, pending_pairs, max_workers=max_workers)
        for item in scored:
            score_chosen = item["score_chosen"]
            score_rejected = item["score_rejected"]
            subset = item["subset"]

            is_correct = False
            val = 0.0
            if score_chosen > score_rejected:
                correct += 1
                val = 1.0
                is_correct = True
            elif score_chosen == score_rejected:
                correct += 0.5
                val = 0.5

            total += 1
            if subset not in subset_stats:
                subset_stats[subset] = {"correct": 0, "total": 0}
            subset_stats[subset]["correct"] += val
            subset_stats[subset]["total"] += 1

            results.append({
                "prompt": item["prompt"],
                "rubric": item["rubric"],
                "score_chosen": score_chosen,
                "score_rejected": score_rejected,
                "is_correct": is_correct,
                "category": item["category"],
            })
        pending_pairs = []

    for row in tqdm(dataset, desc="RMB"):
        # Extract prompt from conversation
        conversation = row['conversation']
        # Assuming the last user message is the prompt, or we concatenate.
        # For simplicity, let's take the last user message content.
        prompt = ""
        for msg in reversed(conversation):
            if msg['role'] == 'user':
                prompt = msg['content']
                break
        
        if not prompt:
            continue
            
        responses = row['responses']
        preferred_index = row['preferred_index']
        
        if preferred_index not in [0, 1]:
            continue
            
        chosen = responses[preferred_index]['answer']
        rejected = responses[1 - preferred_index]['answer']
        
        category = row.get('category_path', 'unknown')
        
        rubric = grm.generate_rubric(prompt)
        subset = category.split('/')[1] if '/' in category else category
        pending_pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "rubric": rubric,
            "category": category,
            "subset": subset,
        })
        if len(pending_pairs) >= eval_batch_size:
            flush_pending()

    flush_pending()
        
    accuracy = correct / total if total > 0 else 0
    print(f"RMB Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    print("\nSubset Breakdown:")
    for subset, stats in subset_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {subset}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "rmb_results.json"), "w") as f:
        json.dump({
            "accuracy": accuracy, 
            "subset_stats": subset_stats,
            "details": results
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained GRM model")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["rewardbench"], help="Benchmarks to run: rewardbench, ppe, rmb, healthbench_rubric")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to run (for debugging)")
    parser.add_argument("--output_dir", type=str, default="out/bench", help="Output directory")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Number of pairwise items per batched judge scoring call")
    parser.add_argument("--eval_workers", type=int, default=int(os.getenv("GRM_ORACLE_WORKERS", "8")), help="Parallel workers for judge API calls")
    parser.add_argument("--healthbench_benchmark_ratio", type=float, default=0.2, help="Reserved prompt ratio for HealthBench rubric benchmark")
    parser.add_argument("--healthbench_split_seed", type=int, default=42, help="Split seed to guarantee no SFT/benchmark leakage")
    
    args = parser.parse_args()
    
    # Initialize models
    project_config = ProjectConfig()
    grm = RubricGenerator(args.model_path)
    judge = Judge(
        model_name=project_config.oracle_model_name,
        api_key=project_config.oracle_api_key,
        api_base=project_config.oracle_api_base
    )
    
    if "rewardbench" in args.benchmarks:
        run_reward_bench(grm, judge, args.num_samples, args.output_dir, eval_batch_size=args.eval_batch_size, max_workers=args.eval_workers)
        
    if "ppe" in args.benchmarks:
        run_ppe_benchmark(grm, judge, args.num_samples, args.output_dir, eval_batch_size=args.eval_batch_size, max_workers=args.eval_workers)
        
    if "rmb" in args.benchmarks:
        run_rmb_benchmark(grm, judge, args.num_samples, args.output_dir, eval_batch_size=args.eval_batch_size, max_workers=args.eval_workers)

    if "healthbench_rubric" in args.benchmarks:
        run_healthbench_rubric_quality(
            grm=grm,
            judge=judge,
            output_dir=args.output_dir,
            benchmark_ratio=args.healthbench_benchmark_ratio,
            split_seed=args.healthbench_split_seed,
            max_prompts=args.num_samples,
            max_completions_per_prompt=8,
            eval_workers=args.eval_workers,
        )
