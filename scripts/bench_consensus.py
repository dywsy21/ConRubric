#!/usr/bin/env python3
"""
Cross-rubric consensus benchmark — measures what RL was trained to optimize.

For each prompt:
1. Generate K rubrics from the GRM
2. Score each completion with each rubric using the judge
3. Measure inter-rubric agreement (low variance = good consensus)

This directly tests the RL training objective: rubrics that produce
consistent scores despite being independently generated.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr
from dotenv import load_dotenv

load_dotenv()

hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")

from tqdm import tqdm

from src.config import ProjectConfig
from src.data.prepare_healthbench import ensure_healthbench_splits
from src.evaluation.judge import Judge
from src.models.grm import RubricGenerator


def run_consensus_benchmark(
    grm: RubricGenerator,
    judge: Judge,
    max_prompts: int = 30,
    num_rubrics: int = 5,
    max_completions: int = 4,
    eval_workers: int = 8,
):
    """
    Generate K rubrics per prompt, score completions with each rubric,
    and measure cross-rubric agreement.
    """
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=0.2,
        seed=42,
        force_rebuild=False,
    )

    grouped = defaultdict(list)
    with open(split_paths.benchmark_meta_eval, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            grouped[row.get("prompt_id")].append(row)

    prompt_items = list(grouped.items())[:max_prompts]
    print(f"Consensus benchmark: {len(prompt_items)} prompts × {num_rubrics} rubrics × ≤{max_completions} completions")

    results = []

    for prompt_id, rows in tqdm(prompt_items, desc="Consensus"):
        prompt = rows[0]["prompt"]
        if isinstance(prompt, list):
            parts = []
            for m in prompt:
                if isinstance(m, dict):
                    c = (m.get("content") or "").strip()
                    if c:
                        parts.append(c)
            prompt_text = "\n".join(parts)
        else:
            prompt_text = str(prompt)

        # Generate K rubrics
        rubrics = []
        for _ in range(num_rubrics):
            rubrics.append(grm.generate_rubric(prompt_text))

        rows = rows[:max_completions]
        completions = [r.get("completion", "") for r in rows]
        labels = []
        for r in rows:
            bl = r.get("binary_labels", [])
            labels.append(float(sum(1 for x in bl if x) / len(bl)) if bl else 0.0)

        # Score matrix: [num_rubrics × num_completions]
        score_matrix = np.zeros((len(rubrics), len(completions)))

        # Batch all judge calls for this prompt
        q_batch = []
        c_batch = []
        r_batch = []
        for ri, rubric in enumerate(rubrics):
            for ci, comp in enumerate(completions):
                q_batch.append(prompt_text)
                c_batch.append(comp)
                r_batch.append(rubric)

        scores = judge.evaluate_batch(
            questions=q_batch,
            answers=c_batch,
            rubrics=r_batch,
            show_progress=False,
            max_workers=eval_workers,
        )

        idx = 0
        for ri in range(len(rubrics)):
            for ci in range(len(completions)):
                score_matrix[ri, ci] = float(scores[idx])
                idx += 1

        # Metrics per prompt
        # 1. Score variance across rubrics for each completion (lower = better consensus)
        per_completion_var = np.var(score_matrix, axis=0)  # variance across rubrics
        avg_score_var = float(np.mean(per_completion_var))

        # 2. Ranking agreement: for each pair of rubrics, compute Spearman rank correlation
        #    High agreement = rubrics rank completions similarly
        spearman_vals = []
        for i in range(len(rubrics)):
            for j in range(i + 1, len(rubrics)):
                if len(set(score_matrix[i])) >= 2 and len(set(score_matrix[j])) >= 2:
                    r, _ = spearmanr(score_matrix[i], score_matrix[j])
                    if not math.isnan(r):
                        spearman_vals.append(r)

        avg_rank_agreement = float(np.mean(spearman_vals)) if spearman_vals else None

        # 3. Label correlation: how well does the average score correlate with labels
        avg_scores = np.mean(score_matrix, axis=0)
        if len(set(avg_scores)) >= 2 and len(set(labels)) >= 2:
            label_corr, _ = spearmanr(avg_scores, labels)
            label_corr = float(label_corr) if not math.isnan(label_corr) else None
        else:
            label_corr = None

        # 4. Rubric-level label correlations
        rubric_label_corrs = []
        for ri in range(len(rubrics)):
            if len(set(score_matrix[ri])) >= 2 and len(set(labels)) >= 2:
                r, _ = spearmanr(score_matrix[ri], labels)
                if not math.isnan(r):
                    rubric_label_corrs.append(r)

        # 5. Collapse rate: how many rubrics produce only 1 unique score
        collapsed_rubrics = sum(1 for ri in range(len(rubrics)) if len(set(score_matrix[ri])) <= 1)

        results.append({
            "prompt_id": prompt_id,
            "num_rubrics": len(rubrics),
            "num_completions": len(completions),
            "avg_score_variance": avg_score_var,
            "avg_rank_agreement": avg_rank_agreement,
            "ensemble_label_corr": label_corr,
            "avg_individual_label_corr": float(np.mean(rubric_label_corrs)) if rubric_label_corrs else None,
            "collapsed_rubrics": collapsed_rubrics,
            "rubric_lengths": [len(r) for r in rubrics],
        })

    # Aggregate
    valid_vars = [r["avg_score_variance"] for r in results]
    valid_agreements = [r["avg_rank_agreement"] for r in results if r["avg_rank_agreement"] is not None]
    valid_ens_corrs = [r["ensemble_label_corr"] for r in results if r["ensemble_label_corr"] is not None]
    valid_ind_corrs = [r["avg_individual_label_corr"] for r in results if r["avg_individual_label_corr"] is not None]
    total_collapsed = sum(r["collapsed_rubrics"] for r in results)
    total_rubrics = sum(r["num_rubrics"] for r in results)
    avg_rubric_len = float(np.mean([l for r in results for l in r["rubric_lengths"]]))

    summary = {
        "benchmark": "consensus",
        "num_prompts": len(results),
        "num_rubrics_per_prompt": num_rubrics,
        "avg_score_variance": float(np.mean(valid_vars)),
        "avg_rank_agreement": float(np.mean(valid_agreements)) if valid_agreements else None,
        "n_prompts_rank_agreement": len(valid_agreements),
        "ensemble_label_corr": float(np.mean(valid_ens_corrs)) if valid_ens_corrs else None,
        "avg_individual_label_corr": float(np.mean(valid_ind_corrs)) if valid_ind_corrs else None,
        "collapse_rate": total_collapsed / total_rubrics if total_rubrics > 0 else 0,
        "avg_rubric_length": avg_rubric_len,
    }

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Cross-rubric consensus benchmark")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="out/bench/consensus")
    parser.add_argument("--max_prompts", type=int, default=30)
    parser.add_argument("--num_rubrics", type=int, default=5)
    parser.add_argument("--max_completions", type=int, default=4)
    parser.add_argument("--eval_workers", type=int, default=8)
    args = parser.parse_args()

    config = ProjectConfig()
    grm = RubricGenerator(args.model_path)
    judge = Judge(
        model_name=config.oracle_model_name,
        api_key=config.oracle_api_key,
        api_base=config.oracle_api_base,
    )

    summary, per_prompt = run_consensus_benchmark(
        grm=grm,
        judge=judge,
        max_prompts=args.max_prompts,
        num_rubrics=args.num_rubrics,
        max_completions=args.max_completions,
        eval_workers=args.eval_workers,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "consensus_results.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "per_prompt": per_prompt}, f, indent=2)

    print("\n=== Consensus Benchmark Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
