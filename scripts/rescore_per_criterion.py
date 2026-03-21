"""Re-score existing rubric caches using per-criterion judge mode.

Uses cached rubrics from out/bench_full/{model}/rubric_cache.json and
re-evaluates all HealthBench completions with JUDGE_MODE=per_criterion.

Usage:
    # Force per-criterion mode
    JUDGE_MODE=per_criterion python scripts/rescore_per_criterion.py --models base_qwen3_8b rl_step_960 glm5
    
    # Or set in .env: JUDGE_MODE=per_criterion
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from scipy.stats import spearmanr
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

# Ensure per-criterion mode
if os.environ.get("JUDGE_MODE", "holistic") != "per_criterion":
    print("WARNING: JUDGE_MODE not set to 'per_criterion', forcing it now")
    os.environ["JUDGE_MODE"] = "per_criterion"

hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")

from src.config import ProjectConfig
from src.data.prepare_healthbench import ensure_healthbench_splits
from src.evaluation.judge import Judge, JUDGE_MODE


def _label_score(binary_labels):
    if not binary_labels:
        return 0.0
    return float(sum(1 for x in binary_labels if x) / len(binary_labels))


def load_data():
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=0.2, seed=42, force_rebuild=False,
    )
    sft_pids = set()
    with open(split_paths.sft_train) as f:
        for line in f:
            sft_pids.add(json.loads(line).get("prompt_id"))

    grouped = defaultdict(list)
    with open(split_paths.benchmark_meta_eval) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("prompt_id")
            if pid in sft_pids:
                raise RuntimeError(f"Leakage: {pid}")
            grouped[pid].append(row)

    prompt_items = list(grouped.items())
    prompt_texts = {}
    for pid, rows in prompt_items:
        prompt = rows[0]["prompt"]
        if isinstance(prompt, list):
            parts = []
            for m in prompt:
                if isinstance(m, dict):
                    c = (m.get("content") or "").strip()
                    if c:
                        parts.append(c)
            prompt_texts[pid] = "\n".join(parts)
        else:
            prompt_texts[pid] = str(prompt)
    return prompt_items, prompt_texts


def score_model(model_name: str, judge: Judge, prompt_items, prompt_texts, eval_workers: int):
    """Score all prompts for a model's cached rubrics using per-criterion mode."""
    cache_file = f"out/bench_full/{model_name}/rubric_cache.json"
    if not os.path.exists(cache_file):
        print(f"ERROR: {cache_file} not found, skipping {model_name}")
        return None

    with open(cache_file) as f:
        rubrics = json.load(f)
    n_valid = sum(1 for v in rubrics.values() if v and v.strip())
    print(f"\n{'='*60}")
    print(f"Scoring {model_name}: {n_valid} rubrics, mode={JUDGE_MODE}")
    print(f"{'='*60}")

    results_list = []
    per_prompt_pred_labels = []

    # Build tasks: (pid, prompt_text, rubric, rows)
    tasks = []
    for pid, rows in prompt_items:
        rubric = rubrics.get(pid, "")
        if not rubric or not rubric.strip():
            continue
        tasks.append((pid, prompt_texts[pid], rubric, rows[:8]))

    print(f"Scoring {len(tasks)} prompts with {eval_workers} workers...")

    def score_one_prompt(task):
        pid, prompt_text, rubric, rows = task
        label_scores = [_label_score(r.get("binary_labels", [])) for r in rows]
        pred_scores = []

        for r in rows:
            completion = r.get("completion", "")
            score = judge.evaluate_answer(prompt_text, completion, rubric=rubric)
            pred_scores.append(float(score))

        # Pairwise metrics
        correct = 0.0
        total_pairs = 0
        gap_sum = 0.0
        for i in range(len(pred_scores)):
            for j in range(i + 1, len(pred_scores)):
                if label_scores[i] == label_scores[j]:
                    continue
                total_pairs += 1
                gap_sum += abs(pred_scores[i] - pred_scores[j])
                label_order = label_scores[i] > label_scores[j]
                score_order = pred_scores[i] > pred_scores[j]
                if pred_scores[i] == pred_scores[j]:
                    correct += 0.5
                elif label_order == score_order:
                    correct += 1.0

        pair_acc = correct / total_pairs if total_pairs > 0 else 0.0
        resolution = gap_sum / total_pairs if total_pairs > 0 else 0.0

        return {
            "prompt_id": pid,
            "category": rows[0].get("category", "unknown"),
            "num_completions": len(rows),
            "pairwise_acc": pair_acc,
            "resolution": resolution,
            "pairs_used": float(total_pairs),
            "pred_scores": pred_scores,
            "label_scores": label_scores,
        }

    # Parallel scoring at the prompt level (each prompt's completions scored sequentially
    # within the thread, but multiple prompts scored in parallel)
    with ThreadPoolExecutor(max_workers=eval_workers) as executor:
        futures = {executor.submit(score_one_prompt, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {model_name}"):
            result = future.result()
            results_list.append(result)
            per_prompt_pred_labels.append((result["pred_scores"], result["label_scores"]))

    # Compute aggregate metrics
    spearman_vals = []
    top_bottom_gaps = []
    for preds, labels in per_prompt_pred_labels:
        if len(preds) < 2:
            continue
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            continue
        rho, _ = spearmanr(preds, labels)
        if not math.isnan(rho):
            spearman_vals.append(rho)
        best_label = max(unique_labels)
        worst_label = min(unique_labels)
        top_preds = [s for s, y in zip(preds, labels) if y == best_label]
        bot_preds = [s for s, y in zip(preds, labels) if y == worst_label]
        if top_preds and bot_preds:
            gap = (sum(top_preds) / len(top_preds)) - (sum(bot_preds) / len(bot_preds))
            top_bottom_gaps.append(gap)

    n = len(results_list)
    avg_pa = sum(r["pairwise_acc"] for r in results_list) / n if n else 0
    avg_res = sum(r["resolution"] for r in results_list) / n if n else 0
    avg_sp = sum(spearman_vals) / len(spearman_vals) if spearman_vals else 0
    avg_tb = sum(top_bottom_gaps) / len(top_bottom_gaps) if top_bottom_gaps else 0

    summary = {
        "benchmark": "healthbench_rubric_quality",
        "judge_mode": "per_criterion",
        "leakage_check": "passed",
        "num_prompts": n,
        "avg_pairwise_acc": avg_pa,
        "avg_resolution": avg_res,
        "discrimination": {
            "avg_spearman": avg_sp,
            "avg_top_bottom_gap": avg_tb,
            "n_prompts_spearman": len(spearman_vals),
            "n_prompts_top_bottom": len(top_bottom_gaps),
        },
    }

    output_dir = f"out/bench_full/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "healthbench_rubric_quality_per_criterion.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "per_prompt": results_list}, f, ensure_ascii=False, indent=2)

    print(f"\n  {model_name} per-criterion results:")
    print(f"    prompts={n}  pairwise={avg_pa:.4f}  spearman={avg_sp:.4f}  "
          f"top_bottom={avg_tb:.4f}  resolution={avg_res:.4f}")
    print(f"    Saved: {out_file}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["base_qwen3_8b", "rl_step_960", "glm5"])
    parser.add_argument("--eval_workers", type=int, default=2,
                        help="Number of parallel prompt-level workers")
    args = parser.parse_args()

    config = ProjectConfig()
    judge = Judge(
        model_name=config.oracle_model_name,
        api_key=config.oracle_api_key,
        api_base=config.oracle_api_base,
    )

    prompt_items, prompt_texts = load_data()
    print(f"Loaded {len(prompt_items)} benchmark prompts")
    print(f"Judge mode: {JUDGE_MODE}")
    print(f"Judge model: {config.oracle_model_name}")

    all_summaries = {}
    for model_name in args.models:
        summary = score_model(model_name, judge, prompt_items, prompt_texts, args.eval_workers)
        if summary:
            all_summaries[model_name] = summary

    # Print comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*70}")
        print("Per-Criterion Mode Comparison")
        print(f"{'='*70}")
        header = f"{'Model':>20s} | {'Pairwise':>9s} | {'Spearman':>9s} | {'TopBot':>9s} | {'Resolution':>10s} | {'N':>4s}"
        print(header)
        print("-" * len(header))
        for name, s in all_summaries.items():
            print(f"{name:>20s} | {s['avg_pairwise_acc']:>9.4f} | "
                  f"{s['discrimination']['avg_spearman']:>9.4f} | "
                  f"{s['discrimination']['avg_top_bottom_gap']:>9.4f} | "
                  f"{s['avg_resolution']:>10.4f} | "
                  f"{s['num_prompts']:>4d}")


if __name__ == "__main__":
    main()
