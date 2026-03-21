"""Re-score existing rubric caches using per-criterion judge mode.

Uses cached rubrics from out/bench_full/{model}/rubric_cache.json and
re-evaluates HealthBench discriminative prompts (label variance > 0.01)
with JUDGE_MODE=per_criterion.

Usage:
    JUDGE_MODE=per_criterion python scripts/rescore_per_criterion.py --models base_qwen3_8b rl_step_960 glm5
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, ndcg_score
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
    """Load HealthBench data and filter to discriminative prompts (label var > 0.01)."""
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

    # Filter to discriminative prompts: label variance > 0.01
    all_items = list(grouped.items())
    prompt_items = []
    for pid, rows in all_items:
        label_scores = [_label_score(r.get("binary_labels", [])) for r in rows[:8]]
        if len(label_scores) >= 2:
            var = statistics.variance(label_scores)
        else:
            var = 0
        if var > 0.01:
            prompt_items.append((pid, rows))

    print(f"Total prompts: {len(all_items)}, discriminative (var>0.01): {len(prompt_items)}")

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
    """Score discriminative prompts for a model's cached rubrics using per-criterion mode."""
    cache_file = f"out/bench_full/{model_name}/rubric_cache.json"
    if not os.path.exists(cache_file):
        print(f"ERROR: {cache_file} not found, skipping {model_name}")
        return None

    with open(cache_file) as f:
        rubrics = json.load(f)
    n_valid = sum(1 for v in rubrics.values() if v and v.strip())
    print(f"\n{'='*60}")
    print(f"Scoring {model_name}: {n_valid} cached rubrics, mode={JUDGE_MODE}")
    print(f"{'='*60}")

    # Check for existing partial results
    output_dir = f"out/bench_full/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "healthbench_rubric_quality_per_criterion.json")
    done_pids = set()
    results_list = []
    if os.path.exists(out_file):
        try:
            existing = json.load(open(out_file))
            for r in existing.get("per_prompt", []):
                done_pids.add(r["prompt_id"])
                results_list.append(r)
            print(f"  Resuming: {len(done_pids)} prompts already scored")
        except Exception:
            pass

    # Build tasks: only discriminative prompts with valid rubrics that aren't done
    tasks = []
    skipped_no_rubric = 0
    for pid, rows in prompt_items:
        if pid in done_pids:
            continue
        rubric = rubrics.get(pid, "")
        if not rubric or not rubric.strip():
            skipped_no_rubric += 1
            continue
        tasks.append((pid, prompt_texts[pid], rubric, rows[:8]))

    if skipped_no_rubric:
        print(f"  Skipped {skipped_no_rubric} prompts with no rubric")
    print(f"  Scoring {len(tasks)} remaining prompts with {eval_workers} workers...")

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

    # Score with incremental saves
    save_interval = 25
    completed_in_batch = 0
    with ThreadPoolExecutor(max_workers=eval_workers) as executor:
        futures = {executor.submit(score_one_prompt, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {model_name}"):
            result = future.result()
            results_list.append(result)
            completed_in_batch += 1

            # Incremental save
            if completed_in_batch % save_interval == 0:
                _save_results(results_list, output_dir, out_file)

    # Final save with full metrics
    summary = _compute_full_metrics(results_list)
    summary["judge_mode"] = "per_criterion"

    with open(out_file, "w") as f:
        json.dump({"summary": summary, "per_prompt": results_list}, f, ensure_ascii=False, indent=2)

    print(f"\n  {model_name} per-criterion results ({summary['num_prompts']} discriminative prompts):")
    print(f"    Pairwise={summary['avg_pairwise_acc']:.4f}  "
          f"Kendall={summary['kendall_tau']:.4f}  "
          f"Spearman={summary['discrimination']['avg_spearman']:.4f}")
    print(f"    NDCG@4={summary['ndcg_at_4']:.4f}  "
          f"AUC-ROC={summary['auc_roc']:.4f}  "
          f"ConfPairAcc={summary['confident_pair_acc']:.4f}")
    print(f"    Sensitivity={summary['score_sensitivity']:.1%}  "
          f"Resolution={summary['avg_resolution']:.4f}")
    print(f"    Saved: {out_file}")
    return summary


def _save_results(results_list, output_dir, out_file):
    """Incremental save of partial results."""
    partial = {"summary": {"partial": True, "num_prompts": len(results_list)},
               "per_prompt": results_list}
    with open(out_file, "w") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)


def _compute_full_metrics(results_list):
    """Compute all benchmark metrics from per-prompt results."""
    n = len(results_list)
    if n == 0:
        return {"num_prompts": 0}

    # Aggregate pairwise and resolution
    avg_pa = sum(r["pairwise_acc"] for r in results_list) / n
    avg_res = sum(r["resolution"] for r in results_list) / n

    # Per-prompt Spearman, Kendall, top-bottom gap
    spearman_vals = []
    kendall_vals = []
    top_bottom_gaps = []
    ndcg_vals = []
    auc_vals = []
    confident_pair_accs = []
    n_sensitive = 0

    for r in results_list:
        preds = r["pred_scores"]
        labels = r["label_scores"]

        # Score sensitivity: at least 2 distinct integer scores
        int_scores = set(int(s) for s in preds)
        if len(int_scores) >= 2:
            n_sensitive += 1

        unique_labels = set(labels)
        if len(unique_labels) < 2 or len(preds) < 2:
            continue

        # Spearman
        rho, _ = spearmanr(preds, labels)
        if not math.isnan(rho):
            spearman_vals.append(rho)

        # Kendall tau-b
        tau, _ = kendalltau(preds, labels)
        if not math.isnan(tau):
            kendall_vals.append(tau)

        # Top-bottom gap
        best_label = max(unique_labels)
        worst_label = min(unique_labels)
        top_preds = [s for s, y in zip(preds, labels) if y == best_label]
        bot_preds = [s for s, y in zip(preds, labels) if y == worst_label]
        if top_preds and bot_preds:
            gap = (sum(top_preds) / len(top_preds)) - (sum(bot_preds) / len(bot_preds))
            top_bottom_gaps.append(gap)

        # NDCG@4
        try:
            ndcg = ndcg_score([labels], [preds], k=4)
            ndcg_vals.append(ndcg)
        except Exception:
            pass

        # AUC-ROC (binary: label > 0.5 as positive)
        try:
            binary_labels = [1 if l > 0.5 else 0 for l in labels]
            if len(set(binary_labels)) >= 2:
                auc = roc_auc_score(binary_labels, preds)
                auc_vals.append(auc)
        except Exception:
            pass

        # Confident pair accuracy (label gap > 0.2)
        conf_correct = 0.0
        conf_total = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if abs(labels[i] - labels[j]) <= 0.2:
                    continue
                conf_total += 1
                label_order = labels[i] > labels[j]
                score_order = preds[i] > preds[j]
                if preds[i] == preds[j]:
                    conf_correct += 0.5
                elif label_order == score_order:
                    conf_correct += 1.0
        if conf_total > 0:
            confident_pair_accs.append(conf_correct / conf_total)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "benchmark": "healthbench_rubric_quality",
        "num_prompts": n,
        "avg_pairwise_acc": avg_pa,
        "avg_resolution": avg_res,
        "kendall_tau": _mean(kendall_vals),
        "ndcg_at_4": _mean(ndcg_vals),
        "auc_roc": _mean(auc_vals),
        "confident_pair_acc": _mean(confident_pair_accs),
        "score_sensitivity": n_sensitive / n if n else 0,
        "discrimination": {
            "avg_spearman": _mean(spearman_vals),
            "avg_top_bottom_gap": _mean(top_bottom_gaps),
            "n_prompts_spearman": len(spearman_vals),
            "n_prompts_top_bottom": len(top_bottom_gaps),
        },
    }


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
        print(f"\n{'='*80}")
        print("Per-Criterion Mode Comparison (discriminative prompts only)")
        print(f"{'='*80}")
        header = (f"{'Model':>15s} | {'PW Acc':>7s} | {'Kendall':>8s} | {'Spearman':>8s} | "
                  f"{'NDCG@4':>7s} | {'AUC':>6s} | {'ConfPW':>7s} | {'Sensit':>7s} | {'N':>4s}")
        print(header)
        print("-" * len(header))
        for name, s in all_summaries.items():
            print(f"{name:>15s} | {s['avg_pairwise_acc']:>7.4f} | "
                  f"{s['kendall_tau']:>8.4f} | "
                  f"{s['discrimination']['avg_spearman']:>8.4f} | "
                  f"{s['ndcg_at_4']:>7.4f} | "
                  f"{s['auc_roc']:>6.4f} | "
                  f"{s['confident_pair_acc']:>7.4f} | "
                  f"{s['score_sensitivity']:>6.1%} | "
                  f"{s['num_prompts']:>4d}")


if __name__ == "__main__":
    main()
