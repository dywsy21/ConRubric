"""Full HealthBench rubric quality benchmark with pipelined generation + scoring.

Rubric generation and Judge scoring run in parallel:
- Main thread generates rubrics in chunks, saves incrementally to rubric_cache.json
- Background thread picks up completed rubrics and scores them via Judge API

Usage:
    python /tmp/bench_pipeline.py --model_name base_qwen3_8b --model_path models/Qwen3-8B
    python /tmp/bench_pipeline.py --model_name rl_step_960 --model_path out/rl/global_step_960/actor/huggingface
    python /tmp/bench_pipeline.py --model_name base_qwen3_8b --skip_generation  # reuse cached rubrics
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Tuple

from scipy.stats import spearmanr
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")

from src.config import ProjectConfig
from src.data.prepare_healthbench import ensure_healthbench_splits
from src.evaluation.judge import Judge
from src.models.grm import RubricGenerator


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


def judge_worker(
    judge: Judge,
    task_queue: Queue,
    results_list: list,
    results_lock: threading.Lock,
    prompt_texts: dict,
    eval_workers: int,
    progress_bar: tqdm,
):
    """Background thread that scores rubrics as they arrive on the queue."""
    while True:
        item = task_queue.get()
        if item is None:  # poison pill
            task_queue.task_done()
            break

        pid, rubric, rows = item
        rows = rows[:8]
        prompt_text = prompt_texts[pid]

        questions_batch = [prompt_text] * len(rows)
        completions_batch = [r.get("completion", "") for r in rows]
        rubrics_batch = [rubric] * len(rows)
        label_scores = [_label_score(r.get("binary_labels", [])) for r in rows]

        try:
            pred_scores = [
                float(x) for x in judge.evaluate_batch(
                    questions=questions_batch,
                    answers=completions_batch,
                    rubrics=rubrics_batch,
                    show_progress=False,
                    max_workers=eval_workers,
                )
            ]
        except Exception as e:
            print(f"Judge error for {pid}: {e}")
            task_queue.task_done()
            progress_bar.update(1)
            continue

        # Compute per-prompt metrics
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

        result = {
            "prompt_id": pid,
            "category": rows[0].get("category", "unknown"),
            "num_completions": len(rows),
            "pairwise_acc": pair_acc,
            "resolution": resolution,
            "pairs_used": float(total_pairs),
            "pred_scores": pred_scores,
            "label_scores": label_scores,
        }

        with results_lock:
            results_list.append((result, pred_scores, label_scores))

        task_queue.task_done()
        progress_bar.update(1)


def compute_summary(results_list, output_dir):
    """Compute aggregate metrics and save."""
    per_prompt_results = [r[0] for r in results_list]
    per_prompt_pred_labels = [(r[1], r[2]) for r in results_list]

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

    n = len(per_prompt_results)
    avg_pa = sum(r["pairwise_acc"] for r in per_prompt_results) / n if n else 0
    avg_res = sum(r["resolution"] for r in per_prompt_results) / n if n else 0
    avg_sp = sum(spearman_vals) / len(spearman_vals) if spearman_vals else 0
    avg_tb = sum(top_bottom_gaps) / len(top_bottom_gaps) if top_bottom_gaps else 0

    summary = {
        "benchmark": "healthbench_rubric_quality",
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

    out_file = os.path.join(output_dir, "healthbench_rubric_quality.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "per_prompt": per_prompt_results}, f, ensure_ascii=False, indent=2)

    print(f"\n=== {output_dir} Summary ===")
    print(f"  prompts={n}  pairwise={avg_pa:.4f}  spearman={avg_sp:.4f}  "
          f"top_bottom={avg_tb:.4f}  resolution={avg_res:.4f}")
    print(f"  Saved: {out_file}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval_workers", type=int, default=4)
    parser.add_argument("--gen_chunk", type=int, default=50,
                        help="Generate rubrics in chunks of this size")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip rubric generation, only run Judge scoring from cache")
    args = parser.parse_args()

    output_dir = f"out/bench_full/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, "rubric_cache.json")

    prompt_items, prompt_texts = load_data()
    print(f"Loaded {len(prompt_items)} prompts")

    # Build pid -> rows lookup
    pid_rows = {pid: rows for pid, rows in prompt_items}

    # Load existing cache
    rubrics: Dict[str, str] = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            rubrics = json.load(f)
        n_cached = sum(1 for v in rubrics.values() if v)
        print(f"Loaded {n_cached}/{len(rubrics)} rubrics from cache")

    # ---- Setup Judge scoring thread ----
    config = ProjectConfig()
    judge = Judge(
        model_name=config.oracle_model_name,
        api_key=config.oracle_api_key,
        api_base=config.oracle_api_base,
    )

    task_queue: Queue = Queue()
    results_list: list = []
    results_lock = threading.Lock()

    # Count total prompts that will be scored (for progress bar)
    total_to_score = sum(1 for pid, _ in prompt_items
                         if rubrics.get(pid, "").strip() or pid not in rubrics)
    score_pbar = tqdm(total=total_to_score, desc="Judge scoring", position=1)

    judge_thread = threading.Thread(
        target=judge_worker,
        args=(judge, task_queue, results_list, results_lock,
              prompt_texts, args.eval_workers, score_pbar),
        daemon=True,
    )
    judge_thread.start()

    # ---- Queue already-cached rubrics for scoring immediately ----
    n_queued_from_cache = 0
    for pid, rows in prompt_items:
        cached = rubrics.get(pid, "")
        if cached and cached.strip():
            task_queue.put((pid, cached, rows))
            n_queued_from_cache += 1

    if n_queued_from_cache:
        print(f"Queued {n_queued_from_cache} cached rubrics for Judge scoring")

    # ---- Generate missing rubrics ----
    if not args.skip_generation:
        missing = [pid for pid, _ in prompt_items if pid not in rubrics or not rubrics[pid]]

        if missing:
            print(f"Generating {len(missing)} rubrics (model: {args.model_path}, GPU: {args.gpu})...")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            grm = RubricGenerator(args.model_path)

            # Generate in chunks, pipeline with Judge scoring
            gen_pbar = tqdm(total=len(missing), desc="Rubric generation", position=0)
            for chunk_start in range(0, len(missing), args.gen_chunk):
                chunk_pids = missing[chunk_start : chunk_start + args.gen_chunk]
                chunk_texts = [prompt_texts[pid] for pid in chunk_pids]

                chunk_rubrics = grm.generate_batch(chunk_texts)

                # Update cache and queue for scoring
                for pid, rubric in zip(chunk_pids, chunk_rubrics):
                    rubrics[pid] = rubric
                    if rubric and rubric.strip():
                        task_queue.put((pid, rubric, pid_rows[pid]))

                gen_pbar.update(len(chunk_pids))

                # Save cache incrementally
                with open(cache_file, "w") as f:
                    json.dump(rubrics, f, ensure_ascii=False, indent=2)

                n_ok = sum(1 for v in rubrics.values() if v)
                gen_pbar.set_postfix(cached=n_ok, queued=task_queue.qsize())

            gen_pbar.close()

            # Free GPU memory
            del grm
            import torch
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("GPU memory freed, waiting for Judge scoring to finish...")
        else:
            print("All rubrics cached, no generation needed.")
    else:
        print("Skipping generation (--skip_generation)")

    # Update progress bar total to actual count
    actual_total = n_queued_from_cache + sum(
        1 for pid in (missing if not args.skip_generation else [])
        if rubrics.get(pid, "").strip()
    )
    score_pbar.total = actual_total
    score_pbar.refresh()

    # ---- Wait for Judge scoring to finish ----
    task_queue.put(None)  # poison pill
    judge_thread.join()
    score_pbar.close()

    # Final cache save
    with open(cache_file, "w") as f:
        json.dump(rubrics, f, ensure_ascii=False, indent=2)
    n_ok = sum(1 for v in rubrics.values() if v)
    print(f"Final cache: {n_ok}/{len(rubrics)} rubrics in {cache_file}")

    # Compute and save summary
    compute_summary(results_list, output_dir)


if __name__ == "__main__":
    main()
