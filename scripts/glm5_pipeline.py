"""GLM-5 benchmark with pipelined rubric generation + Judge scoring.

Rubric generation (4 API workers) runs in parallel with Judge scoring.
As each rubric completes, it's immediately queued for Judge evaluation.
"""
import json
import math
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List, Tuple

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
from src.models.api_grm import APIRubricGenerator


def _label_score(binary_labels):
    if not binary_labels:
        return 0.0
    return float(sum(1 for x in binary_labels if x) / len(binary_labels))


def judge_worker(judge, task_queue, results_list, results_lock, prompt_texts,
                 eval_workers, progress_bar):
    """Background thread scoring rubrics as they arrive."""
    while True:
        item = task_queue.get()
        if item is None:
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

    print(f"\n=== GLM-5 HealthBench Rubric Quality Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_file}")
    return summary


def main():
    output_dir = "out/bench_full/glm5"
    rubric_cache_path = os.path.join(output_dir, "rubric_cache.json")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=0.2, seed=42, force_rebuild=False,
    )

    sft_prompt_ids = set()
    with open(split_paths.sft_train) as f:
        for line in f:
            sft_prompt_ids.add(json.loads(line).get("prompt_id"))

    grouped = defaultdict(list)
    with open(split_paths.benchmark_meta_eval) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("prompt_id")
            if pid in sft_prompt_ids:
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

    pid_rows = {pid: rows for pid, rows in prompt_items}
    print(f"Total prompts: {len(prompt_items)}")

    # Load cached rubrics
    rubrics = {}
    if os.path.exists(rubric_cache_path):
        with open(rubric_cache_path) as f:
            rubrics = json.load(f)
        n_cached = sum(1 for v in rubrics.values() if v)
        print(f"Loaded {n_cached} cached rubrics")

    # Setup Judge thread
    config = ProjectConfig()
    judge = Judge(
        model_name=config.oracle_model_name,
        api_key=config.oracle_api_key,
        api_base=config.oracle_api_base,
    )

    task_queue = Queue()
    results_list = []
    results_lock = threading.Lock()
    score_pbar = tqdm(total=len(prompt_items), desc="Judge scoring", position=1)

    judge_thread = threading.Thread(
        target=judge_worker,
        args=(judge, task_queue, results_list, results_lock,
              prompt_texts, 4, score_pbar),
        daemon=True,
    )
    judge_thread.start()

    # Queue cached rubrics first
    n_cached_queued = 0
    for pid, rows in prompt_items:
        cached = rubrics.get(pid, "")
        if cached and cached.strip():
            task_queue.put((pid, cached, rows))
            n_cached_queued += 1
    if n_cached_queued:
        print(f"Queued {n_cached_queued} cached rubrics for scoring")

    # Generate missing rubrics via API
    missing = [(pid, prompt_texts[pid]) for pid, _ in prompt_items
               if pid not in rubrics or not rubrics[pid]]

    if missing:
        n_api_workers = 2  # keep low to avoid 429 rate limits
        print(f"Generating {len(missing)} rubrics via GLM-5 API ({n_api_workers} workers)...")
        grm = APIRubricGenerator(
            api_base="https://open.bigmodel.cn/api/paas/v4",
            api_key="e7556592ec324ef7b7bb65ddac18b108.2fAWr5h24Jy2cpuN",
            model="glm-5",
            max_workers=n_api_workers,
            max_tokens=4096,
            thinking=True,
        )

        gen_pbar = tqdm(total=len(missing), desc="GLM-5 rubrics", position=0)
        with ThreadPoolExecutor(max_workers=n_api_workers) as pool:
            futures = {}
            for pid, text in missing:
                fut = pool.submit(grm.generate_rubric, text)
                futures[fut] = pid

            done = 0
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    rubric = fut.result()
                    rubrics[pid] = rubric
                    if rubric and rubric.strip():
                        task_queue.put((pid, rubric, pid_rows[pid]))
                except Exception as e:
                    print(f"Error for {pid}: {e}")
                    rubrics[pid] = ""

                done += 1
                gen_pbar.update(1)

                # Incremental save every 25
                if done % 25 == 0:
                    with open(rubric_cache_path, "w") as f:
                        json.dump(rubrics, f, ensure_ascii=False)
                    n_ok = sum(1 for v in rubrics.values() if v)
                    gen_pbar.set_postfix(cached=n_ok, judge_q=task_queue.qsize())

        gen_pbar.close()

        # Final cache save
        with open(rubric_cache_path, "w") as f:
            json.dump(rubrics, f, ensure_ascii=False)
        n_ok = sum(1 for v in rubrics.values() if v)
        print(f"Saved {n_ok} rubrics to {rubric_cache_path}")
        print("Waiting for Judge scoring to finish...")
    else:
        print("All rubrics cached.")

    # Update score progress bar total
    actual_scored = n_cached_queued + sum(
        1 for pid, _ in missing if rubrics.get(pid, "").strip()
    )
    score_pbar.total = actual_scored
    score_pbar.refresh()

    # Wait for Judge
    task_queue.put(None)
    judge_thread.join()
    score_pbar.close()

    compute_summary(results_list, output_dir)


if __name__ == "__main__":
    main()
