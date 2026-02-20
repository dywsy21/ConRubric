import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from scipy.stats import spearmanr

from dotenv import load_dotenv

load_dotenv()

# set HF cache env early
hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")

from tqdm import tqdm

from src.config import ProjectConfig
from src.data.prepare_healthbench import ensure_healthbench_splits
from src.evaluation.judge import Judge
from src.models.grm import RubricGenerator


def _label_score(binary_labels: List[bool]) -> float:
    if not binary_labels:
        return 0.0
    return float(sum(1 for x in binary_labels if x) / len(binary_labels))


def _pairwise_metrics(scores: List[float], labels: List[float]) -> Tuple[float, float, float]:
    """
    Returns (pairwise_acc, resolution, n_pairs_used)
    pairwise_acc: ordering agreement on all unequal-label pairs
    resolution: average |score_i - score_j| on unequal-label pairs
    """
    correct = 0.0
    total = 0
    gap_sum = 0.0

    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if labels[i] == labels[j]:
                continue
            total += 1
            gap_sum += abs(scores[i] - scores[j])
            label_order = labels[i] > labels[j]
            score_order = scores[i] > scores[j]
            if scores[i] == scores[j]:
                correct += 0.5
            elif label_order == score_order:
                correct += 1.0

    if total == 0:
        return 0.0, 0.0, 0.0
    return correct / total, gap_sum / total, float(total)


def _discrimination_metrics(
    per_prompt_data: List[Tuple[List[float], List[float]]],
) -> Dict[str, float]:
    """Rubric discrimination quality: does the rubric make the Judge rank
    completions in agreement with ground-truth labels?

    Returns three complementary metrics, all computed *within* each prompt
    (so cross-rubric scale differences cancel out) then averaged:

    1. **avg_spearman**: Mean Spearman rank-correlation between predicted
       scores and label scores across prompts.  Ranges [-1, 1]; higher is
       better.  Only computed for prompts with ≥ 2 distinct label values
       (otherwise ranking is undefined).

    2. **avg_top_bottom_gap**: For each prompt, computes
       ``mean(pred | label == best) − mean(pred | label == worst)``.
       Positive means the rubric correctly pushes up better completions.
       Only requires that the prompt has at least two different label
       values (always the case in practice), so it never suffers from
       empty-bucket problems.

    3. **avg_score_label_corr**: Pearson-like dot-product correlation
       (via Spearman) for sanity-checking.
    """
    spearman_vals: List[float] = []
    top_bottom_gaps: List[float] = []

    for preds, labels in per_prompt_data:
        if len(preds) < 2:
            continue

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            # All completions share the same label → ranking is undefined.
            continue

        # --- Spearman ---
        rho, _ = spearmanr(preds, labels)
        if not math.isnan(rho):
            spearman_vals.append(rho)

        # --- Top-bottom gap ---
        best_label = max(unique_labels)
        worst_label = min(unique_labels)
        top_preds = [s for s, y in zip(preds, labels) if y == best_label]
        bot_preds = [s for s, y in zip(preds, labels) if y == worst_label]
        if top_preds and bot_preds:
            gap = (sum(top_preds) / len(top_preds)) - (sum(bot_preds) / len(bot_preds))
            top_bottom_gaps.append(gap)

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "avg_spearman": _mean(spearman_vals),
        "avg_top_bottom_gap": _mean(top_bottom_gaps),
        "n_prompts_spearman": len(spearman_vals),
        "n_prompts_top_bottom": len(top_bottom_gaps),
    }


def run_healthbench_rubric_quality(
    grm: RubricGenerator,
    judge: Judge,
    output_dir: str,
    benchmark_ratio: float = 0.2,
    split_seed: int = 42,
    max_prompts: int = None,
    max_completions_per_prompt: int = 8,
    eval_workers: int = None,
):
    if eval_workers is None:
        eval_workers = int(os.environ.get("GRM_ORACLE_WORKERS", "8"))

    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=benchmark_ratio,
        seed=split_seed,
        force_rebuild=False,
    )

    # leakage check
    sft_prompt_ids = set()
    with open(split_paths.sft_train, "r", encoding="utf-8") as f:
        for line in f:
            sft_prompt_ids.add(json.loads(line).get("prompt_id"))

    grouped = defaultdict(list)
    with open(split_paths.benchmark_meta_eval, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("prompt_id")
            if pid in sft_prompt_ids:
                raise RuntimeError(f"Leakage detected: prompt_id {pid} present in both SFT and benchmark")
            grouped[pid].append(row)

    prompt_items = list(grouped.items())
    if max_prompts is not None:
        prompt_items = prompt_items[:max_prompts]

    print(f"Benchmark prompts: {len(prompt_items)} (meta_eval rows: {sum(len(v) for _, v in prompt_items)})")

    per_prompt_results = []
    per_prompt_pred_labels: List[Tuple[List[float], List[float]]] = []

    for prompt_id, rows in tqdm(prompt_items, desc="HealthBench-RubricQuality"):
        prompt = rows[0]["prompt"]
        if isinstance(prompt, list):
            # keep only user-content text for GRM question
            parts = []
            for m in prompt:
                if isinstance(m, dict):
                    c = (m.get("content") or "").strip()
                    if c:
                        parts.append(c)
            prompt_text = "\n".join(parts)
        else:
            prompt_text = str(prompt)

        rubric = grm.generate_rubric(prompt_text)

        # pick capped number of completions per prompt
        rows = rows[:max_completions_per_prompt]
        pred_scores = []
        label_scores = []

        questions_batch = []
        completions_batch = []
        rubrics_batch = []

        for r in rows:
            completion = r.get("completion", "")
            label = _label_score(r.get("binary_labels", []))
            questions_batch.append(prompt_text)
            completions_batch.append(completion)
            rubrics_batch.append(rubric)
            label_scores.append(float(label))

        pred_scores = [
            float(x)
            for x in judge.evaluate_batch(
                questions=questions_batch,
                answers=completions_batch,
                rubrics=rubrics_batch,
                show_progress=False,
                max_workers=eval_workers,
            )
        ]
        per_prompt_pred_labels.append((pred_scores, label_scores))

        pair_acc, resolution, n_pairs = _pairwise_metrics(pred_scores, label_scores)

        per_prompt_results.append(
            {
                "prompt_id": prompt_id,
                "category": rows[0].get("category", "unknown"),
                "num_completions": len(rows),
                "pairwise_acc": pair_acc,
                "resolution": resolution,
                "pairs_used": n_pairs,
                "pred_scores": pred_scores,
                "label_scores": label_scores,
            }
        )

    avg_pair_acc = sum(x["pairwise_acc"] for x in per_prompt_results) / len(per_prompt_results) if per_prompt_results else 0.0
    avg_resolution = sum(x["resolution"] for x in per_prompt_results) / len(per_prompt_results) if per_prompt_results else 0.0
    discrimination = _discrimination_metrics(per_prompt_pred_labels)

    summary = {
        "benchmark": "healthbench_rubric_quality",
        "leakage_check": "passed",
        "num_prompts": len(per_prompt_results),
        "avg_pairwise_acc": avg_pair_acc,
        "avg_resolution": avg_resolution,
        "discrimination": discrimination,
        "split": {
            "benchmark_ratio": benchmark_ratio,
            "split_seed": split_seed,
            "sft_train_file": split_paths.sft_train,
            "benchmark_meta_eval_file": split_paths.benchmark_meta_eval,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "healthbench_rubric_quality.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_prompt": per_prompt_results}, f, ensure_ascii=False, indent=2)

    print("\n=== HealthBench Rubric Quality Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Fine-grained rubric quality benchmark for GRM on HealthBench")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="out/bench")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--max_completions_per_prompt", type=int, default=8)
    parser.add_argument("--eval_workers", type=int, default=int(os.environ.get("GRM_ORACLE_WORKERS", "8")))
    parser.add_argument("--healthbench_benchmark_ratio", type=float, default=0.2)
    parser.add_argument("--healthbench_split_seed", type=int, default=42)
    args = parser.parse_args()

    config = ProjectConfig()
    grm = RubricGenerator(args.model_path)
    judge = Judge(
        model_name=config.oracle_model_name,
        api_key=config.oracle_api_key,
        api_base=config.oracle_api_base,
    )

    run_healthbench_rubric_quality(
        grm=grm,
        judge=judge,
        output_dir=args.output_dir,
        benchmark_ratio=args.healthbench_benchmark_ratio,
        split_seed=args.healthbench_split_seed,
        max_prompts=args.max_prompts,
        max_completions_per_prompt=args.max_completions_per_prompt,
        eval_workers=args.eval_workers,
    )


if __name__ == "__main__":
    main()
