"""Benchmark frontier API models as rubric generators on HealthBench.

Each model generates rubrics for all 709 benchmark prompts via the external
API (svip.xty.app). The server's vLLM (configured in .env) scores completions
against each rubric. Results saved to out/bench/api_models/{model_name}/.

Run on server after: git pull && source .venv/bin/activate

Usage:
    python scripts/bench_api_models.py
    python scripts/bench_api_models.py --models gpt-5 deepseek-v3.1
    python scripts/bench_api_models.py --max_prompts 20   # quick smoke test
    python scripts/bench_api_models.py --skip_generation  # re-score cached rubrics
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

load_dotenv()

hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")

from src.data.prepare_healthbench import ensure_healthbench_splits
from src.evaluation.judge import Judge
from src.utils.prompts import RUBRIC_GENERATION_PROMPT

# ──────────────────────────────────────────────────────────────────────────────
# Models to benchmark
# ──────────────────────────────────────────────────────────────────────────────

MODELS = [
    "gpt-5",
    "deepseek-v3.1",
    "gemini-3.1-pro-preview",
    "claude-opus-4-7",
    "grok-4",
]

# Approximate cost per 1M tokens (input, output) in USD
# Used only for the price-score chart; not billed through our own infra.
PRICE_PER_1M = {
    "gpt-5":                 (15.0,  60.0),
    "deepseek-v3.1":         (0.27,   1.1),
    "gemini-3.1-pro-preview":(7.0,   21.0),
    "claude-opus-4-7":       (15.0,  75.0),
    "grok-4":                (3.0,   15.0),
}

# External API (rubric generation)
EXT_API_BASE = os.getenv("EXT_API_BASE", "https://svip.xty.app/v1")
EXT_API_KEY  = os.getenv("EXT_API_KEY",
                "sk-Dk3063EgG7Lezr9BJ3nkpgtGsv89KR4CKrODFLB0lgqr0E4d")

# Generation settings
GEN_MAX_TOKENS  = int(os.getenv("GEN_MAX_TOKENS", "2048"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.7"))
GEN_WORKERS     = int(os.getenv("GEN_WORKERS", "20"))  # per model
JUDGE_WORKERS   = int(os.getenv("JUDGE_WORKERS", "50"))

OUTPUT_BASE = "out/bench/api_models"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (identical to bench_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_data(max_prompts: Optional[int] = None):
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=0.2,
        seed=42,
        force_rebuild=False,
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
                raise RuntimeError(f"Data leakage: prompt_id {pid} in both SFT and benchmark")
            grouped[pid].append(row)

    prompt_items = list(grouped.items())
    if max_prompts:
        prompt_items = prompt_items[:max_prompts]

    prompt_texts: Dict[str, str] = {}
    for pid, rows in prompt_items:
        prompt = rows[0]["prompt"]
        if isinstance(prompt, list):
            parts = [
                (m.get("content") or "").strip()
                for m in prompt
                if isinstance(m, dict) and (m.get("content") or "").strip()
            ]
            prompt_texts[pid] = "\n".join(parts)
        else:
            prompt_texts[pid] = str(prompt)

    print(f"Loaded {len(prompt_items)} benchmark prompts (leakage check passed)")
    return prompt_items, prompt_texts


# ──────────────────────────────────────────────────────────────────────────────
# Rubric generation via external API
# ──────────────────────────────────────────────────────────────────────────────

def _build_ext_client(timeout: float = 120.0) -> OpenAI:
    http = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0))
    return OpenAI(api_key=EXT_API_KEY, base_url=EXT_API_BASE, http_client=http)


def _generate_one(client: OpenAI, model: str, question: str,
                  usage_in: list, usage_out: list,
                  retries: int = 3) -> str:
    """Generate a single rubric; accumulates token counts."""
    prompt = RUBRIC_GENERATION_PROMPT.format(question=question)
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=GEN_MAX_TOKENS,
                temperature=GEN_TEMPERATURE,
            )
            if resp.usage:
                usage_in.append(resp.usage.prompt_tokens)
                usage_out.append(resp.usage.completion_tokens)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  [{model}] gen error attempt {attempt+1}/{retries}: {str(e)[:80]}; retry in {wait}s")
            time.sleep(wait)
    print(f"  [{model}] generation failed after {retries} attempts: {last_err}")
    return ""


def generate_rubrics_for_model(
    model: str,
    prompt_items: List[Tuple[str, list]],
    prompt_texts: Dict[str, str],
    output_dir: str,
    skip_generation: bool = False,
    workers: int = GEN_WORKERS,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Generate rubrics for all prompts for one model, with caching & resume.

    Returns (rubrics dict pid->text, usage dict with 'input_tokens'/'output_tokens').
    """
    cache_file = os.path.join(output_dir, "rubric_cache.json")
    usage_file = os.path.join(output_dir, "usage.json")
    os.makedirs(output_dir, exist_ok=True)

    # Load existing cache
    rubrics: Dict[str, str] = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            rubrics = json.load(f)
        n_cached = sum(1 for v in rubrics.values() if v)
        print(f"  [{model}] Loaded {n_cached}/{len(rubrics)} rubrics from cache")

    if skip_generation:
        usage = _load_usage(usage_file)
        return rubrics, usage

    missing = [pid for pid, _ in prompt_items if not rubrics.get(pid, "").strip()]
    if not missing:
        print(f"  [{model}] All {len(prompt_items)} rubrics already cached, skipping generation")
        usage = _load_usage(usage_file)
        return rubrics, usage

    print(f"  [{model}] Generating {len(missing)} rubrics with {workers} workers...")

    client = _build_ext_client(timeout=120.0)
    usage_in: List[int] = []
    usage_out: List[int] = []
    lock = threading.Lock()

    def _gen_task(pid: str) -> Tuple[str, str]:
        text = _generate_one(client, model, prompt_texts[pid], usage_in, usage_out)
        return pid, text

    pbar = tqdm(total=len(missing), desc=f"  [{model}] gen", leave=False)
    batch_size = 50  # save cache every 50 completions

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_gen_task, pid): pid for pid in missing}
        completed = 0
        for future in as_completed(futures):
            pid, rubric = future.result()
            with lock:
                rubrics[pid] = rubric
                completed += 1
                if completed % batch_size == 0:
                    _save_json(rubrics, cache_file)
            pbar.update(1)

    pbar.close()
    _save_json(rubrics, cache_file)

    # Save usage
    prev_usage = _load_usage(usage_file)
    usage = {
        "input_tokens":  prev_usage.get("input_tokens", 0) + sum(usage_in),
        "output_tokens": prev_usage.get("output_tokens", 0) + sum(usage_out),
    }
    _save_json(usage, usage_file)

    n_ok = sum(1 for v in rubrics.values() if v)
    total_in  = usage["input_tokens"]
    total_out = usage["output_tokens"]
    p_in, p_out = PRICE_PER_1M.get(model, (0, 0))
    cost = total_in / 1e6 * p_in + total_out / 1e6 * p_out
    print(f"  [{model}] Generated {n_ok}/{len(prompt_items)} rubrics | "
          f"tokens: {total_in:,} in / {total_out:,} out | "
          f"estimated cost: ${cost:.4f}")
    return rubrics, usage


def _load_usage(usage_file: str) -> Dict[str, int]:
    if os.path.exists(usage_file):
        with open(usage_file) as f:
            return json.load(f)
    return {"input_tokens": 0, "output_tokens": 0}


def _save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Judge scoring
# ──────────────────────────────────────────────────────────────────────────────

def _label_score(binary_labels: List[bool]) -> float:
    if not binary_labels:
        return 0.0
    return float(sum(1 for x in binary_labels if x) / len(binary_labels))


def score_rubrics(
    judge: Judge,
    prompt_items: List[Tuple[str, list]],
    rubrics: Dict[str, str],
    prompt_texts: Dict[str, str],
    eval_workers: int = JUDGE_WORKERS,
    max_completions: int = 8,
) -> List[dict]:
    """Score all prompts using the judge. Returns per-prompt result dicts."""

    def _score_prompt(pid: str, rows: list) -> Optional[dict]:
        rubric = rubrics.get(pid, "").strip()
        if not rubric:
            return None
        rows = rows[:max_completions]
        pt = prompt_texts[pid]
        questions_b = [pt] * len(rows)
        answers_b   = [r.get("completion", "") for r in rows]
        rubrics_b   = [rubric] * len(rows)
        labels_b    = [_label_score(r.get("binary_labels", [])) for r in rows]

        try:
            preds = [
                float(x) for x in judge.evaluate_batch(
                    questions=questions_b, answers=answers_b, rubrics=rubrics_b,
                    show_progress=False, max_workers=min(8, eval_workers),
                )
            ]
        except Exception as e:
            print(f"  Judge error for {pid}: {e}")
            return None

        # Per-prompt pairwise acc
        correct = 0.0
        total_pairs = 0
        gap_sum = 0.0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if labels_b[i] == labels_b[j]:
                    continue
                total_pairs += 1
                gap_sum += abs(preds[i] - preds[j])
                if preds[i] == preds[j]:
                    correct += 0.5
                elif (labels_b[i] > labels_b[j]) == (preds[i] > preds[j]):
                    correct += 1.0

        pair_acc   = correct / total_pairs if total_pairs > 0 else 0.0
        resolution = gap_sum / total_pairs if total_pairs > 0 else 0.0

        return {
            "prompt_id":       pid,
            "category":        rows[0].get("category", "unknown"),
            "num_completions": len(rows),
            "pairwise_acc":    pair_acc,
            "resolution":      resolution,
            "pairs_used":      float(total_pairs),
            "pred_scores":     preds,
            "label_scores":    labels_b,
        }

    # Parallel scoring across prompts
    to_score = [(pid, rows) for pid, rows in prompt_items if rubrics.get(pid, "").strip()]
    results: List[Optional[dict]] = [None] * len(to_score)
    lock = threading.Lock()

    pbar = tqdm(total=len(to_score), desc="  Judge scoring", leave=False)

    with ThreadPoolExecutor(max_workers=eval_workers) as executor:
        future_to_idx = {
            executor.submit(_score_prompt, pid, rows): i
            for i, (pid, rows) in enumerate(to_score)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            pbar.update(1)

    pbar.close()
    return [r for r in results if r is not None]


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(per_prompt: List[dict]) -> dict:
    """Compute aggregate metrics from per-prompt results."""
    spearman_vals: List[float] = []
    kendall_vals:  List[float] = []
    top_bottom_gaps: List[float] = []

    for r in per_prompt:
        preds  = r["pred_scores"]
        labels = r["label_scores"]
        if len(preds) < 2:
            continue
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            continue

        rho, _ = spearmanr(preds, labels)
        if not math.isnan(rho):
            spearman_vals.append(rho)

        tau, _ = kendalltau(preds, labels, variant="b")
        if not math.isnan(tau):
            kendall_vals.append(tau)

        best_label  = max(unique_labels)
        worst_label = min(unique_labels)
        top_p = [s for s, y in zip(preds, labels) if y == best_label]
        bot_p = [s for s, y in zip(preds, labels) if y == worst_label]
        if top_p and bot_p:
            top_bottom_gaps.append(
                sum(top_p) / len(top_p) - sum(bot_p) / len(bot_p)
            )

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    n = len(per_prompt)
    return {
        "num_prompts":         n,
        "avg_pairwise_acc":    _mean([r["pairwise_acc"] for r in per_prompt]),
        "avg_resolution":      _mean([r["resolution"] for r in per_prompt]),
        "avg_spearman":        _mean(spearman_vals),
        "avg_kendall_tau_b":   _mean(kendall_vals),
        "avg_top_bottom_gap":  _mean(top_bottom_gaps),
        "n_spearman_prompts":  len(spearman_vals),
        "n_kendall_prompts":   len(kendall_vals),
        "n_top_bottom_prompts":len(top_bottom_gaps),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-model runner
# ──────────────────────────────────────────────────────────────────────────────

def run_model(
    model: str,
    prompt_items: List[Tuple[str, list]],
    prompt_texts: Dict[str, str],
    judge: Judge,
    skip_generation: bool = False,
    gen_workers: int = GEN_WORKERS,
    judge_workers: int = JUDGE_WORKERS,
):
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    model_dir = os.path.join(OUTPUT_BASE, model.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    results_file = os.path.join(model_dir, "results.json")

    # ── Rubric generation ──
    t0 = time.time()
    rubrics, usage = generate_rubrics_for_model(
        model, prompt_items, prompt_texts, model_dir, skip_generation, gen_workers
    )
    gen_time = time.time() - t0

    # ── Judge scoring ──
    t0 = time.time()
    per_prompt = score_rubrics(judge, prompt_items, rubrics, prompt_texts, judge_workers)
    score_time = time.time() - t0

    # ── Metrics ──
    metrics = compute_metrics(per_prompt)

    # ── Cost estimate ──
    p_in, p_out = PRICE_PER_1M.get(model, (0, 0))
    cost_usd = (
        usage.get("input_tokens", 0) / 1e6 * p_in +
        usage.get("output_tokens", 0) / 1e6 * p_out
    )

    result = {
        "model":       model,
        "output_dir":  model_dir,
        "gen_time_s":  round(gen_time, 1),
        "score_time_s":round(score_time, 1),
        "usage":       usage,
        "cost_usd_estimate": round(cost_usd, 4),
        "metrics":     metrics,
    }

    with open(results_file, "w") as f:
        json.dump({"result": result, "per_prompt": per_prompt}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n  [{model}] DONE in {gen_time+score_time:.0f}s")
    print(f"  prompts={metrics['num_prompts']}  "
          f"pairwise_acc={metrics['avg_pairwise_acc']:.4f}  "
          f"spearman={metrics['avg_spearman']:.4f}  "
          f"kendall_tau_b={metrics['avg_kendall_tau_b']:.4f}  "
          f"top_bottom={metrics['avg_top_bottom_gap']:.4f}")
    print(f"  tokens: {usage.get('input_tokens',0):,} in / {usage.get('output_tokens',0):,} out  "
          f"  est. cost: ${cost_usd:.4f}")
    print(f"  Saved: {results_file}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark frontier API models on HealthBench")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Which models to benchmark (default: all 5)")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Limit number of prompts (for quick smoke tests)")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip API rubric generation, re-score from cache")
    parser.add_argument("--gen_workers", type=int, default=GEN_WORKERS,
                        help=f"API generation workers per model (default: {GEN_WORKERS})")
    parser.add_argument("--judge_workers", type=int, default=JUDGE_WORKERS,
                        help=f"Judge scoring workers (default: {JUDGE_WORKERS})")
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially (default: parallel across models)")
    args = parser.parse_args()

    # Override module-level defaults if provided
    gen_workers   = args.gen_workers
    judge_workers = args.judge_workers

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ── Load data ──
    prompt_items, prompt_texts = load_data(args.max_prompts)

    # ── Judge (server vLLM) ──
    judge = Judge()  # reads ORACLE_* from .env
    print(f"Judge: {judge.model_name} @ {judge.api_base}")

    # ── Run models ──
    models = args.models
    all_results = {}

    if args.sequential or len(models) == 1:
        # Sequential: one model at a time
        for model in models:
            result = run_model(model, prompt_items, prompt_texts, judge,
                               args.skip_generation, gen_workers, judge_workers)
            all_results[model] = result
    else:
        # Parallel: all models run concurrently
        # Note: each model uses gen_workers API threads; judge is shared.
        print(f"\nRunning {len(models)} models in parallel...")
        results_lock = threading.Lock()

        def _run(model):
            r = run_model(model, prompt_items, prompt_texts, judge,
                          args.skip_generation, gen_workers, judge_workers)
            with results_lock:
                all_results[model] = r

        threads = [threading.Thread(target=_run, args=(m,), daemon=True)
                   for m in models]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'PW Acc':>8} {'Spearman':>10} {'Kendall τb':>11} "
          f"{'TopBot':>8} {'Cost $':>8}")
    print("-" * 80)
    for model in models:
        if model not in all_results:
            continue
        m = all_results[model]["metrics"]
        cost = all_results[model]["cost_usd_estimate"]
        print(f"{model:<30} {m['avg_pairwise_acc']:>8.4f} {m['avg_spearman']:>10.4f} "
              f"{m['avg_kendall_tau_b']:>11.4f} {m['avg_top_bottom_gap']:>8.4f} "
              f"{cost:>8.4f}")

    # Save combined summary
    summary_file = os.path.join(OUTPUT_BASE, "all_results_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nCombined summary saved: {summary_file}")


if __name__ == "__main__":
    main()
