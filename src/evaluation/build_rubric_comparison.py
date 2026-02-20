"""Build a pre-vs-post SFT comparison report.

Part 1 — Metrics table: loads JSON results from the pre/post benchmark
         directories and shows accuracy / pairwise-acc side by side.
Part 2 — Rubric samples: generates rubrics from both models on a small
         sample of benchmark prompts for qualitative inspection.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.data.prepare_healthbench import ensure_healthbench_splits
from src.models.grm import RubricGenerator


# ── Metrics loading helpers ───────────────────────────────────────

_BENCH_FILES = {
    "RewardBench":     "reward_bench_results.json",
    "PPE":             "ppe_results.json",
    "RMB":             "rmb_results.json",
    "HB-RubricQuality": "healthbench_rubric_quality.json",
}


def _load_metrics(bench_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return {bench_name: {metric: value, ...}} from a benchmark dir."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, fname in _BENCH_FILES.items():
        fpath = bench_dir / fname
        if not fpath.exists():
            continue
        data = json.loads(fpath.read_text())
        summary = data.get("summary", data)
        if name == "HB-RubricQuality":
            disc = summary.get("discrimination", {})
            # Also support legacy "excellent_good_gap" key for old result files
            if not disc:
                gap = summary.get("excellent_good_gap", {})
                disc = {
                    "avg_top_bottom_gap": gap.get("excellent_minus_poor"),
                    "avg_spearman": None,
                }
            out[name] = {
                "pairwise_acc": summary.get("avg_pairwise_acc"),
                "resolution": summary.get("avg_resolution"),
                "spearman": disc.get("avg_spearman"),
                "top−bottom_gap": disc.get("avg_top_bottom_gap"),
            }
        else:
            out[name] = {"accuracy": summary.get("accuracy")}
            subset = summary.get("subset_stats", {})
            for sub, v in subset.items():
                if isinstance(v, dict) and "total" in v:
                    out[name][sub] = v["correct"] / v["total"]
    return out


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _delta(pre_v: Any, post_v: Any) -> str:
    if pre_v is None or post_v is None:
        return "—"
    if isinstance(pre_v, float) and isinstance(post_v, float):
        if math.isnan(pre_v) or math.isnan(post_v):
            return "—"
        d = post_v - pre_v
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"
    return "—"


# ── Rubric sample helpers ────────────────────────────────────────

def _load_prompts(
    max_samples: int,
    benchmark_ratio: float = 0.2,
    split_seed: int = 42,
) -> List[Dict[str, Any]]:
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=benchmark_ratio,
        seed=split_seed,
        force_rebuild=False,
    )
    prompts: List[Dict[str, Any]] = []
    with open(split_paths.benchmark_prompt_pool, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            question = row.get("question", "")
            if isinstance(question, list):
                parts = [
                    (m.get("content") or "").strip()
                    for m in question
                    if isinstance(m, dict)
                ]
                question_text = "\n".join(p for p in parts if p)
            else:
                question_text = str(question or "").strip()
            if not question_text:
                continue
            prompts.append({
                "prompt_id": row.get("prompt_id"),
                "source": row.get("source", "healthbench"),
                "question": question_text,
            })
            if len(prompts) >= max_samples:
                break
    return prompts


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build pre/post SFT comparison report")
    parser.add_argument("--pre_dir", type=str, required=True, help="Dir with pre-SFT benchmark JSONs")
    parser.add_argument("--post_dir", type=str, required=True, help="Dir with post-SFT benchmark JSONs")
    parser.add_argument("--pre_model", type=str, default=None, help="Pre-SFT model (for rubric samples)")
    parser.add_argument("--post_model", type=str, default=None, help="Post-SFT model (for rubric samples)")
    parser.add_argument("--num_rubric_samples", type=int, default=10, help="Rubric side-by-side samples")
    parser.add_argument("--output_md", type=str, default="out/bench/sft/rubric_comparison.md")
    parser.add_argument("--output_jsonl", type=str, default="out/bench/sft/rubric_comparison.jsonl")
    args = parser.parse_args()

    pre_dir = Path(args.pre_dir)
    post_dir = Path(args.post_dir)

    pre_metrics = _load_metrics(pre_dir)
    post_metrics = _load_metrics(post_dir)

    all_benches = sorted(set(pre_metrics) | set(post_metrics))

    # ── Build markdown ────────────────────────────────────────────
    lines: List[str] = []
    lines.append("# SFT Comparison: Base → Fine-tuned\n")

    # Part 1: Metrics table
    lines.append("## Benchmark Metrics\n")
    lines.append("| Benchmark | Metric | Base (pre) | SFT (post) | Δ |")
    lines.append("|---|---|---:|---:|---:|")

    jsonl_rows: List[Dict[str, Any]] = []

    for bench in all_benches:
        pre_m = pre_metrics.get(bench, {})
        post_m = post_metrics.get(bench, {})
        all_keys = list(dict.fromkeys(list(pre_m) + list(post_m)))
        first = True
        for key in all_keys:
            pv = pre_m.get(key)
            sv = post_m.get(key)
            bench_col = bench if first else ""
            lines.append(f"| {bench_col} | {key} | {_fmt(pv)} | {_fmt(sv)} | {_delta(pv, sv)} |")
            jsonl_rows.append({
                "benchmark": bench,
                "metric": key,
                "pre": pv,
                "post": sv,
            })
            first = False

    lines.append("")

    # Part 2: Rubric samples (optional — needs models)
    if args.pre_model and args.post_model:
        lines.append("## Rubric Samples (side-by-side)\n")
        prompts = _load_prompts(args.num_rubric_samples)
        if prompts:
            pre_grm = RubricGenerator(args.pre_model)
            post_grm = RubricGenerator(args.post_model)

            lines.append("| # | Question (truncated) | Base rubric | SFT rubric |")
            lines.append("|---:|---|---|---|")

            sample_rows: List[Dict[str, Any]] = []
            for idx, p in enumerate(tqdm(prompts, desc="Generating rubric samples"), start=1):
                q = p["question"]
                pre_rub = pre_grm.generate_rubric(q)
                post_rub = post_grm.generate_rubric(q)
                lines.append(
                    f"| {idx} "
                    f"| {_escape_md(q[:200])} "
                    f"| {_escape_md(pre_rub[:400])} "
                    f"| {_escape_md(post_rub[:400])} |"
                )
                sample_rows.append({
                    "index": idx,
                    "prompt_id": p["prompt_id"],
                    "question": q,
                    "pre_rubric": pre_rub,
                    "post_rubric": post_rub,
                })

            for sr in sample_rows:
                jsonl_rows.append(sr)

            lines.append("")
        else:
            lines.append("_No benchmark prompts available for rubric samples._\n")
    else:
        lines.append("_Rubric samples skipped (pass --pre_model and --post_model to enable)._\n")

    # ── Write outputs ─────────────────────────────────────────────
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in jsonl_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved comparison markdown: {out_md}")
    print(f"Saved comparison jsonl:    {out_jsonl}")


if __name__ == "__main__":
    main()
