import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from src.data.prepare_healthbench import ensure_healthbench_splits
from src.models.grm import RubricGenerator


def _load_prompts(max_samples: int, benchmark_ratio: float, split_seed: int) -> List[Dict[str, Any]]:
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
            prompt = row.get("prompt")
            if isinstance(prompt, list):
                parts = []
                for m in prompt:
                    if isinstance(m, dict):
                        c = (m.get("content") or "").strip()
                        if c:
                            parts.append(c)
                prompt_text = "\n".join(parts)
            else:
                prompt_text = str(prompt or "").strip()

            if not prompt_text:
                continue

            prompts.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "source": row.get("source", "healthbench"),
                    "question": prompt_text,
                }
            )
            if len(prompts) >= max_samples:
                break

    return prompts


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser(description="Build pre/post SFT rubric comparison table")
    parser.add_argument("--pre_model", type=str, required=True)
    parser.add_argument("--post_model", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=120)
    parser.add_argument("--benchmark_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--output_md", type=str, default="out/bench/sft/rubric_comparison.md")
    parser.add_argument("--output_jsonl", type=str, default="out/bench/sft/rubric_comparison.jsonl")
    args = parser.parse_args()

    if args.num_samples <= 100:
        raise ValueError("--num_samples must be > 100")

    prompts = _load_prompts(args.num_samples, args.benchmark_ratio, args.split_seed)
    if len(prompts) <= 100:
        raise ValueError(f"Need >100 prompts, got {len(prompts)}")

    pre = RubricGenerator(args.pre_model)
    post = RubricGenerator(args.post_model)

    rows: List[Dict[str, Any]] = []
    for idx, p in enumerate(prompts, start=1):
        q = p["question"]
        pre_rubric = pre.generate_rubric(q)
        post_rubric = post.generate_rubric(q)
        rows.append(
            {
                "index": idx,
                "prompt_id": p["prompt_id"],
                "source": p["source"],
                "question": q,
                "pre_rubric": pre_rubric,
                "post_rubric": post_rubric,
            }
        )

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Rubric Comparison (Pre vs Post SFT)\n\n")
        f.write(f"Total samples: {len(rows)}\n\n")
        f.write("| # | prompt_id | question | pre_rubric | post_rubric |\n")
        f.write("|---:|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['index']} | {_escape_md(str(r['prompt_id']))} | {_escape_md(r['question'][:220])} | {_escape_md(r['pre_rubric'][:320])} | {_escape_md(r['post_rubric'][:320])} |\\n"
            )

    print(f"Saved comparison markdown: {out_md}")
    print(f"Saved comparison jsonl: {out_jsonl}")


if __name__ == "__main__":
    main()
