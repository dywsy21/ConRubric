"""Prepare RL training data (verl-compatible parquet).

Merges multiple sources — RubricHub, synthetic rubrics, and HealthBench SFT
split — into a single parquet file that verl's RL dataset loader can consume.

The HealthBench *benchmark* split (20 %) is excluded to avoid evaluation leakage.

When ``--uniform-mix`` is enabled the smaller source is **upsampled** (with
repetition) so that both contribute equally many rows.  The final dataframe
is shuffled so that sources are interleaved ("round-robin-ish").

Usage:
    python -m src.scripts.prepare_rl_data [--synthetic-limit N] [--healthbench-limit N]
    python -m src.scripts.prepare_rl_data --rubrichub-path data/rubrichub_sft.jsonl --uniform-mix
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from src.data.prepare_healthbench import ensure_healthbench_splits
from src.utils.prompts import RUBRIC_GENERATION_PROMPT

OUTPUT_PATH = "data/rl_train.parquet"


# ── helpers ───────────────────────────────────────────────────────────────
def _format_rubric_item(item) -> str:
    if isinstance(item, dict):
        criterion = str(item.get("criterion", "")).strip()
        points = item.get("points", None)
        if points is None:
            return f"- {criterion}" if criterion else ""
        sign = "+" if float(points) > 0 else ""
        return f"- [{sign}{points}] {criterion}" if criterion else ""
    if isinstance(item, str):
        text = item.strip()
        return f"- {text}" if text else ""
    return ""


def _make_row(question: str, rubric_items: list, source: str, gold_answer: str = "") -> dict:
    """Build a single verl-compatible row."""
    prompt_messages = [
        {
            "role": "user",
            "content": RUBRIC_GENERATION_PROMPT.format(question=question),
        }
    ]

    rubric_lines = [_format_rubric_item(r) for r in rubric_items]
    rubric_lines = [ln for ln in rubric_lines if ln]
    rubric_text = "\n".join(rubric_lines)

    return {
        "prompt": prompt_messages,
        "response": rubric_text,
        "question": question,
        "gold_answer": gold_answer,
        "data_source": source,
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "question": question,
                "gold_answer": gold_answer,
                "gold_rubric": rubric_text,
            },
        },
        "extra_info": {
            "question": question,
        },
    }


# ── loaders ───────────────────────────────────────────────────────────────
def _load_synthetic(path: str, limit: int | None = None) -> list[dict]:
    if not os.path.exists(path):
        print(f"[prepare_rl_data] Synthetic file not found: {path}")
        return []

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = (item.get("question") or "").strip()
            rubric_list = item.get("rubrics") or item.get("rubric", [])
            if not question or not rubric_list:
                continue
            rows.append(
                _make_row(
                    question=question,
                    rubric_items=rubric_list,
                    source="grm_synthetic",
                    gold_answer=item.get("gold_answer", ""),
                )
            )
            if limit is not None and len(rows) >= limit:
                break

    print(f"[prepare_rl_data] Loaded {len(rows)} synthetic rows from {path}")
    return rows


def _load_healthbench(limit: int | None = None) -> list[dict]:
    """Load HealthBench SFT-split rows (benchmark split is excluded)."""
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=0.2,
        seed=42,
    )

    rows: list[dict] = []
    with open(split_paths.sft_train, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = (item.get("question") or "").strip()
            rubric_items = item.get("rubrics", [])
            if not question or not rubric_items:
                continue
            rows.append(
                _make_row(
                    question=question,
                    rubric_items=rubric_items,
                    source=item.get("source", "healthbench"),
                    gold_answer="",
                )
            )
            if limit is not None and len(rows) >= limit:
                break

    print(f"[prepare_rl_data] Loaded {len(rows)} HealthBench SFT-split rows from {split_paths.sft_train}")
    return rows


def _load_rubrichub(path: str, limit: int | None = None) -> list[dict]:
    """Load RubricHub SFT JSONL (already converted by prepare_rubrichub.py)."""
    if not os.path.exists(path):
        print(f"[prepare_rl_data] RubricHub file not found: {path}")
        return []

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = (item.get("question") or "").strip()
            rubric_list = item.get("rubrics") or item.get("rubric", [])
            if not question or not rubric_list:
                continue
            rows.append(
                _make_row(
                    question=question,
                    rubric_items=rubric_list,
                    source=item.get("source", "rubrichub"),
                    gold_answer=item.get("gold_answer", ""),
                )
            )
            if limit is not None and len(rows) >= limit:
                break

    print(f"[prepare_rl_data] Loaded {len(rows)} RubricHub rows from {path}")
    return rows


# ── main ──────────────────────────────────────────────────────────────────
VAL_OUTPUT_PATH = "data/rl_val_healthbench.parquet"


def _prepare_val_parquet(
    max_prompts: int = 50,
    seed: int = 42,
    output_path: str = VAL_OUTPUT_PATH,
) -> str:
    """Prepare a small HealthBench validation parquet from benchmark_prompt_pool.

    Uses the benchmark split (20%) which is guaranteed leak-free from SFT/RL train.
    The gold rubric is stored in ground_truth for keyword coverage evaluation.
    """
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=0.2,
        seed=seed,
        force_rebuild=False,
    )

    rows: list[dict] = []
    with open(split_paths.benchmark_prompt_pool, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = (item.get("question") or "").strip()
            rubric_list = item.get("rubrics", [])
            if not question or not rubric_list:
                continue
            rows.append(
                _make_row(
                    question=question,
                    rubric_items=rubric_list,
                    source="healthbench_val",
                )
            )

    rng = np.random.default_rng(seed=seed)
    if max_prompts and len(rows) > max_prompts:
        indices = rng.choice(len(rows), size=max_prompts, replace=False)
        rows = [rows[i] for i in sorted(indices)]

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path)

    print(f"[prepare_rl_data] Validation parquet: {len(df)} rows → {output_path}")
    return output_path


# ── main ──────────────────────────────────────────────────────────────────
def _upsample(rows: list[dict], target_size: int, rng: np.random.Generator) -> list[dict]:
    """Upsample *rows* to *target_size* by repeating + random sampling the remainder."""
    if len(rows) >= target_size:
        return rows[:target_size]
    full_copies = target_size // len(rows)
    remainder = target_size % len(rows)
    result = rows * full_copies
    if remainder > 0:
        indices = rng.choice(len(rows), size=remainder, replace=False)
        result.extend(rows[i] for i in indices)
    return result


def prepare(args: argparse.Namespace) -> None:
    synthetic_path = args.synthetic_path or "data/synthetic_rubrics.jsonl"
    output_path = args.output or OUTPUT_PATH
    rng = np.random.default_rng(seed=args.seed)

    # ── Optionally prepare validation parquet from HealthBench benchmark split ──
    if getattr(args, "prepare_val", False):
        _prepare_val_parquet(
            max_prompts=getattr(args, "val_max_prompts", 50),
            seed=args.seed,
        )
        return

    # ── Collect rows from each source ────────────────────────────────
    source_buckets: dict[str, list[dict]] = {}

    if not args.no_synthetic:
        synth = _load_synthetic(synthetic_path, limit=args.synthetic_limit)
        if synth:
            source_buckets["synthetic"] = synth

    if not args.no_healthbench:
        hb = _load_healthbench(limit=args.healthbench_limit)
        if hb:
            source_buckets["healthbench"] = hb

    if args.rubrichub_path:
        rh = _load_rubrichub(args.rubrichub_path, limit=args.rubrichub_limit)
        if rh:
            source_buckets["rubrichub"] = rh

    if not source_buckets:
        print("[prepare_rl_data] No data found. Make sure at least one source exists.")
        return

    # ── Uniform mix: upsample smaller sources to match largest ────────
    if args.uniform_mix and len(source_buckets) > 1:
        max_size = max(len(v) for v in source_buckets.values())
        print(f"[prepare_rl_data] Uniform mix enabled — upsampling all sources to {max_size} rows each")
        for name, bucket in source_buckets.items():
            if len(bucket) < max_size:
                print(f"  {name}: {len(bucket)} → {max_size} (×{max_size / len(bucket):.1f})")
                source_buckets[name] = _upsample(bucket, max_size, rng)

    # ── Merge and interleave ──────────────────────────────────────────
    rows: list[dict] = []
    for bucket in source_buckets.values():
        rows.extend(bucket)

    # Always shuffle so sources are interleaved (important for RL batches)
    rng.shuffle(rows)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path)

    source_counts = df["data_source"].value_counts().to_dict()
    print(f"[prepare_rl_data] Wrote {len(df)} rows to {output_path}")
    print(f"[prepare_rl_data] Source breakdown: {source_counts}")


def main():
    parser = argparse.ArgumentParser(description="Prepare RL training parquet from synthetic + HealthBench data")
    parser.add_argument("--synthetic-path", type=str, default=None, help="Path to synthetic_rubrics.jsonl")
    parser.add_argument("--synthetic-limit", type=int, default=None, help="Max synthetic rows")
    parser.add_argument("--healthbench-limit", type=int, default=None, help="Max HealthBench rows")
    parser.add_argument("--rubrichub-path", type=str, default=None,
                        help="Path to rubrichub_sft.jsonl (converted by prepare_rubrichub.py)")
    parser.add_argument("--rubrichub-limit", type=int, default=None, help="Max RubricHub rows")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic data")
    parser.add_argument("--no-healthbench", action="store_true", help="Skip HealthBench data")
    parser.add_argument("--uniform-mix", action="store_true",
                        help="Upsample smaller sources to match the largest, then shuffle")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for upsampling/shuffling")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help=f"Output parquet path (default: {OUTPUT_PATH})")
    parser.add_argument("--prepare-val", action="store_true",
                        help="Only prepare the HealthBench validation parquet (fast, no training data)")
    parser.add_argument("--val-max-prompts", type=int, default=50,
                        help="Max prompts for validation parquet (default: 50)")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
