"""
Convert downloaded RubricHub Parquet files (RuRL/) to GRM SFT JSONL format.

RubricHub schema (Parquet):
    prompt:      list[{content, role}]
    Rubrics:     list[{criterion: str, points: int}]   (points 0-12, all positive)
    data_source: str

GRM SFT schema:
    question:  str   (plain text extracted from prompt)
    rubrics:   list[{criterion: str, points: int, tags: list}]   (points in [-10,10])
    source:    str
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd


def _extract_question(prompt: Any) -> str:
    """Extract plain-text question from chat-format prompt."""
    import numpy as np
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                if content:
                    parts.append(content)
        return "\n".join(parts).strip()
    return str(prompt).strip()


def _convert_rubrics(rubrics: Any) -> List[Dict[str, Any]]:
    """Convert RubricHub RuRL rubrics (all-positive 0-12) to GRM signed format."""
    import numpy as np
    if isinstance(rubrics, np.ndarray):
        rubrics = rubrics.tolist()
    if not isinstance(rubrics, list):
        return []
    out = []
    for r in rubrics:
        if not isinstance(r, dict):
            continue
        criterion = str(r.get("criterion", "")).strip()
        if not criterion:
            continue
        raw_points = r.get("points", 1)
        try:
            points = max(1, min(10, int(raw_points)))
        except (ValueError, TypeError):
            points = 1
        out.append({"criterion": criterion, "points": points, "tags": []})
    return out


def _convert_ruft_rubrics(rubrics: Any) -> List[Dict[str, Any]]:
    """Convert RubricHub RuFT rubrics (criterion + weight as float) to GRM signed format."""
    import numpy as np
    if isinstance(rubrics, np.ndarray):
        rubrics = rubrics.tolist()
    if not isinstance(rubrics, list):
        return []
    out = []
    for r in rubrics:
        if not isinstance(r, dict):
            continue
        criterion = str(r.get("criterion", r.get("criteria", ""))).strip()
        if not criterion:
            continue
        # RuFT uses 'weight' (float 0-1) instead of 'points'
        raw_weight = r.get("weight", r.get("points", 1))
        try:
            weight = float(raw_weight)
            # Map weight [0,1] to points [1,10], or if already int-like keep as-is
            if 0 < weight <= 1.0:
                points = max(1, round(weight * 10))
            else:
                points = max(1, min(10, int(weight)))
        except (ValueError, TypeError):
            points = 1
        out.append({"criterion": criterion, "points": points, "tags": []})
    return out


def convert_rubrichub(
    input_dir: str = "data/rubrichub_raw/RuRL",
    ruft_dir: str = "",
    output_path: str = "data/rubrichub_sft.jsonl",
    limit: int = 0,
    min_rubrics: int = 2,
    max_rubrics: int = 50,
    min_question_len: int = 10,
) -> str:
    """Read downloaded Parquet files and convert to GRM SFT JSONL format.

    Supports both RuRL format (prompt+Rubrics) and RuFT format (query+rubrics).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Collect parquet files from both directories
    parquet_files = []
    if input_dir and os.path.isdir(input_dir):
        parquet_files += [(pf, "RuRL") for pf in sorted(glob.glob(os.path.join(input_dir, "*.parquet")))]
    if ruft_dir and os.path.isdir(ruft_dir):
        parquet_files += [(pf, "RuFT") for pf in sorted(glob.glob(os.path.join(ruft_dir, "*.parquet")))]

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir} or {ruft_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    written = 0
    skipped = {"no_question": 0, "short_question": 0, "no_rubrics": 0,
               "too_few_rubrics": 0, "too_many_rubrics": 0, "error": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        for pf, subset in parquet_files:
            fname = os.path.basename(pf)
            print(f"  Processing {fname} ({subset})...")
            df = pd.read_parquet(pf)
            file_written = 0

            for _, row in df.iterrows():
                try:
                    # RuFT format: query + rubrics (list of {criterion, weight})
                    if subset == "RuFT":
                        question = str(row.get("query", "")).strip()
                        if not question:
                            skipped["no_question"] += 1
                            continue
                        if len(question) < min_question_len:
                            skipped["short_question"] += 1
                            continue
                        raw_rubrics = row.get("rubrics", [])
                        rubrics = _convert_ruft_rubrics(raw_rubrics)
                        source = f"rubrichub:ruft:{row.get('source', 'unknown')}"
                    else:
                        # RuRL format: prompt + Rubrics
                        question = _extract_question(row.get("prompt"))
                        if not question:
                            skipped["no_question"] += 1
                            continue
                        if len(question) < min_question_len:
                            skipped["short_question"] += 1
                            continue
                        raw_rubrics = row.get("Rubrics", [])
                        rubrics = _convert_rubrics(raw_rubrics)
                        source = f"rubrichub:{row.get('data_source', 'unknown')}"

                    if not rubrics:
                        skipped["no_rubrics"] += 1
                        continue
                    if len(rubrics) < min_rubrics:
                        skipped["too_few_rubrics"] += 1
                        continue
                    if len(rubrics) > max_rubrics:
                        skipped["too_many_rubrics"] += 1
                        continue

                    record = {
                        "question": question,
                        "rubrics": rubrics,
                        "source": source,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    file_written += 1
                except Exception as e:
                    skipped["error"] += 1
                    if skipped["error"] <= 5:
                        print(f"    Error on row: {e}")

                if limit > 0 and written >= limit:
                    break

            print(f"    -> {file_written} rows from {fname}")
            if limit > 0 and written >= limit:
                break

    print(f"\nConversion complete:")
    print(f"  Written: {written}")
    print(f"  Skipped: {skipped}")
    print(f"  Output:  {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RubricHub Parquet to GRM SFT format")
    parser.add_argument("--input-dir", type=str, default="data/rubrichub_raw/RuRL",
                        help="Directory with RuRL parquet files")
    parser.add_argument("--ruft-dir", type=str, default="",
                        help="Directory with RuFT (SFT) parquet files")
    parser.add_argument("--output", type=str, default="data/rubrichub_sft.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to convert (0=all)")
    parser.add_argument("--min-rubrics", type=int, default=2)
    parser.add_argument("--max-rubrics", type=int, default=50)
    parser.add_argument("--min-question-len", type=int, default=10)
    args = parser.parse_args()

    convert_rubrichub(
        input_dir=args.input_dir,
        ruft_dir=args.ruft_dir,
        output_path=args.output,
        limit=args.limit,
        min_rubrics=args.min_rubrics,
        max_rubrics=args.max_rubrics,
        min_question_len=args.min_question_len,
    )
