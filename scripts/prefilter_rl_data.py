#!/usr/bin/env python3
"""Pre-filter RL training data by prompt length.

Tokenizes every prompt with the model's chat template, drops rows exceeding
max_prompt_length tokens, and writes a new parquet file.  This turns the
~10-minute runtime filter into a one-time offline step.

Usage:
    python -m scripts.prefilter_rl_data \
        --input  data/rl_train.parquet \
        --output data/rl_train_filtered.parquet \
        --model  out/sft_lora_a16/huggingface \
        --max-prompt-length 1024 \
        --num-proc 8
"""

from __future__ import annotations

import argparse
import os
import time

import datasets
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Pre-filter RL data by prompt token length")
    parser.add_argument("--input", default="data/rl_train.parquet", help="Input parquet file")
    parser.add_argument("--output", default="data/rl_train_filtered.parquet", help="Output parquet file")
    parser.add_argument("--model", default=None, help="Model path for tokenizer (default: GRM_MODEL_NAME or BASE_MODEL)")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Max prompt length in tokens")
    parser.add_argument("--num-proc", type=int, default=None, help="Number of workers for filtering (default: cpu_count/4)")
    parser.add_argument("--prompt-key", default="prompt", help="Column name for prompt")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model or os.getenv("GRM_MODEL_NAME") or os.getenv("BASE_MODEL")
    if not model_path:
        raise ValueError("No model path: set --model, GRM_MODEL_NAME, or BASE_MODEL")

    num_proc = args.num_proc or max(1, os.cpu_count() // 4)

    print(f"[prefilter] Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[prefilter] Loading dataset from: {args.input}")
    ds = datasets.load_dataset("parquet", data_files=args.input)["train"]
    total = len(ds)
    print(f"[prefilter] Total rows: {total}")

    prompt_key = args.prompt_key
    max_len = args.max_prompt_length

    def prompt_within_length(doc) -> bool:
        try:
            tokens = tokenizer.apply_chat_template(
                doc[prompt_key], add_generation_prompt=True
            )
            return len(tokens) <= max_len
        except Exception:
            return False

    print(f"[prefilter] Filtering prompts > {max_len} tokens using {num_proc} workers...")
    t0 = time.time()
    ds_filtered = ds.filter(
        prompt_within_length,
        num_proc=num_proc,
        desc=f"Filtering prompts > {max_len} tokens",
    )
    elapsed = time.time() - t0

    kept = len(ds_filtered)
    dropped = total - kept
    print(f"[prefilter] Done in {elapsed:.1f}s: kept {kept}/{total} ({dropped} dropped, {dropped/total*100:.1f}%)")

    print(f"[prefilter] Saving to: {args.output}")
    ds_filtered.to_parquet(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[prefilter] Saved {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
