#!/usr/bin/env python3
"""Convert a verl FSDP checkpoint to a standard HuggingFace model directory.

verl saves:
  global_step_N/
    model_world_size_1_rank_0.pt   ← full state_dict
    huggingface/                   ← config.json + tokenizer only
    optim_world_size_1_rank_0.pt
    ...

This script produces:
  global_step_N/
    huggingface/                   ← now includes model.safetensors (or .bin)

Usage:
  python -m src.training.convert_verl_ckpt out/sft/global_step_6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def convert(step_dir: str | Path, *, force: bool = False) -> Path:
    step_dir = Path(step_dir)
    hf_dir = step_dir / "huggingface"

    if not hf_dir.exists():
        raise FileNotFoundError(f"No huggingface/ subfolder in {step_dir}")

    # Already converted?
    weights = list(hf_dir.glob("model*.safetensors")) + list(hf_dir.glob("pytorch_model*.bin"))
    if weights and not force:
        print(f"[convert] Already converted: {weights[0].name}")
        return hf_dir

    # Find the FSDP state dict
    pt_files = sorted(step_dir.glob("model_world_size_*_rank_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No model_world_size_*_rank_*.pt in {step_dir}")

    print(f"[convert] Loading state_dict from {pt_files[0].name} …")
    state_dict = torch.load(pt_files[0], map_location="cpu", weights_only=False)

    print(f"[convert] Loading config from {hf_dir} …")
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)

    print("[convert] Instantiating model on CPU …")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.load_state_dict(state_dict, strict=True)

    print(f"[convert] Saving HF model to {hf_dir} …")
    model.save_pretrained(hf_dir, safe_serialization=True)

    # Also re-save tokenizer to be safe (already there, but ensures consistency)
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    tokenizer.save_pretrained(hf_dir)

    print("[convert] Done ✓")
    return hf_dir


def main():
    parser = argparse.ArgumentParser(description="Convert verl FSDP checkpoint → HF model")
    parser.add_argument("step_dir", help="Path to global_step_N directory")
    parser.add_argument("--force", action="store_true", help="Re-convert even if weights exist")
    args = parser.parse_args()
    convert(args.step_dir, force=args.force)


if __name__ == "__main__":
    main()
