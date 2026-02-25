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


def _to_plain(v: torch.Tensor) -> torch.Tensor:
    """Extract plain torch.Tensor from a DTensor (or return as-is)."""
    if type(v).__name__ == "DTensor" or hasattr(v, "_local_tensor"):
        local = v._local_tensor if hasattr(v, "_local_tensor") else v
        return local.detach().cpu().clone()
    elif isinstance(v, torch.Tensor):
        return v.detach().cpu()
    return v


def _load_and_merge_shards(step_dir: Path) -> dict:
    """Load all FSDP rank shards and concatenate along dim 0.

    verl FSDP2 saves per-parameter shards (DTensor with Shard(0) placement).
    Each rank file has 1/world_size of every parameter along dimension 0.
    We load all ranks and torch.cat them to reconstruct full parameters.
    """
    import re

    # Find all rank files and determine world_size
    all_pt = sorted(step_dir.glob("model_world_size_*_rank_*.pt"))
    if not all_pt:
        raise FileNotFoundError(f"No model_world_size_*_rank_*.pt in {step_dir}")

    # Parse world_size and rank from filenames
    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
    rank_files = {}
    world_size = None
    for f in all_pt:
        m = pattern.match(f.name)
        if not m:
            continue
        ws, rank = int(m.group(1)), int(m.group(2))
        if world_size is None:
            world_size = ws
        rank_files[rank] = f

    if world_size == 1:
        # Single-rank checkpoint — no sharding, just load and de-tensor
        print(f"[convert] Single-rank checkpoint, loading {rank_files[0].name} …")
        sd = torch.load(rank_files[0], map_location="cpu", weights_only=False)
        return {k: _to_plain(v) for k, v in sd.items()}

    print(f"[convert] Found {len(rank_files)} shards (world_size={world_size})")
    assert len(rank_files) == world_size, (
        f"Expected {world_size} rank files, found {len(rank_files)}"
    )

    # Load all shards
    shards = {}
    for rank in range(world_size):
        print(f"[convert]   Loading rank {rank} …")
        sd = torch.load(rank_files[rank], map_location="cpu", weights_only=False)
        shards[rank] = {k: _to_plain(v) for k, v in sd.items()}

    # Merge: concatenate each parameter across ranks along dim 0
    keys = list(shards[0].keys())
    merged = {}
    for k in keys:
        parts = [shards[r][k] for r in range(world_size)]
        if isinstance(parts[0], torch.Tensor) and parts[0].dim() >= 1:
            merged[k] = torch.cat(parts, dim=0)
        else:
            # Scalar or non-tensor — just take rank 0
            merged[k] = parts[0]

    # Free shard memory
    del shards
    return merged


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

    # Load and merge all FSDP shards
    state_dict = _load_and_merge_shards(step_dir)

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
