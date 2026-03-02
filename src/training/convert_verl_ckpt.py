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


def _is_lora_state_dict(state_dict: dict) -> bool:
    """Check if a state dict contains LoRA adapter keys."""
    return any("lora_A" in k or "lora_B" in k for k in state_dict)


def _strip_peft_prefix(state_dict: dict) -> dict:
    """Remove the 'base_model.model.' prefix that PEFT adds to all keys."""
    prefix = "base_model.model."
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v
        else:
            new_sd[k] = v
    return new_sd


def _separate_base_and_lora(state_dict: dict) -> tuple[dict, dict]:
    """Separate a PEFT state dict into base model weights and LoRA adapter weights.

    After stripping the 'base_model.model.' prefix, LoRA keys look like:
      model.layers.0.self_attn.q_proj.lora_A.default.weight
      model.layers.0.self_attn.q_proj.lora_B.default.weight
    Base keys look like:
      model.layers.0.self_attn.q_proj.base_layer.weight
    """
    stripped = _strip_peft_prefix(state_dict)
    base_sd = {}
    lora_sd = {}
    for k, v in stripped.items():
        if "lora_A" in k or "lora_B" in k:
            lora_sd[k] = v
        elif ".base_layer." in k:
            # PEFT wraps original params under .base_layer — unwrap for base model
            clean_key = k.replace(".base_layer.", ".")
            base_sd[clean_key] = v
        else:
            base_sd[k] = v
    return base_sd, lora_sd


def _merge_lora_into_base(base_sd: dict, lora_sd: dict, lora_alpha: int = 64, lora_rank: int = 32) -> dict:
    """Manually merge LoRA weights: W' = W + (alpha/r) * B @ A."""
    import re
    scaling = lora_alpha / lora_rank

    # Group LoRA A/B pairs by module path
    # Keys like: model.layers.0.self_attn.q_proj.lora_A.default.weight
    lora_pairs: dict[str, dict] = {}
    for k, v in lora_sd.items():
        # Extract the module path (e.g., model.layers.0.self_attn.q_proj)
        match = re.match(r"(.+)\.(lora_[AB])\.default\.weight", k)
        if match:
            module_path = match.group(1)
            ab = match.group(2)  # "lora_A" or "lora_B"
            if module_path not in lora_pairs:
                lora_pairs[module_path] = {}
            lora_pairs[module_path][ab] = v

    merged_sd = dict(base_sd)
    for module_path, pair in lora_pairs.items():
        if "lora_A" not in pair or "lora_B" not in pair:
            print(f"[convert] WARNING: Incomplete LoRA pair for {module_path}, skipping")
            continue
        weight_key = f"{module_path}.weight"
        if weight_key not in merged_sd:
            print(f"[convert] WARNING: Base weight {weight_key} not found, skipping")
            continue

        A = pair["lora_A"].float()  # (r, in_features)
        B = pair["lora_B"].float()  # (out_features, r)
        W = merged_sd[weight_key].float()
        merged_sd[weight_key] = (W + scaling * (B @ A)).to(base_sd[weight_key].dtype)

    n_merged = len(lora_pairs)
    print(f"[convert] Merged {n_merged} LoRA adapter pairs (alpha={lora_alpha}, r={lora_rank}, scale={scaling})")
    return merged_sd


def convert(step_dir: str | Path, *, force: bool = False,
            lora_rank: int = 0, lora_alpha: int = 0) -> Path:
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

    is_lora = _is_lora_state_dict(state_dict)

    if is_lora:
        print("[convert] Detected LoRA checkpoint — merging adapter into base model …")
        # Auto-detect rank if not provided
        if lora_rank <= 0:
            for k, v in state_dict.items():
                if "lora_A" in k and isinstance(v, torch.Tensor):
                    lora_rank = v.shape[0]
                    break
            if lora_rank <= 0:
                lora_rank = 32  # fallback default
            print(f"[convert] Auto-detected LoRA rank = {lora_rank}")
        if lora_alpha <= 0:
            lora_alpha = lora_rank * 2  # common default
            print(f"[convert] Using LoRA alpha = {lora_alpha}")

        base_sd, lora_sd = _separate_base_and_lora(state_dict)
        del state_dict
        state_dict = _merge_lora_into_base(base_sd, lora_sd, lora_alpha, lora_rank)
        del base_sd, lora_sd

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
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (auto-detected if 0)")
    parser.add_argument("--lora-alpha", type=int, default=0, help="LoRA alpha (defaults to 2×rank)")
    args = parser.parse_args()
    convert(args.step_dir, force=args.force, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)


if __name__ == "__main__":
    main()
