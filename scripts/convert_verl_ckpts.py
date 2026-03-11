#!/usr/bin/env python3
"""Convert verl actor checkpoints to HuggingFace safetensors format.

Each RL checkpoint has:
  out/rl/global_step_{N}/actor/model_world_size_1_rank_0.pt  (state_dict, fp32)
  out/rl/global_step_{N}/actor/huggingface/config.json       (already exists)

This script loads the PT state_dict, casts to float16, and saves as
safetensors in the huggingface/ directory using model.save_pretrained().

Usage:
    python scripts/convert_verl_ckpts.py                      # all missing
    python scripts/convert_verl_ckpts.py --steps 300,500,700   # specific steps
    python scripts/convert_verl_ckpts.py --force               # overwrite existing
"""

import argparse
import glob
import os
import re
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def find_missing_checkpoints(rl_dir: str, steps: list[int] | None = None, force: bool = False) -> list[tuple[int, str, str]]:
    """Return list of (step, pt_path, hf_dir) that need conversion."""
    results = []
    if steps:
        dirs = [os.path.join(rl_dir, f"global_step_{s}") for s in steps]
    else:
        dirs = sorted(glob.glob(os.path.join(rl_dir, "global_step_*")))
    
    for d in dirs:
        if not os.path.isdir(d):
            continue
        m = re.search(r"global_step_(\d+)$", d)
        if not m:
            continue
        step = int(m.group(1))
        pt_path = os.path.join(d, "actor", "model_world_size_1_rank_0.pt")
        hf_dir = os.path.join(d, "actor", "huggingface")
        
        if not os.path.isfile(pt_path):
            print(f"  [SKIP] Step {step}: no PT checkpoint found")
            continue
        if not os.path.isdir(hf_dir) or not os.path.isfile(os.path.join(hf_dir, "config.json")):
            print(f"  [SKIP] Step {step}: no huggingface config dir")
            continue
        
        # Check if already has safetensors
        has_weights = any(
            f.endswith(".safetensors") 
            for f in os.listdir(hf_dir)
        )
        if has_weights and not force:
            print(f"  [SKIP] Step {step}: already has safetensors")
            continue
        
        results.append((step, pt_path, hf_dir))
    
    return results


def convert_checkpoint(step: int, pt_path: str, hf_dir: str) -> bool:
    """Convert a single checkpoint. Returns True on success."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Converting step {step}")
    print(f"  PT:  {pt_path}")
    print(f"  HF:  {hf_dir}")
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
        
        # Load state dict
        print(f"  Loading state dict...", end=" ", flush=True)
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        print(f"done ({len(state_dict)} keys)")
        
        # Cast to float16
        print(f"  Casting to float16...", end=" ", flush=True)
        for k in state_dict:
            if state_dict[k].dtype == torch.float32:
                state_dict[k] = state_dict[k].half()
        print("done")
        
        # Create model from config and load state dict
        print(f"  Creating model from config...", end=" ", flush=True)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        print("done")
        
        print(f"  Loading state dict into model...", end=" ", flush=True)
        model.load_state_dict(state_dict, strict=True)
        print("done")
        
        # Free the extra state dict reference
        del state_dict
        
        # Save in HuggingFace format
        print(f"  Saving safetensors to {hf_dir}...", end=" ", flush=True)
        model.save_pretrained(hf_dir, safe_serialization=True, max_shard_size="5GB")
        print("done")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        elapsed = time.time() - t0
        print(f"  Step {step} converted in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n  [ERROR] Step {step}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert verl checkpoints to HF format")
    parser.add_argument("--rl-dir", default="out/rl", help="RL output directory")
    parser.add_argument("--steps", type=str, default=None, help="Comma-separated step numbers")
    parser.add_argument("--force", action="store_true", help="Overwrite existing safetensors")
    args = parser.parse_args()
    
    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    
    print(f"Scanning {args.rl_dir} for checkpoints to convert...")
    todo = find_missing_checkpoints(args.rl_dir, steps, args.force)
    
    if not todo:
        print("No checkpoints need conversion.")
        return
    
    print(f"\n{len(todo)} checkpoints to convert:")
    for step, _, _ in todo:
        print(f"  - Step {step}")
    
    success = 0
    failed = 0
    for step, pt_path, hf_dir in todo:
        if convert_checkpoint(step, pt_path, hf_dir):
            success += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {success} success, {failed} failed")


if __name__ == "__main__":
    main()
