"""Merge a PEFT LoRA adapter into the base model to produce a standalone HF model.

Usage:
    python -m src.scripts.merge_lora \
        --base-model out/sft/global_step_500/huggingface \
        --lora-adapter out/sft_lora/global_step_300/lora_adapter \
        --output out/sft_lora_merged

The merged model can then be loaded by RubricGenerator or any
AutoModelForCausalLM.from_pretrained() call without PEFT dependency.
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights into base model"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to the base HuggingFace model (e.g. SFT-v1 checkpoint)",
    )
    parser.add_argument(
        "--lora-adapter",
        required=True,
        help="Path to the LoRA adapter directory (contains adapter_config.json + adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the merged model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model loading and saving",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading base model from: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter from: {args.lora_adapter}")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_adapter)

    print("Merging LoRA weights into base model (W' = W + B×A)...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")
    merged_model.save_pretrained(args.output)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    tokenizer.save_pretrained(args.output)

    n_params = sum(p.numel() for p in merged_model.parameters())
    print(f"Done. Merged model: {n_params / 1e9:.2f}B parameters.")
    print(f"Load with: RubricGenerator('{args.output}')")


if __name__ == "__main__":
    main()
