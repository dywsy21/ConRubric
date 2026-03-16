"""Probe a checkpoint's rubric quality with a few test questions.

Usage:
    python -m scripts.probe_checkpoint_quality --step 200
    python -m scripts.probe_checkpoint_quality --step 200 --convert  # also convert FSDP→HF
    python -m scripts.probe_checkpoint_quality --step 200 --cleanup  # delete HF weights after

Tests whether the checkpoint exhibits "question parroting" (rubric just
restates the question) vs generating genuine evaluation criteria.
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_HOME", os.path.abspath("./data"))

TEST_QUESTIONS = [
    "My insulin vial is cloudy can I still use it?",
    "What are the key differences between type 1 and type 2 diabetes?",
    "I have a persistent cough for 3 weeks, should I be worried?",
]


def convert_checkpoint(step: int):
    """Convert FSDP checkpoint to HuggingFace format."""
    actor_dir = f"out/rl/global_step_{step}/actor"
    print(f"Converting step {step} FSDP → HF...")
    result = subprocess.run(
        [sys.executable, "-m", "src.training.convert_verl_ckpt", actor_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Conversion failed: {result.stderr}")
        sys.exit(1)
    print("Conversion done.")


def cleanup_hf_weights(step: int):
    """Remove HF safetensors to save disk space."""
    import glob
    hf_dir = f"out/rl/global_step_{step}/actor/huggingface"
    for f in glob.glob(os.path.join(hf_dir, "model*.safetensors")):
        os.remove(f)
        print(f"Removed {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--convert", action="store_true", help="Convert FSDP first")
    parser.add_argument("--cleanup", action="store_true", help="Delete HF weights after")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    hf_dir = f"out/rl/global_step_{args.step}/actor/huggingface"

    # Check if HF weights exist, convert if needed
    has_weights = any(
        os.path.exists(os.path.join(hf_dir, f))
        for f in ["model.safetensors", "model-00001-of-00004.safetensors"]
    )

    if not has_weights:
        if args.convert:
            convert_checkpoint(args.step)
        else:
            print(f"No HF weights at {hf_dir}. Use --convert to convert from FSDP.")
            sys.exit(1)

    # Load model and generate rubrics
    from src.models.grm import RubricGenerator
    print(f"\nLoading model from {hf_dir} on {args.device}...")
    grm = RubricGenerator(model_path=hf_dir, device=args.device)

    print(f"\n{'='*80}")
    print(f"STEP {args.step} — Rubric Quality Probe")
    print(f"{'='*80}")

    for i, q in enumerate(TEST_QUESTIONS):
        rubric = grm.generate_rubric(q)
        print(f"\n[Q{i+1}] {q}")
        print(f"{'─'*60}")
        # Show first 800 chars to avoid flooding
        display = rubric[:800]
        if len(rubric) > 800:
            display += f"\n... ({len(rubric)} chars total)"
        print(display)
        print()

    # Cleanup if requested
    if args.cleanup:
        # First delete model from GPU
        del grm
        import torch
        torch.cuda.empty_cache()
        cleanup_hf_weights(args.step)

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
