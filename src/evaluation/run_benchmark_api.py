"""Run benchmarks using an API-based rubric generator (e.g., GLM-5).

Usage:
    python -m src.evaluation.run_benchmark_api \
        --api_base https://open.bigmodel.cn/api/paas/v4 \
        --api_key YOUR_KEY \
        --api_model glm-5 \
        --benchmarks rewardbench ppe rmb healthbench_rubric \
        --num_samples 200 \
        --output_dir out/bench/glm5 \
        --api_workers 4
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

# Set HF_HOME before importing HF libs
hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")

from src.config import ProjectConfig
from src.evaluation.judge import Judge
from src.evaluation.run_benchmark import (
    run_reward_bench,
    run_ppe_benchmark,
    run_rmb_benchmark,
)
from src.evaluation.run_rubric_quality_benchmark import run_healthbench_rubric_quality
from src.models.api_grm import APIRubricGenerator


def main():
    parser = argparse.ArgumentParser(description="Run GRM benchmarks with an API-based rubric generator")
    parser.add_argument("--api_base", type=str, required=True, help="API base URL")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--api_model", type=str, required=True, help="Model name for the API")
    parser.add_argument("--api_workers", type=int, default=4, help="Parallel API workers for rubric generation")
    parser.add_argument("--api_max_tokens", type=int, default=4096, help="Max tokens for rubric generation")
    parser.add_argument("--no_thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["rewardbench"],
                        help="Benchmarks: rewardbench, ppe, rmb, healthbench_rubric")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="out/bench/api")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_workers", type=int,
                        default=int(os.getenv("GRM_ORACLE_WORKERS", "8")))

    args = parser.parse_args()

    # Initialize API-based rubric generator
    grm = APIRubricGenerator(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.api_model,
        max_workers=args.api_workers,
        max_tokens=args.api_max_tokens,
        thinking=not args.no_thinking,
    )

    # Initialize judge (same as standard benchmark)
    project_config = ProjectConfig()
    judge = Judge(
        model_name=project_config.oracle_model_name,
        api_key=project_config.oracle_api_key,
        api_base=project_config.oracle_api_base,
    )

    if "rewardbench" in args.benchmarks:
        run_reward_bench(grm, judge, args.num_samples, args.output_dir,
                         eval_batch_size=args.eval_batch_size, max_workers=args.eval_workers)

    if "ppe" in args.benchmarks:
        run_ppe_benchmark(grm, judge, args.num_samples, args.output_dir,
                          eval_batch_size=args.eval_batch_size, max_workers=args.eval_workers)

    if "rmb" in args.benchmarks:
        run_rmb_benchmark(grm, judge, args.num_samples, args.output_dir,
                          eval_batch_size=args.eval_batch_size, max_workers=args.eval_workers)

    if "healthbench_rubric" in args.benchmarks:
        run_healthbench_rubric_quality(
            grm=grm,
            judge=judge,
            output_dir=args.output_dir,
            max_prompts=args.num_samples,
            max_completions_per_prompt=8,
            eval_workers=args.eval_workers,
        )

    print(f"\n[DONE] All benchmarks complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
