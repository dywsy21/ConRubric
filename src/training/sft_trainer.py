import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.config import ProjectConfig
from src.data.prepare_healthbench import ensure_healthbench_splits
from src.utils.prompts import RUBRIC_GENERATION_PROMPT

load_dotenv()

HEALTHBENCH_REPO = "openai/healthbench"
HEALTHBENCH_SFT_FILES = [
    "2025-05-07-06-14-12_oss_eval.jsonl",
    "consensus_2025-05-09-20-00-46.jsonl",
    "hard_2025-05-08-21-00-10.jsonl",
]


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        lines = []
        for m in prompt:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines).strip()
    return str(prompt).strip()


def _normalize_rubrics(raw_rubrics: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw_rubrics, list):
        return out

    for r in raw_rubrics:
        if isinstance(r, dict):
            criterion = str(r.get("criterion", "")).strip()
            if not criterion:
                continue
            points = int(r.get("points", 1))
            tags = r.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            out.append({"criterion": criterion, "points": points, "tags": tags})
        else:
            criterion = str(r).strip()
            if criterion:
                out.append({"criterion": criterion, "points": 1, "tags": []})
    return out


def _load_healthbench_records(
    limit_per_file: Optional[int] = None,
    limit_total: Optional[int] = None,
    benchmark_ratio: float = 0.2,
    seed: int = 42,
    force_rebuild_split: bool = False,
) -> List[Dict[str, Any]]:
    # Load only from SFT split to avoid benchmark leakage.
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=benchmark_ratio,
        seed=seed,
        force_rebuild=force_rebuild_split,
    )

    records: List[Dict[str, Any]] = []
    with open(split_paths.sft_train, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = (item.get("question") or "").strip()
            rubrics = _normalize_rubrics(item.get("rubrics", []))
            if not question or not rubrics:
                continue
            records.append({"question": question, "rubrics": rubrics, "source": item.get("source", "healthbench:sft_train")})
            if limit_total is not None and len(records) >= limit_total:
                break
            if limit_total is None and limit_per_file is not None and len(records) >= (limit_per_file * len(HEALTHBENCH_SFT_FILES)):
                break

    print(f"Loaded {len(records)} HealthBench SFT rows from split file: {split_paths.sft_train}")
    return records


def _load_synthetic_records(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Synthetic file not found: {path}")
        return []

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            question = (item.get("question") or "").strip()
            rubrics = _normalize_rubrics(item.get("rubrics", item.get("rubric", [])))
            if not question or not rubrics:
                continue
            records.append({"question": question, "rubrics": rubrics, "source": item.get("source", "synthetic")})
            if limit is not None and len(records) >= limit:
                break
    print(f"Loaded {len(records)} synthetic rows from {path}")
    return records


def _build_weighted_sft_jsonl(args, config: ProjectConfig) -> str:
    os.makedirs(config.data_dir, exist_ok=True)
    output_jsonl = os.path.abspath(args.output_jsonl or os.path.join(config.data_dir, "sft_weighted_mix.jsonl"))

    records: List[Dict[str, Any]] = []
    if not args.no_healthbench:
        records.extend(
            _load_healthbench_records(
                limit_per_file=args.healthbench_limit_per_file,
                limit_total=args.healthbench_limit,
                benchmark_ratio=args.healthbench_benchmark_ratio,
                seed=args.healthbench_split_seed,
                force_rebuild_split=args.rebuild_healthbench_split,
            )
        )

    if not args.no_synthetic:
        synthetic_path = args.synthetic_path or os.path.join(config.data_dir, "synthetic_rubrics.jsonl")
        records.extend(_load_synthetic_records(synthetic_path, limit=args.synthetic_limit))

    if not records:
        raise ValueError("No SFT records prepared. Enable at least one source or fix paths.")

    if args.preview_samples > 0:
        n = min(args.preview_samples, len(records))
        print(f"\nPreviewing {n} SFT samples:")
        for i in range(n):
            r = records[i]
            question = r["question"]
            prompt = RUBRIC_GENERATION_PROMPT.format(question=question)
            rubrics = r.get("rubrics", [])
            print("=" * 80)
            print(f"[Sample {i + 1}] source={r.get('source', 'unknown')}")
            print("Prompt fed to model:")
            print(prompt[:1200])
            print("Target rubric (first 6 criteria):")
            for j, item in enumerate(rubrics[:6], start=1):
                pts = int(item.get("points", 1))
                criterion = item.get("criterion", "")
                tags = item.get("tags", [])
                tag_str = f" | tags: {', '.join(tags)}" if tags else ""
                print(f"  {j}. [{pts:+d}] {criterion}{tag_str}")
            if len(rubrics) > 6:
                print(f"  ... ({len(rubrics) - 6} more criteria)")
        print("=" * 80 + "\n")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Prepared weighted SFT jsonl: {output_jsonl} ({len(records)} rows)")
    return output_jsonl


def _merge_fsdp_checkpoint_if_needed(output_dir: str, nproc: int) -> Optional[str]:
    """Check if latest checkpoint was saved with a different world_size.

    When elastic scaling changes the GPU count, FSDP sharded checkpoints
    cannot be loaded because the filenames contain the old world_size.
    This function detects the mismatch and merges the FSDP shards into
    HuggingFace format so training can restart from the merged weights.

    Returns the path to the HF model directory if conversion was needed,
    or ``None`` if no conversion is required (same world_size or no ckpt).
    """
    tracker = os.path.join(output_dir, "latest_checkpointed_iteration.txt")
    if not os.path.exists(tracker):
        return None

    with open(tracker) as f:
        step = f.read().strip()
    ckpt_dir = os.path.join(output_dir, f"global_step_{step}")

    fsdp_config_path = os.path.join(ckpt_dir, "fsdp_config.json")
    if not os.path.exists(fsdp_config_path):
        return None

    with open(fsdp_config_path) as f:
        fsdp_cfg = json.load(f)
    ckpt_world_size = fsdp_cfg.get("world_size")
    if ckpt_world_size is None or ckpt_world_size == nproc:
        return None  # same world_size → normal resume works

    hf_path = os.path.join(ckpt_dir, "huggingface")

    # Check if HF model weights already exist (not just config/tokenizer).
    hf_weight_files = (
        glob.glob(os.path.join(hf_path, "model*.safetensors"))
        + glob.glob(os.path.join(hf_path, "model*.bin"))
    )
    if hf_weight_files:
        print(
            f"[elastic] World-size mismatch (ckpt={ckpt_world_size}, "
            f"new={nproc}) but HF weights already at {hf_path}"
        )
        return hf_path

    # ── Merge FSDP shards → HuggingFace ───────────────────────────
    print(
        f"[elastic] World-size mismatch: checkpoint has {ckpt_world_size} "
        f"ranks, new nproc={nproc}"
    )
    print(f"[elastic] Merging FSDP checkpoint → HF format …")

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    merge_config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=ckpt_dir,
        target_dir=hf_path,
        hf_model_config_path=hf_path,  # config.json already saved here
        trust_remote_code=True,
    )
    merger = FSDPModelMerger(merge_config)
    merger.merge_and_save()
    print(f"[elastic] Merged checkpoint saved to {hf_path}")
    return hf_path


def _run_verl_sft(args, config: ProjectConfig, train_file: str):
    if args.no_elastic:
        _run_verl_sft_static(args, config, train_file)
    else:
        _run_verl_sft_elastic(args, config, train_file)


def _run_verl_sft_static(args, config: ProjectConfig, train_file: str):
    """Fixed-GPU SFT launch (no elastic scaling)."""
    from model_worker import device_scope

    with device_scope(
        min_devices=max(args.nproc_per_node, args.min_devices),
        max_devices=args.max_devices,
        warmup=True,
        poll_interval=30.0,
        verbose=True,
    ) as slot:
        nproc = str(slot.count)
        slot.release()
        proc = _launch_verl_sft(args, config, train_file, nproc)
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, proc.args)


def _run_verl_sft_elastic(args, config: ProjectConfig, train_file: str):
    """Elastic SFT — restart with more GPUs when they appear after checkpoints."""
    from model_worker.elastic import elastic_run

    checkpoint_dir = os.path.abspath(args.output_dir)
    checkpoint_indicator = os.path.join(
        checkpoint_dir, "latest_checkpointed_iteration.txt"
    )

    def launch_fn(nproc: int, cuda_visible: str) -> subprocess.Popen:
        # Check if we need to convert the checkpoint for a world-size change.
        model_path_override = None
        resume_mode = "auto"
        hf_path = _merge_fsdp_checkpoint_if_needed(checkpoint_dir, nproc)
        if hf_path is not None:
            model_path_override = os.path.abspath(hf_path)
            # Use "metadata_only" so verl restores the step counter and
            # dataloader position from the checkpoint WITHOUT trying to
            # load the (incompatible) FSDP model shards — the model
            # weights are already supplied via model_path_override.
            resume_mode = "metadata_only"
            print(
                f"[elastic] Restarting from merged HF weights at "
                f"{model_path_override} (resume_mode=metadata_only)"
            )
        return _launch_verl_sft(
            args, config, train_file, str(nproc),
            model_path_override=model_path_override,
            resume_mode=resume_mode,
        )

    rc = elastic_run(
        launch_fn=launch_fn,
        checkpoint_indicator=checkpoint_indicator,
        min_devices=max(args.nproc_per_node, args.min_devices),
        max_devices=args.max_devices,
        check_interval=args.elastic_check_interval,
        warmup=True,
        verbose=True,
    )
    if rc != 0:
        raise RuntimeError(f"SFT training failed with exit code {rc}")


def _launch_verl_sft(
    args,
    config: ProjectConfig,
    train_file: str,
    nproc: str,
    *,
    model_path_override: Optional[str] = None,
    resume_mode: str = "auto",
):
    model_path = model_path_override or config.grm_model_name
    # verl SFT requires distributed env; torchrun handles this even for single GPU.
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "verl.trainer.sft_trainer",
        "--config-name",
        "sft_trainer_engine",
        f"model.path={model_path}",
        "model.trust_remote_code=True",
        "+model.override_config.attn_implementation=sdpa",
        "engine.model_dtype=bf16",
        f"data.train_files=['{train_file}']",
        "data.val_files=null",
        "data.custom_cls.path=pkg://src.training.weighted_sft_dataset",
        "data.custom_cls.name=WeightedRubricSFTDataset",
        f"data.max_length={args.max_length}",
        "data.truncation=left",
        "data.pad_mode=no_padding",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"+data.point_alpha={args.point_alpha}",
        f"+data.negative_boost={args.negative_boost}",
        f"+data.min_weight={args.min_weight}",
        f"+data.max_weight={args.max_weight}",
        f"+data.sft_instruction_template={json.dumps(RUBRIC_GENERATION_PROMPT)}",
        f"optim.lr={args.lr}",
        f"optim.lr_scheduler_type={args.lr_scheduler_type}",
        f"optim.lr_warmup_steps_ratio={args.lr_warmup_ratio}",
        f"optim.min_lr_ratio={args.min_lr_ratio}",
        f"trainer.total_epochs={args.epochs}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.n_gpus_per_node={int(nproc)}",
        "trainer.nnodes=1",
        f"trainer.save_freq={args.save_freq}",
        f"trainer.test_freq={args.test_freq}",
        "trainer.logger=['console']",
        f"trainer.default_local_dir={args.output_dir}",
        f"trainer.resume_mode={resume_mode}",
        "engine.use_torch_compile=False",
    ]

    print("Launching verl SFT:")
    print(" ".join(cmd))
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    return subprocess.Popen(cmd, env=env)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="verl-based weighted SFT using HealthBench + synthetic data")
    parser.add_argument("--no-healthbench", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--healthbench-limit", type=int, default=None, help="Total number of HealthBench SFT rows to include")
    parser.add_argument("--healthbench-limit-per-file", type=int, default=None)
    parser.add_argument("--synthetic-path", type=str, default=None)
    parser.add_argument("--synthetic-limit", type=int, default=None)
    parser.add_argument("--output-jsonl", type=str, default=None)

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--point-alpha", type=float, default=0.08)
    parser.add_argument("--negative-boost", type=float, default=1.3)
    parser.add_argument("--min-weight", type=float, default=0.2)
    parser.add_argument("--max-weight", type=float, default=3.0)

    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.05, help="Warmup steps as fraction of total steps")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Min LR as fraction of peak LR (for cosine)")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--min-devices", type=int, default=2, help="Minimum GPU count for elastic scaling")
    parser.add_argument("--max-devices", type=int, default=8, help="Maximum GPU count for elastic scaling")
    parser.add_argument("--no-elastic", action="store_true", help="Disable elastic GPU scaling (fixed device count)")
    parser.add_argument("--elastic-check-interval", type=float, default=60.0, help="Seconds between GPU expansion checks")

    parser.add_argument("--project-name", type=str, default="grm-sft")
    parser.add_argument("--experiment-name", type=str, default="healthbench_weighted")
    parser.add_argument("--output-dir", type=str, default="out/sft")
    parser.add_argument("--save-freq", type=int, default=-1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare mixed SFT jsonl, do not launch training")
    parser.add_argument("--preview-samples", type=int, default=0, help="Print N prepared SFT samples before training")
    parser.add_argument("--healthbench-benchmark-ratio", type=float, default=0.2, help="Fraction of HealthBench prompts reserved for benchmark (excluded from SFT)")
    parser.add_argument("--healthbench-split-seed", type=int, default=42, help="Deterministic split seed for HealthBench SFT/benchmark")
    parser.add_argument("--rebuild-healthbench-split", action="store_true", help="Force rebuild HealthBench split files")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    config = ProjectConfig()
    train_file = _build_weighted_sft_jsonl(args, config)
    if args.prepare_only:
        print("prepare-only enabled, skip launching verl SFT.")
    else:
        _run_verl_sft(args, config, train_file)
