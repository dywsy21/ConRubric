import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

from huggingface_hub import hf_hub_download

HEALTHBENCH_REPO = "openai/healthbench"
RUBRIC_FILES = [
    "2025-05-07-06-14-12_oss_eval.jsonl",
    "consensus_2025-05-09-20-00-46.jsonl",
    "hard_2025-05-08-21-00-10.jsonl",
]
META_EVAL_FILE = "2025-05-07-06-14-12_oss_meta_eval.jsonl"


@dataclass
class HealthBenchSplitPaths:
    sft_train: str
    benchmark_meta_eval: str
    benchmark_prompt_pool: str


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


def _normalize_rubrics(raw_rubrics: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw_rubrics, list):
        return out
    for r in raw_rubrics:
        if isinstance(r, dict):
            criterion = str(r.get("criterion", "")).strip()
            if not criterion:
                continue
            points = int(r.get("points", 1))
            out.append({"criterion": criterion, "points": points})
        else:
            criterion = str(r).strip()
            if criterion:
                out.append({"criterion": criterion, "points": 1})
    return out


def _stable_bucket(prompt_id: str, seed: int = 42) -> int:
    h = hashlib.md5(f"{seed}:{prompt_id}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def ensure_healthbench_splits(
    output_dir: str = "data/healthbench_splits",
    benchmark_ratio: float = 0.2,
    seed: int = 42,
    force_rebuild: bool = False,
) -> HealthBenchSplitPaths:
    os.makedirs(output_dir, exist_ok=True)

    paths = HealthBenchSplitPaths(
        sft_train=os.path.join(output_dir, "sft_train.jsonl"),
        benchmark_meta_eval=os.path.join(output_dir, "benchmark_meta_eval.jsonl"),
        benchmark_prompt_pool=os.path.join(output_dir, "benchmark_prompt_pool.jsonl"),
    )

    if not force_rebuild and all(os.path.exists(p) for p in [paths.sft_train, paths.benchmark_meta_eval, paths.benchmark_prompt_pool]):
        return paths

    benchmark_cutoff = int(benchmark_ratio * 100)

    # 1) build prompt-id split from rubric-bearing files
    sft_rows = []
    benchmark_prompt_rows = []
    benchmark_prompt_ids = set()

    for file_name in RUBRIC_FILES:
        local_path = hf_hub_download(HEALTHBENCH_REPO, file_name, repo_type="dataset")
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt_id = str(item.get("prompt_id") or "").strip()
                if not prompt_id:
                    question_tmp = _prompt_to_text(item.get("prompt"))
                    prompt_id = hashlib.md5(question_tmp.encode()).hexdigest()

                question = _prompt_to_text(item.get("prompt"))
                rubrics = _normalize_rubrics(item.get("rubrics", []))
                if not question or not rubrics:
                    continue

                # Extract ideal_completion from ideal_completions_data
                icd = item.get("ideal_completions_data") or {}
                ideal_completion = (icd.get("ideal_completion") or "").strip() if isinstance(icd, dict) else ""

                row = {
                    "prompt_id": prompt_id,
                    "question": question,
                    "rubrics": rubrics,
                    "ideal_completion": ideal_completion,
                    "source": f"healthbench:{file_name}",
                }

                if _stable_bucket(prompt_id, seed=seed) < benchmark_cutoff:
                    benchmark_prompt_rows.append(row)
                    benchmark_prompt_ids.add(prompt_id)
                else:
                    sft_rows.append(row)

    # 2) meta-eval benchmark rows (only prompt_ids in benchmark set)
    meta_path = hf_hub_download(HEALTHBENCH_REPO, META_EVAL_FILE, repo_type="dataset")
    benchmark_meta_rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt_id = str(item.get("prompt_id") or "").strip()
            if prompt_id in benchmark_prompt_ids:
                benchmark_meta_rows.append(item)

    # write outputs
    _ensure_parent(paths.sft_train)
    with open(paths.sft_train, "w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(paths.benchmark_prompt_pool, "w", encoding="utf-8") as f:
        for row in benchmark_prompt_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(paths.benchmark_meta_eval, "w", encoding="utf-8") as f:
        for row in benchmark_meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"HealthBench split built. sft_train={len(sft_rows)}, "
        f"benchmark_prompt_pool={len(benchmark_prompt_rows)}, "
        f"benchmark_meta_eval={len(benchmark_meta_rows)}"
    )

    return paths
