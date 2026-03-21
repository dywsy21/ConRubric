"""GLM-5 rubric generation only (no Judge scoring).
Generates all rubrics via API and saves to cache. Judge scoring is separate."""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "hub")

from src.data.prepare_healthbench import ensure_healthbench_splits
from src.models.api_grm import APIRubricGenerator


def main():
    output_dir = "out/bench_full/glm5"
    rubric_cache_path = os.path.join(output_dir, "rubric_cache.json")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    split_paths = ensure_healthbench_splits(
        output_dir="data/healthbench_splits",
        benchmark_ratio=0.2, seed=42, force_rebuild=False,
    )

    sft_prompt_ids = set()
    with open(split_paths.sft_train) as f:
        for line in f:
            sft_prompt_ids.add(json.loads(line).get("prompt_id"))

    grouped = defaultdict(list)
    with open(split_paths.benchmark_meta_eval) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("prompt_id")
            if pid in sft_prompt_ids:
                raise RuntimeError(f"Leakage: {pid}")
            grouped[pid].append(row)

    prompt_items = list(grouped.items())
    prompt_texts = {}
    for pid, rows in prompt_items:
        prompt = rows[0]["prompt"]
        if isinstance(prompt, list):
            parts = []
            for m in prompt:
                if isinstance(m, dict):
                    c = (m.get("content") or "").strip()
                    if c:
                        parts.append(c)
            prompt_texts[pid] = "\n".join(parts)
        else:
            prompt_texts[pid] = str(prompt)

    print(f"Total prompts: {len(prompt_items)}")

    # Load cached rubrics
    rubrics = {}
    if os.path.exists(rubric_cache_path):
        with open(rubric_cache_path) as f:
            rubrics = json.load(f)
        n_cached = sum(1 for v in rubrics.values() if v)
        print(f"Loaded {n_cached} cached rubrics")

    # Find missing
    missing = [(pid, prompt_texts[pid]) for pid, _ in prompt_items
               if pid not in rubrics or not rubrics[pid]]

    if not missing:
        print("All rubrics already cached!")
        return

    n_api_workers = 2
    print(f"Generating {len(missing)} rubrics via GLM-5 API ({n_api_workers} workers)...")

    grm = APIRubricGenerator(
        api_base="https://open.bigmodel.cn/api/paas/v4",
        api_key="e7556592ec324ef7b7bb65ddac18b108.2fAWr5h24Jy2cpuN",
        model="glm-5",
        max_workers=n_api_workers,
        max_tokens=4096,
        thinking=True,
    )

    pbar = tqdm(total=len(missing), desc="GLM-5 rubrics")
    with ThreadPoolExecutor(max_workers=n_api_workers) as pool:
        futures = {}
        for pid, text in missing:
            fut = pool.submit(grm.generate_rubric, text)
            futures[fut] = pid

        done = 0
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                rubric = fut.result()
                rubrics[pid] = rubric
            except Exception as e:
                print(f"Error for {pid}: {e}")
                rubrics[pid] = ""

            done += 1
            pbar.update(1)

            # Incremental save every 25
            if done % 25 == 0:
                with open(rubric_cache_path, "w") as f:
                    json.dump(rubrics, f, ensure_ascii=False)
                n_ok = sum(1 for v in rubrics.values() if v)
                pbar.set_postfix(cached=n_ok)

    pbar.close()

    # Final save
    with open(rubric_cache_path, "w") as f:
        json.dump(rubrics, f, ensure_ascii=False)
    n_ok = sum(1 for v in rubrics.values() if v)
    n_empty = sum(1 for v in rubrics.values() if not v)
    print(f"\nDone! {n_ok} valid rubrics, {n_empty} empty, total {len(rubrics)}")
    print(f"Saved to {rubric_cache_path}")


if __name__ == "__main__":
    main()
