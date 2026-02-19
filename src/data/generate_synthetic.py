import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables before importing other libraries
load_dotenv()

from datasets import load_dataset
from tqdm import tqdm
from src.evaluation.judge import Oracle
from src.config import DataConfig, ProjectConfig

# Parallel workers for Oracle API calls during synthetic generation
DEFAULT_ORACLE_WORKERS = int(os.environ.get("GRM_ORACLE_WORKERS", 8))


def extract_qa(sample, dataset_name):
    question = ""
    gold_answer = ""
    
    if "no_robots" in dataset_name:
        messages = sample.get('messages', [])
        for msg in messages:
            if msg['role'] == 'user':
                question = msg['content']
            elif msg['role'] == 'assistant':
                gold_answer = msg['content']
    elif "open-instruct" in dataset_name:
        question = sample.get('instruction', "")
        gold_answer = sample.get('output', "")
        if not question:
             question = sample.get('input', "")
    elif "roleplay" in dataset_name:
        pass
    elif "OpenMathInstruct" in dataset_name:
        question = sample.get('question', "")
        gold_answer = sample.get('generated_solution', "")
        if not gold_answer:
            gold_answer = sample.get('expected_answer', "")
    elif "Evol-Instruct-Code" in dataset_name:
        question = sample.get('instruction', "")
        gold_answer = sample.get('output', "")
        
    return question, gold_answer


def _process_one(oracle: Oracle, question: str, gold_answer: str, dataset_name: str):
    """Reverse-engineer a rubric for a single (Q, A) pair. Thread-safe."""
    rubric = oracle.reverse_engineer_rubric(question, gold_answer)
    if rubric:
        return {
            "source": dataset_name,
            "question": question,
            "gold_answer": gold_answer,
            "rubric": rubric,
        }
    return None


def generate_synthetic_data(limit: int = None, max_workers: int = None):
    if max_workers is None:
        max_workers = DEFAULT_ORACLE_WORKERS

    config = DataConfig()
    project_config = ProjectConfig()
    
    oracle = Oracle(
        model_name=project_config.oracle_model_name,
        api_key=project_config.oracle_api_key,
        api_base=project_config.oracle_api_base
    )
    output_path = os.path.join(project_config.data_dir, "synthetic_rubrics.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_samples = 0
    
    # Check existing progress
    existing_counts = {}
    if os.path.exists(output_path):
        print(f"Found existing output file at {output_path}. Scanning for progress...")
        with open(output_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    src = data.get("source")
                    if src:
                        existing_counts[src] = existing_counts.get(src, 0) + 1
                except:
                    pass
        print(f"Existing progress: {existing_counts}")
        total_samples = sum(existing_counts.values())

    with open(output_path, "a") as f:
        for dataset_name in config.dataset_names:
            current_limit = limit if limit else (config.num_samples // len(config.dataset_names))
            already_processed = existing_counts.get(dataset_name, 0)
            
            if already_processed >= current_limit:
                print(f"Skipping {dataset_name}: already have {already_processed} samples (limit: {current_limit})")
                continue
                
            needed = current_limit - already_processed
            print(f"Processing dataset: {dataset_name}. Need {needed} more samples. (workers={max_workers})")

            try:
                ds = load_dataset(dataset_name, split="train", streaming=True)
                
                # Collect candidate (question, gold_answer) pairs from the stream
                candidates = []
                for sample in ds:
                    if len(candidates) >= needed:
                        break
                    question, gold_answer = extract_qa(sample, dataset_name)
                    if question and gold_answer:
                        candidates.append((question, gold_answer))

                if not candidates:
                    print(f"  No valid Q/A pairs found in {dataset_name}, skipping.")
                    continue

                # Process candidates in parallel using Oracle workers
                count = 0
                pbar = tqdm(total=needed, desc=f"  {dataset_name.split('/')[-1]}")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_process_one, oracle, q, a, dataset_name): (q, a)
                        for q, a in candidates
                    }
                    for future in as_completed(futures):
                        try:
                            record = future.result()
                            if record is not None:
                                f.write(json.dumps(record) + "\n")
                                f.flush()
                                count += 1
                                total_samples += 1
                                pbar.update(1)
                        except Exception as e:
                            print(f"  Worker error: {e}")
                pbar.close()
                print(f"  {dataset_name}: wrote {count} samples")

            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")

    print(f"Done. Total samples: {total_samples}. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    generate_synthetic_data(limit=args.limit)
