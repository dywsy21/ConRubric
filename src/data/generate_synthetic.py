import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables before importing other libraries
load_dotenv()

from datasets import load_dataset
from tqdm import tqdm
from src.evaluation.judge import Oracle
from src.config import DataConfig, ProjectConfig

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
        if not question: # Try 'input' if instruction is empty or combined
             question = sample.get('input', "")
    elif "roleplay" in dataset_name:
        # Simplified handling for roleplay, might need adjustment based on actual columns
        # Assuming 'text' field or similar, but let's skip if structure is complex for now
        # Or try to find 'instruction' / 'response' if mapped
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

def generate_synthetic_data(limit: int = None):
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
            print(f"Processing dataset: {dataset_name}. Need {needed} more samples.")

            try:
                # Load a subset to save time/memory if needed, or stream
                ds = load_dataset(dataset_name, split="train", streaming=True)
                
                count = 0
                for sample in tqdm(ds):
                    if count >= needed:
                        break
                        
                    question, gold_answer = extract_qa(sample, dataset_name)
                    
                    if not question or not gold_answer:
                        continue
                        
                    # Reverse engineer rubric
                    rubric = oracle.reverse_engineer_rubric(question, gold_answer)
                    
                    if rubric:
                        record = {
                            "source": dataset_name,
                            "question": question,
                            "gold_answer": gold_answer,
                            "rubric": rubric
                        }
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        count += 1
                        total_samples += 1
                        
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")

    print(f"Done. Total samples: {total_samples}. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    generate_synthetic_data(limit=args.limit)
