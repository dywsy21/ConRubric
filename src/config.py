import os
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

def get_list_from_env(key: str, default: List[str]) -> List[str]:
    val = os.getenv(key)
    if val:
        return [x.strip() for x in val.split(",") if x.strip()]
    return default

@dataclass
class ProjectConfig:
    # Paths
    data_dir: str = os.getenv("DATA_DIR", "data")
    output_dir: str = os.getenv("OUTPUT_DIR", "output")
    
    # Models
    grm_model_name: str = os.getenv("GRM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") # Using a stable public model as base
    
    # Solver Pool (for rotation to maximize generalization)
    solver_model_names: List[str] = field(default_factory=lambda: get_list_from_env("SOLVER_MODEL_NAMES", [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-2-2b-it",
        "microsoft/Phi-3.5-mini-instruct"
    ]))
    solver_model_name: str = os.getenv("SOLVER_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") # Default/Active solver
    
    # Oracle / Judge
    oracle_model_name: str = os.getenv("ORACLE_MODEL_NAME", "gpt-4o")
    oracle_api_key: Optional[str] = os.getenv("ORACLE_API_KEY")
    oracle_api_base: Optional[str] = os.getenv("ORACLE_API_BASE")
    
    # Training
    seed: int = int(os.getenv("SEED", "42"))
    
@dataclass
class DataConfig:
    # Datasets to mix for diverse training
    dataset_names: List[str] = field(default_factory=lambda: get_list_from_env("DATASET_NAMES", [
        "HuggingFaceH4/no_robots",      # High quality general instructions
        "hieunguyenminh/roleplay",      # Roleplay and creative writing
        "hakurei/open-instruct-v1",     # Diverse instruction following
        "nvidia/OpenMathInstruct-1",    # Math (Deterministic)
        "nickrosh/Evol-Instruct-Code-80k-v1" # Code (Deterministic)
    ]))
    dataset_name: str = os.getenv("DATASET_NAME", "HuggingFaceH4/no_robots") # Default
    
    num_samples: int = int(os.getenv("NUM_SAMPLES", "5000"))
    output_file: str = os.getenv("OUTPUT_FILE", "data/synthetic_rubrics.jsonl")

@dataclass
class RLConfig:
    algorithm: str = os.getenv("RL_ALGORITHM", "dapo")
    total_steps: int = int(os.getenv("RL_TOTAL_STEPS", "1000")) 
