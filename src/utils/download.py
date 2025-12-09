import os
import argparse
from huggingface_hub import snapshot_download
from datasets import load_dataset
from src.config import ProjectConfig, DataConfig

def download_models(project_config: ProjectConfig):
    models_to_download = set()
    models_to_download.add(project_config.grm_model_name)
    for model in project_config.solver_model_names:
        models_to_download.add(model)
    
    print(f"Downloading {len(models_to_download)} models...")
    
    for model_name in models_to_download:
        print(f"Downloading model: {model_name}")
        try:
            snapshot_download(repo_id=model_name)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")

def download_datasets(data_config: DataConfig):
    print(f"Downloading {len(data_config.dataset_names)} datasets...")
    
    for dataset_name in data_config.dataset_names:
        print(f"Downloading dataset: {dataset_name}")
        try:
            # Just loading it will trigger download and caching
            load_dataset(dataset_name, split="train", streaming=False) 
            print(f"Successfully downloaded {dataset_name}")
        except Exception as e:
            print(f"Error downloading dataset {dataset_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download all necessary models and datasets.")
    parser.add_argument("--models", action="store_true", help="Download models only")
    parser.add_argument("--datasets", action="store_true", help="Download datasets only")
    args = parser.parse_args()
    
    # If neither is specified, download both
    download_all = not (args.models or args.datasets)
    
    project_config = ProjectConfig()
    data_config = DataConfig()
    
    if download_all or args.models:
        download_models(project_config)
        
    if download_all or args.datasets:
        download_datasets(data_config)

if __name__ == "__main__":
    main()
