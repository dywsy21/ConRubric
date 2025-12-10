import os
import socket
import ray
import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
from dotenv import load_dotenv

# Load env vars BEFORE any other imports that might use them
load_dotenv()

# Set HF_HOME to absolute path if relative
hf_home = os.getenv("HF_HOME", "./data")
if not os.path.isabs(hf_home):
    hf_home = os.path.abspath(hf_home)
    os.environ["HF_HOME"] = hf_home
print(f"HF_HOME set to: {hf_home}")

from src.config import ProjectConfig
from src.training.meta_reward import MetaRewardFunction

# verl imports
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RewardModelWorker
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

class MetaRewardManager:
    def __init__(self, tokenizer, project_config: ProjectConfig):
        self.tokenizer = tokenizer
        self.project_config = project_config
        self.reward_fn = MetaRewardFunction(
            solver_model_name=project_config.solver_model_name,
            oracle_model_name=project_config.oracle_model_name,
            oracle_api_key=project_config.oracle_api_key,
            oracle_api_base=project_config.oracle_api_base,
            solver_remote=project_config.solver_remote,
            solver_api_key=project_config.solver_api_key,
            solver_api_base=project_config.solver_api_base
        )

    def __call__(self, data: DataProto, return_dict=False):
        # Decode prompts and responses
        # data.batch['prompts'] and data.batch['responses'] are tensors
        
        prompts_list = []
        responses_list = []
        
        # We need to extract valid tokens based on attention mask or just decode
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            response_ids = data_item.batch['responses']
            # response_ids might contain padding
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # prompt_ids might be padded too
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            prompts_list.append(prompt_str)
            responses_list.append(response_str)
            
        # Compute rewards
        # MetaRewardFunction expects list of questions and rubrics (responses)
        rewards_tensor = self.reward_fn.compute_reward(prompts_list, responses_list)
        
        if return_dict:
            return {'reward_tensor': rewards_tensor}
        return rewards_tensor

def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset."""
    if not isinstance(data_paths, (list, ListConfig)):
        data_paths = [data_paths]

    dataset = RLHFDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset

def create_rl_sampler(data_config, dataset):
    """Create a sampler."""
    from torch.utils.data import RandomSampler, SequentialSampler
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler

# Determine config name based on env var
config_name = "ppo_trainer"
if os.getenv("RL_ALGORITHM", "").lower() == "dapo":
    config_name = "dapo_trainer"

@hydra.main(config_path="config", config_name=config_name, version_base=None)
def main(config: DictConfig):
    print(f"Loading configuration: {config_name}")
    project_config = ProjectConfig()
    
    if not ray.is_initialized():
        # Get the current Python executable path to pass to Ray workers
        import sys
        python_executable = sys.executable
        
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true", 
                    "NCCL_DEBUG": "WARN",
                    "HF_HOME": hf_home,
                    "HF_HUB_CACHE": os.path.join(hf_home, "hub"),
                    "HUGGINGFACE_HUB_CACHE": os.path.join(hf_home, "hub"),
                    "TRANSFORMERS_CACHE": os.path.join(hf_home, "hub"),
                },
                # Use pip instead of uv to prevent Ray from creating new venv
                "pip": [],
            },
            num_cpus=config.trainer.get("num_cpus", os.cpu_count())
        )

    # Resolve config
    OmegaConf.resolve(config)

    # Hack to remove tensor_model_parallel_size from critic.model if present
    if 'critic' in config and 'model' in config.critic:
        if 'tensor_model_parallel_size' in config.critic.model:
            del config.critic.model.tensor_model_parallel_size

    # Convert critic.optim to a FSDPOptimizerConfig-like object that supports attribute access
    # This is needed because build_optimizer expects config.optim.lr, config.optim.weight_decay, etc.
    if 'critic' in config and 'optim' in config.critic:
        from omegaconf import open_dict
        with open_dict(config):
            if '_target_' not in config.critic.optim:
                config.critic.optim._target_ = 'verl.workers.config.optimizer.FSDPOptimizerConfig'

    # Tokenizer
    local_path = config.actor_rollout_ref.model.path
    # In a real setup, we might need to download model to local path first if it's a repo ID
    # But hf_tokenizer handles repo ID too usually
    tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    processor = None # hf_processor(local_path) if multimodal

    # Define Worker Classes
    # Assuming FSDP for now as per config
    actor_rollout_cls = ActorRolloutRefWorker
    critic_cls = CriticWorker
    ray_worker_group_cls = RayWorkerGroup

    # Role Mapping
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(critic_cls),
    }

    # Resource Pool
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    
    use_kl_in_reward = config.algorithm.get("use_kl_in_reward", False)
    use_kl_loss = config.actor_rollout_ref.actor.get("use_kl_loss", False)
    
    if use_kl_in_reward or use_kl_loss:
         role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
         mapping[Role.RefPolicy] = global_pool_id

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # Reward Manager
    reward_manager = MetaRewardManager(tokenizer, project_config)
    # We use the same reward manager for validation for now
    val_reward_manager = MetaRewardManager(tokenizer, project_config)

    # Datasets
    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    
    val_dataset = None
    if config.data.val_files:
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    
    # Sampler
    train_sampler = create_rl_sampler(config.data, train_dataset)

    # Initialize Trainer
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_manager,
        val_reward_fn=val_reward_manager,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.get("device", "cuda"),
    )

    trainer.init_workers()
    trainer.fit()

if __name__ == "__main__":
    main()
