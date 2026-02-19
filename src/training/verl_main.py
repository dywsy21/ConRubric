import os

import hydra
from dotenv import load_dotenv

from verl.trainer.main_ppo import run_ppo

# Load .env first so Hydra can interpolate env vars in yaml.
load_dotenv()

def _resolve_config_name() -> str:
    algo = os.getenv("RL_ALGORITHM", "dapo").strip().lower()
    if algo == "grpo":
        return "grpo_trainer"
    if algo == "ppo":
        return "ppo_trainer"
    return "dapo_trainer"


CONFIG_NAME = _resolve_config_name()


@hydra.main(config_path="config", config_name=CONFIG_NAME, version_base=None)
def main(config):
    print(f"Loading configuration: {CONFIG_NAME}")
    run_ppo(config)


if __name__ == "__main__":
    main()
