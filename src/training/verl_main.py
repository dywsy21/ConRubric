import os
import subprocess
import sys

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


def main_elastic():
    """Elastic RL training with auto GPU scaling.

    Wraps ``verl_main.py`` in an elastic loop: acquires devices, spawns
    training as a child process, and monitors for GPU expansion after
    each checkpoint.  The child is re-invoked with ``_ELASTIC_CHILD=1``
    to skip the elastic wrapper and go straight to Hydra / ``run_ppo``.

    Configuration via environment variables:
      - ``ELASTIC_TRAINING=0``      — disable elastic (default: 1)
      - ``RL_MIN_DEVICES=2``        — minimum GPU count
      - ``RL_MAX_DEVICES=8``        — maximum GPU count
      - ``RL_CHECKPOINT_DIR=out/rl`` — checkpoint directory
      - ``ELASTIC_CHECK_INTERVAL=60`` — probe interval (seconds)
    """
    from model_worker.elastic import elastic_run

    rl_output_dir = os.getenv("RL_CHECKPOINT_DIR", "out/rl")
    checkpoint_indicator = os.path.join(
        rl_output_dir, "latest_checkpointed_iteration.txt"
    )

    # Forward all CLI args (Hydra overrides) to child processes.
    hydra_args = sys.argv[1:]

    def launch_fn(nproc: int, cuda_visible: str) -> subprocess.Popen:
        cmd = [
            sys.executable, "-u", "-m", "src.training.verl_main",
            *hydra_args,
            # Override GPU count and enable resume (placed AFTER user args
            # so they always take effect regardless of user overrides).
            f"trainer.n_gpus_per_node={nproc}",
            "trainer.resume_mode=auto",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
        env["_ELASTIC_CHILD"] = "1"
        return subprocess.Popen(cmd, env=env)

    rc = elastic_run(
        launch_fn=launch_fn,
        checkpoint_indicator=checkpoint_indicator,
        min_devices=int(os.getenv("RL_MIN_DEVICES", "2")),
        max_devices=int(os.getenv("RL_MAX_DEVICES", "8")),
        check_interval=float(os.getenv("ELASTIC_CHECK_INTERVAL", "60")),
        verbose=True,
    )
    sys.exit(rc)


if __name__ == "__main__":
    if os.getenv("_ELASTIC_CHILD") == "1" or os.getenv("ELASTIC_TRAINING", "1") == "0":
        # Child process — run training directly via Hydra.
        main()
    else:
        # Parent process — elastic GPU scaling wrapper.
        main_elastic()
