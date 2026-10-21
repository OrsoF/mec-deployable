"""Train a fixed-topology PPO policy from exactly ten industrial scenario JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on exactly 10 JSON network instances"
    )
    parser.add_argument("--instances", type=Path, default=Path("instances/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_bottleneck"))
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episode-steps", type=int, default=96)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-mode", choices=("atomic", "hybrid"), default="hybrid")
    args = parser.parse_args()
    if args.timesteps < 1 or args.episode_steps < 1 or args.paths < 1:
        parser.error("timesteps, episode-steps and paths must be positive")

    import gymnasium
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from network_ppo.environment import NetworkPPOEnv
    from network_ppo.instance_io import load_instance
    from network_ppo.problem import validate_compatible

    files = sorted(args.instances.glob("*.json"))
    if len(files) != 10:
        parser.error(
            f"Expected exactly 10 *.json files in {args.instances}, found {len(files)}"
        )
    scenarios = [load_instance(path) for path in files]
    signature = validate_compatible(scenarios)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def make_env():
        return NetworkPPOEnv(
            scenarios,
            episode_steps=args.episode_steps,
            paths=args.paths,
            action_mode=args.action_mode,
        )

    checker = make_env()
    check_env(checker, warn=True, skip_render_check=True)
    checker.close()
    venv = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False)
    try:
        venv.seed(args.seed)
        model = PPO(
            "MlpPolicy",
            venv,
            seed=args.seed,
            n_steps=256,
            batch_size=64,
            ent_coef=0.01,
            verbose=1,
            device="cpu",
        )
        print(f"Training PPO on 10 instances for ~{args.timesteps} steps")
        model.learn(total_timesteps=args.timesteps)
        args.output.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output / "model.zip"))
        venv.save(str(args.output / "vecnormalize.pkl"))
        metadata = {
            "schema_version": 2,
            "action_mode": args.action_mode,
            "algorithm": "PPO / discrete feasible edits",
            "seed": args.seed,
            "timesteps_requested": args.timesteps,
            "timesteps_actual": model.num_timesteps,
            "episode_steps": args.episode_steps,
            "paths": args.paths,
            "training_instances": [path.name for path in files],
            "training_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files
            },
            "signature": signature,
            "versions": {
                "stable_baselines3": stable_baselines3.__version__,
                "gymnasium": gymnasium.__version__,
                "torch": torch.__version__,
            },
        }
        (args.output / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved checkpoint to {args.output}")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
