"""python train.py --instances instances/train --output outputs/run1 --timesteps 100000"""
from __future__ import annotations

import argparse
import json
import hashlib
import random
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one PPO policy on exactly ten MEC instances")
    parser.add_argument("--instances", type=Path, default=Path("instances/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_factory"))
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episode-steps", type=int, default=96)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.timesteps < 1 or args.episode_steps < 1 or args.paths < 1:
        parser.error("timesteps, episode-steps and paths must be positive")

    from mec_ppo.env import NetworkPPOEnv
    from mec_ppo.instances import load_instance, validate_compatible
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import stable_baselines3
    import gymnasium
    import torch

    files = sorted(args.instances.glob("*.json"))
    if len(files) != 10:
        parser.error(f"Expected exactly 10 JSON files in {args.instances}, found {len(files)}")
    cases = [load_instance(p) for p in files]
    signature = validate_compatible(cases)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def new_env():
        return NetworkPPOEnv(
            problem=[case.problem for case in cases],
            max_step=args.episode_steps,
            max_path_candidates=args.paths,
            warm_start_heuristic=True,
            reuse_best_on_reset=False,
        )

    env_check = new_env()
    check_env(env_check, warn=True, skip_render_check=True)
    env_check.close()
    venv = VecNormalize(DummyVecEnv([new_env]), norm_obs=True, norm_reward=True)
    venv.seed(args.seed)
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=1, n_steps=512,
                batch_size=64, ent_coef=0.01, device="auto")
    print(f"Training PPO on {len(cases)} cases; requested steps: {args.timesteps}")
    model.learn(total_timesteps=args.timesteps)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.zip")
    venv.save(str(args.output / "vecnormalize.pkl"))
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "timesteps_requested": args.timesteps,
        "timesteps_actual": model.num_timesteps,
        "episode_steps": args.episode_steps,
        "paths": args.paths,
        "training_instances": [str(p) for p in files],
        "training_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "signature": signature,
        "versions": {
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": torch.__version__,
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    venv.close()
    print(f"Saved model, normalization statistics and metadata to {args.output}")


if __name__ == "__main__":
    main()
