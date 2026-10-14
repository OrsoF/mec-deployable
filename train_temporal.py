"""Train a *one-decision-per-period* PPO policy on temporal TRAIN split only."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=Path, default=Path("instances/temporal/train"))
    ap.add_argument("--output", type=Path, default=Path("outputs/temporal/seed_42"))
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paths", type=int, default=3)
    ap.add_argument("--forecast-steps", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=256)
    args = ap.parse_args()
    if min(args.timesteps,args.paths,args.forecast_steps,args.n_steps) < 1:
        ap.error("timesteps, paths, forecast-steps, n-steps must be positive")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.env_checker import check_env
    from network_ppo.dynamics import load_temporal_scenario
    from network_ppo.problem import validate_compatible
    from network_ppo.temporal_env import TemporalNetworkEnv
    import stable_baselines3, gymnasium, torch

    paths = sorted(args.scenarios.glob("*.json"))
    if not paths or args.scenarios.name != "train":
        ap.error("Train only on a nonempty instances/temporal/train directory")
    cases = [load_temporal_scenario(p) for p in paths]
    if any(s.split != "train" for s in cases):
        ap.error("Training data must be labeled split=train")
    signature = validate_compatible([s.base for s in cases])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def make_env():
        return TemporalNetworkEnv(cases, paths=args.paths, forecast_steps=args.forecast_steps)

    checker=make_env()
    check_env(checker, warn=True, skip_render_check=True)
    checker.close()
    env=VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False)
    try:
        env.seed(args.seed)
        model=PPO("MlpPolicy", env, seed=args.seed, n_steps=args.n_steps,
                  batch_size=min(64,args.n_steps), ent_coef=0.01, verbose=1, device="cpu")
        model.learn(total_timesteps=args.timesteps)
        args.output.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output/"model.zip"))
        env.save(str(args.output/"vecnormalize.pkl"))
        metadata={
            "schema_version":3, "mode":"temporal", "seed":args.seed,
            "timesteps_requested":args.timesteps, "timesteps_actual":model.num_timesteps,
            "paths":args.paths, "forecast_steps":args.forecast_steps,
            "signature":signature, "training_split":"train",
            "training_instances":[p.name for p in paths],
            "training_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "versions":{"stable_baselines3":stable_baselines3.__version__,
                        "gymnasium":gymnasium.__version__, "torch":torch.__version__},
        }
        (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
        print(f"Wrote temporal PPO checkpoint {args.output}")
    finally:
        env.close()


if __name__=="__main__":
    main()
