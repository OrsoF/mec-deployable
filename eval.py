"""Evaluate a saved PPO checkpoint on one compatible industrial JSON instance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO placement, routing and quantities on one JSON instance")
    parser.add_argument("--instance", type=Path, default=Path("instances/eval/factory_shift_unseen.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ppo_factory"))
    parser.add_argument("--output", type=Path, default=Path("outputs/decision.json"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions; keep best over multiple episodes")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be >= 1")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from network_ppo.environment import NetworkPPOEnv
    from network_ppo.heuristic import greedy_uplift
    from network_ppo.instance_io import load_instance
    from network_ppo.problem import validate_compatible
    from network_ppo.solution_io import decision_json

    metadata = json.loads((args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint metadata version")
    scenario = load_instance(args.instance)
    validate_compatible([scenario], reference=metadata["signature"])
    holder = {}

    def make_env():
        env = NetworkPPOEnv(scenario, episode_steps=int(metadata["episode_steps"]),
                            paths=int(metadata["paths"]))
        holder["env"] = env
        return env

    venv = VecNormalize.load(str(args.checkpoint / "vecnormalize.pkl"), DummyVecEnv([make_env]))
    try:
        venv.training = False   # do not update normalization statistics at evaluation time
        venv.norm_reward = False
        venv.seed(args.seed)
        model = PPO.load(str(args.checkpoint / "model.zip"), env=venv, device="cpu")
        model.set_random_seed(args.seed)
        steps = 0
        for episode in range(args.episodes):
            obs = venv.reset()
            for _ in range(int(metadata["episode_steps"])):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, _, done, _ = venv.step(action)
                steps += 1
                if done[0]:
                    break
        env = holder["env"]
        best = env.best_overall
        if best is None:
            raise RuntimeError("No feasible solution generated")
        result = decision_json(scenario, best, env.catalogue,
                               checkpoint=str(args.checkpoint), policy_steps=steps, seed=args.seed)
        reference = greedy_uplift(scenario, env.catalogue, env.seeds[0])
        ref_mbps = int(reference.quantities.sum())
        result.update({
            "algorithm": "PPO (heuristic feasible seed, policy refinement)",
            "stochastic_actions": args.stochastic,
            "reference_seed_greedy_mbps": ref_mbps,
            "ppo_minus_reference_mbps": result["total_served_mbps"] - ref_mbps,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Instance: {scenario.id} | Feasible: {result['feasible']}")
        print(f"PPO: {result['total_served_mbps']}/{result['max_requested_mbps']} Mbps; "
              f"seed-greedy reference: {ref_mbps} Mbps")
        for p in result["placements"]:
            print(f"  {p['service']} -> {p['node']}")
        print(f"Decision saved to: {args.output}")
    finally:
        venv.close()


if __name__ == "__main__":
    main()
