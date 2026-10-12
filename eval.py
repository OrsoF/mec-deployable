"""python eval.py --instance instances/eval/factory_shift_unseen.json"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Use a PPO checkpoint to route/place/allocate a JSON case")
    parser.add_argument("--instance", type=Path, default=Path("instances/eval/factory_shift_unseen.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ppo_factory"))
    parser.add_argument("--output", type=Path, default=Path("outputs/decision.json"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true", help="Sample PPO actions (best of several rollouts)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be >= 1")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from mec_ppo.env import NetworkPPOEnv
    from mec_ppo.instances import load_instance, validate_compatible
    from mec_ppo.results import format_result
    from mecbench.solvers.fast_heuristic import Solver as FastHeuristic

    checkpoint = args.checkpoint
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf8"))
    case = load_instance(args.instance)
    validate_compatible([case], reference=metadata["signature"])

    def new_env():
        return NetworkPPOEnv(
            problem=case.problem,
            max_step=metadata["episode_steps"],
            max_path_candidates=metadata["paths"],
            warm_start_heuristic=True,
            reuse_best_on_reset=False,
        )

    venv = VecNormalize.load(str(checkpoint / "vecnormalize.pkl"), DummyVecEnv([new_env]))
    venv.training = False
    venv.norm_reward = False
    venv.seed(args.seed)
    model = PPO.load(str(checkpoint / "model.zip"), env=venv, device="cpu")
    model.set_random_seed(args.seed)
    obs = venv.reset()
    total_steps = args.episodes * int(metadata["episode_steps"])
    for _ in range(total_steps):
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, _, done, _ = venv.step(action)
        # DummyVecEnv automatically resets when done; best feasible state is retained.
    env = venv.venv.envs[0]
    best = env.best_state_record[0]
    if best is None:
        raise RuntimeError("No feasible solution found. Check scenario feasibility / baselines.")
    solution = env.solution_from_state(best)
    result = format_result(case, solution, env.path_cache.nth_shortest_path,
                           checkpoint=checkpoint, steps=total_steps, seed=args.seed)
    reference = FastHeuristic(case.problem, limit_shortest_path=metadata["paths"],
                              seed=args.seed).run()
    heuristic_score = None if reference.solution is None else float(reference.solution.quantities.sum())
    result["algorithm"] = "PPO with feasible heuristic initialization"
    result["stochastic_actions"] = args.stochastic
    result["reference_fast_heuristic_mbps"] = heuristic_score
    if heuristic_score is not None:
        result["ppo_minus_heuristic_mbps"] = result["total_served_mbps"] - heuristic_score
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    venv.close()
    print(f"Feasible: {result['feasible']}; served: {result['total_served_mbps']} / "
          f"{result['max_requested_mbps']} Mbps")
    print(f"Heuristic reference: {heuristic_score} Mbps (reported, not substituted)")
    print(f"Decision written to: {args.output}")
    for row in result["placements"]:
        print(f"  {row['service']} -> {row['node']}")


if __name__ == "__main__":
    main()
