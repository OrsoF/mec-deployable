"""Evaluate PPO on a held-out scenario, without silently replacing it with a baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from network_ppo.inference import DecisionEngine
from network_ppo.instance_io import load_instance
from network_ppo.heuristic import solve_baseline
from network_ppo.paths import PathCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path,
                        default=Path("instances/eval/bottleneck_unseen.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ppo_bottleneck"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_decision.json"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    case = load_instance(args.instance)
    engine = DecisionEngine(strategy="ppo", checkpoint=args.checkpoint,
                            episodes=args.episodes, stochastic=args.stochastic,
                            seed=args.seed)
    paths = int(engine.runtime.metadata["paths"])
    engine.paths = paths
    result = engine.decide(case)
    catalog = PathCatalog(case, paths)
    result["reference_greedy_mbps"] = int(solve_baseline(case, catalog, "greedy").quantities.sum())
    result["reference_lookahead_mbps"] = int(solve_baseline(case, catalog, "lookahead").quantities.sum())
    result["ppo_minus_greedy_mbps"] = result["total_served_mbps"] - result["reference_greedy_mbps"]
    result["ppo_minus_lookahead_mbps"] = result["total_served_mbps"] - result["reference_lookahead_mbps"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PPO {result['total_served_mbps']:.0f} Mbps | greedy "
          f"{result['reference_greedy_mbps']} | lookahead "
          f"{result['reference_lookahead_mbps']} | feasible={result['feasible']}")


if __name__ == "__main__":
    main()
