"""File-based deployment: validated JSON instance -> validated JSON decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from network_ppo.inference import DecisionEngine
from network_ppo.instance_io import load_instance


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", choices=("greedy", "lookahead", "multistart", "ppo", "auto"),
                        default="lookahead")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=123)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path,
                        default=Path("instances/eval/bottleneck_unseen.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/decision.json"))
    add_engine_args(parser)
    args = parser.parse_args()
    engine = DecisionEngine(strategy=args.strategy, checkpoint=args.checkpoint,
                            paths=args.paths, episodes=args.episodes,
                            stochastic=args.stochastic, seed=args.seed)
    decision = engine.decide(load_instance(args.instance))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"{decision['instance']}: {decision['selected_method']} -> "
          f"{decision['total_served_mbps']:.0f}/"
          f"{decision['max_requested_mbps']} Mbps, feasible={decision['feasible']}")
    print(f"Decision: {args.output}")


if __name__ == "__main__":
    main()
