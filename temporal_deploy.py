"""Causal one-step temporal deployment; input excludes realized future loads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from network_ppo.temporal_inference import TemporalDecisionEngine


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--snapshot", type=Path, default=Path("instances/temporal/online_example.json")
    )
    ap.add_argument(
        "--output", type=Path, default=Path("outputs/temporal_decision.json")
    )
    ap.add_argument(
        "--strategy",
        choices=("keep", "greedy", "myopic", "mpc", "ppo", "auto"),
        default="mpc",
    )
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--max-candidates", type=int, default=9)
    ap.add_argument("--max-planning-ms", type=float, default=400)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    engine = TemporalDecisionEngine(
        strategy=args.strategy,
        horizon=args.horizon,
        beam_width=args.beam_width,
        max_candidates=args.max_candidates,
        max_planning_ms=args.max_planning_ms,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )
    output = engine.decide(json.loads(args.snapshot.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"{output['selected_method']} reward={output['score']['reward']:.2f}, feasible={output['feasible']}; {args.output}"
    )


if __name__ == "__main__":
    main()
