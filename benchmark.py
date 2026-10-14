"""Benchmark validated decisions; report per-case, per-seed objective and runtime.

No PPO results are fabricated: --checkpoint is required to run PPO.
The same runtime budget should be separately controlled for rigorous claims.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from network_ppo.inference import DecisionEngine
from network_ppo.instance_io import load_instance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", type=Path, default=Path("instances/eval"))
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--paths", type=int, default=3)
    args = p.parse_args()
    files = sorted(args.instances.glob("*.json"))
    if not files:
        p.error("No instances found")
    methods = ["greedy", "lookahead", "multistart"]
    if args.checkpoint:
        methods.append("ppo")
    records = []
    for file in files:
        case = load_instance(file)
        for method in methods:
            for seed in (args.seeds if method == "ppo" else [args.seeds[0]]):
                engine = DecisionEngine(strategy=method,
                                        checkpoint=args.checkpoint if method == "ppo" else None,
                                        paths=args.paths, seed=seed, episodes=args.episodes,
                                        stochastic=args.stochastic)
                result = engine.decide(case)
                records.append({
                    "instance": case.id, "file": file.name, "method": method,
                    "seed": seed, "score_mbps": result["total_served_mbps"],
                    "upper_bound_mbps": result["demand_upper_bound_mbps"],
                    "feasible": result["feasible"],
                    "certified_optimal": result["certified_optimal"],
                    "runtime_ms": result["decision_wall_ms"],
                    "policy_steps": result["policy_steps"],
                })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for file in files:
        for method in methods:
            scores = [r["score_mbps"] for r in records if r["file"] == file.name and r["method"] == method]
            times = [r["runtime_ms"] for r in records if r["file"] == file.name and r["method"] == method]
            summary.append({"file": file.name, "method": method, "mean_mbps": mean(scores),
                            "std_mbps": pstdev(scores), "mean_runtime_ms": mean(times)})
    (args.output / "summary.json").write_text(
        json.dumps({"protocol": {"instances": [x.name for x in files],
                                 "seeds": args.seeds, "stochastic": args.stochastic,
                                 "episodes": args.episodes, "ppo_enabled": bool(args.checkpoint),
                                 "runtime_note": "Wall time includes seed generation and validation; model load excluded."},
                    "results": summary}, indent=2) + "\n", encoding="utf-8")
    for x in summary:
        print(f"{x['file']:25s} {x['method']:11s} "
              f"{x['mean_mbps']:5.1f} +/- {x['std_mbps']:.1f} Mbps; "
              f"{x['mean_runtime_ms']:.1f} ms")
    print(f"Saved {len(records)} runs to {args.output}")


if __name__ == "__main__":
    main()
