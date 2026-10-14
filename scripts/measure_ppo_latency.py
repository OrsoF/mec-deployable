"""Time validated static PPO decisions with a cached, trusted local checkpoint.

Measures the public DecisionEngine path, not just a neural-network forward pass.
No training or comparisons are performed, and algorithms are left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from network_ppo.inference import DecisionEngine
from network_ppo.instance_io import load_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instance", type=Path, default=Path("instances/eval/bottleneck_unseen.json"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=Path("outputs/ppo_latency.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.episodes < 1:
        parser.error("repeats/episodes must be positive; warmup must be nonnegative")

    problem = load_instance(args.instance)
    metadata = json.loads((args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
    start = perf_counter_ns()
    engine = DecisionEngine(strategy="ppo", checkpoint=args.checkpoint,
                            paths=int(metadata["paths"]), episodes=args.episodes,
                            stochastic=False, seed=args.seed)
    initialization_ms = (perf_counter_ns() - start) / 1e6

    for _ in range(args.warmup):
        engine.decide(problem)
    times, scores, steps = [], [], []
    for _ in range(args.repeats):
        start = perf_counter_ns()
        decision = engine.decide(problem)
        times.append((perf_counter_ns() - start) / 1e6)
        if not decision["feasible"] or decision["selected_method"] != "ppo":
            raise RuntimeError("Measurement requires a feasible PPO decision")
        scores.append(decision["total_served_mbps"])
        steps.append(decision["policy_steps"])

    import torch
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "protocol": {
            "strategy": "ppo", "instance": problem.id,
            "instance_sha256": sha(args.instance),
            "checkpoint_sha256": {name: sha(args.checkpoint/name) for name in
                                  ("model.zip", "vecnormalize.pkl", "metadata.json")},
            "training_steps": metadata.get("timesteps_actual"),
            "action_mode": metadata.get("action_mode", "atomic"),
            "episodes": args.episodes, "policy_steps_per_decision": sorted(set(steps)),
            "deterministic": True, "seed": args.seed,
            "warmup": args.warmup, "repeats": args.repeats,
            "timer": "perf_counter_ns outside DecisionEngine.decide",
            "included": ["path/seed/environment construction", "normalization loading",
                         "PPO edit episode", "feasibility checking", "decision dictionary construction"],
            "excluded": ["initial checkpoint load", "input file/JSON parsing",
                         "JSON serialization", "disk writes", "HTTP transport"],
            "initialization": "Separate engine initialization includes lazy RL imports and policy loading.",
        },
        "environment": {
            "python": platform.python_version(), "os": platform.system(),
            "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(), "device": "cpu",
            "versions": {name: version(name) for name in
                         ("numpy", "networkx", "torch", "gymnasium", "stable-baselines3")},
        },
        "engine_initialization_ms": initialization_ms,
        "decision_latency_ms": {
            "p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
            "mean": float(np.mean(times)), "min": min(times), "max": max(times),
        },
        "outcomes": {"all_feasible": True, "served_mbps": sorted(set(scores)),
                     "requested_mbps": sum(d.max_mbps for d in problem.demands)},
        "samples_ms": times,
        "limitations": ["Repeated serial requests on one synthetic instance and one checkpoint.",
                        "Latency varies by hardware, workload and policy actions.",
                        "A smoke checkpoint does not establish placement quality or production performance."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    stats = report["decision_latency_ms"]
    print(f"Validated PPO decision: p50={stats['p50']:.2f} ms, p95={stats['p95']:.2f} ms; "
          f"{args.repeats} samples after {args.warmup} warmups")
    print(f"Engine initialization: {initialization_ms:.2f} ms; output: {args.output}")


if __name__ == "__main__":
    main()
