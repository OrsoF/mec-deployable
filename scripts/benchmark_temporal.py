"""Paired evaluation of temporal baselines and optional independently trained PPO.

Train and validation are never included by default. Every method sees the same
realized traffic and public forecast. MPC is causal, finite-width beam search,
NOT a certified optimum. Report all stage rewards, fallbacks and runtimes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median, pstdev

from network_ppo.dynamics import load_temporal_scenario
from network_ppo.paths import PathCatalog
from network_ppo.temporal_costs import stage_score
from network_ppo.temporal_optimization import TemporalPlanner, initial_allocation


def baseline_rollout(
    scenario,
    *,
    strategy,
    horizon=4,
    beam_width=2,
    max_candidates=9,
    max_planning_ms=None,
    max_expansions=40,
):
    planner = TemporalPlanner(
        strategy=strategy,
        horizon=horizon,
        beam_width=beam_width,
        max_candidates=max_candidates,
        max_planning_ms=max_planning_ms,
        max_expansions=max_expansions,
    )
    previous = initial_allocation(scenario)
    rows = []
    for t in range(scenario.horizon):
        decision = planner.decide(scenario, t, previous)
        actual = scenario.problem_at(t)
        metrics = stage_score(
            scenario, actual, previous, decision.solution, PathCatalog(actual, 3)
        )
        rows.append(
            {
                "period": t,
                "strategy": strategy,
                "reward": metrics.reward,
                "served_mbps": metrics.served_mbps,
                "migrations": metrics.migrations,
                "route_changes": metrics.route_changes,
                "active_nodes": metrics.active_nodes,
                "fallback": decision.fallback,
                "runtime_ms": decision.runtime_ms,
                "valid_proposal": True,
                "candidates_scored": decision.candidate_count,
                "horizon_reached": decision.lookahead_evaluated,
                "planning_cutoff": decision.budget_exhausted,
            }
        )
        previous = decision.solution
    return rows


def summarize(scenario, rows, label, *, train_seed=None, eval_seed=None):
    returns = [r["reward"] for r in rows]
    times = [r["runtime_ms"] for r in rows]
    return {
        "scenario": scenario.id,
        "split": scenario.split,
        "strategy": label,
        "training_seed": train_seed,
        "evaluation_seed": eval_seed,
        "total_reward": round(sum(returns), 5),
        "discounted_reward": round(
            sum(scenario.discount**t * r for t, r in enumerate(returns)), 5
        ),
        "total_mbps_periods": round(sum(r["served_mbps"] for r in rows), 5),
        "mean_mbps": round(mean(r["served_mbps"] for r in rows), 5),
        "migrations": sum(r["migrations"] for r in rows),
        "route_changes": sum(r["route_changes"] for r in rows),
        "fallback_periods": sum(bool(r["fallback"]) for r in rows),
        "invalid_proposals": sum(not r["valid_proposal"] for r in rows),
        "planning_cutoffs": sum(bool(r.get("planning_cutoff", False)) for r in rows),
        "mean_horizon_reached": round(
            mean(r.get("horizon_reached", 1) for r in rows), 2
        ),
        "runtime_total_ms": round(sum(times), 4),
        "runtime_p50_ms": round(median(times), 4),
        "runtime_p95_ms": round(
            float(sorted(times)[min(len(times) - 1, int(0.95 * len(times)))]), 4
        ),
    }


def save_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted(set().union(*(r.keys() for r in records)))
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=Path, default=Path("instances/temporal/test"))
    ap.add_argument("--output", type=Path, default=Path("outputs/temporal_benchmark"))
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--max-candidates", type=int, default=9)
    ap.add_argument(
        "--max-planning-ms",
        type=float,
        default=0,
        help="0 disables wall-time cutoff for deterministic experiments",
    )
    ap.add_argument(
        "--max-expansions",
        type=int,
        default=40,
        help="0 disables expansion cap for full-depth MPC; can be slow",
    )
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--checkpoints", type=Path, nargs="*", default=[])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 202, 303])
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()
    if (
        any(x < 1 for x in args.horizons)
        or min(args.beam_width, args.max_candidates) < 1
    ):
        ap.error("Positive horizons and MPC widths required")
    files = sorted(args.scenarios.glob("*.json"))
    if args.max_cases is not None:
        files = files[: args.max_cases]
    if not files:
        ap.error("No temporal scenarios found")
    scenarios = [load_temporal_scenario(file) for file in files]
    if any(s.split != "test" for s in scenarios):
        ap.error("The benchmark requires held-out split=test scenarios")
    checkpoint_models = []
    if args.checkpoints:
        from network_ppo.temporal_policy import TemporalPolicy

        for path in args.checkpoints:
            runtime = TemporalPolicy(path)
            for file in files:
                checksum = hashlib.sha256(file.read_bytes()).hexdigest()
                if checksum in runtime.metadata.get("training_sha256", {}).values():
                    ap.error(f"Data leakage: {file.name} is part of PPO training")
            checkpoint_models.append(runtime)

    summaries = []
    periods = []
    for scenario in scenarios:
        methods = ["keep", "greedy", "myopic"] + [
            f"mpc{h}" for h in args.horizons if h > 1
        ]
        # "myopic" and MPC1 share exactly the same objective and candidate generator.
        for method in methods:
            h = int(method[3:]) if method.startswith("mpc") else 1
            rows = baseline_rollout(
                scenario,
                strategy="mpc" if method.startswith("mpc") else method,
                horizon=h,
                beam_width=args.beam_width,
                max_candidates=args.max_candidates,
                max_planning_ms=args.max_planning_ms or None,
                max_expansions=args.max_expansions or None,
            )
            summaries.append(summarize(scenario, rows, method))
            periods.extend(
                {"scenario": scenario.id, "method": method, **r} for r in rows
            )
        for runtime in checkpoint_models:
            training_seed = runtime.metadata["seed"]
            for eval_seed in (
                args.eval_seeds if args.stochastic else [args.eval_seeds[0]]
            ):
                rows = runtime.rollout(
                    scenario, deterministic=not args.stochastic, seed=eval_seed
                )
                label = "ppo_stochastic" if args.stochastic else "ppo_deterministic"
                summaries.append(
                    summarize(
                        scenario,
                        rows,
                        label,
                        train_seed=training_seed,
                        eval_seed=eval_seed,
                    )
                )
                periods.extend(
                    {
                        "scenario": scenario.id,
                        "method": label,
                        "training_seed": training_seed,
                        "evaluation_seed": eval_seed,
                        **r,
                    }
                    for r in rows
                )
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "results.csv", summaries)
    save_csv(args.output / "periods.csv", periods)
    grouped = {}
    for row in summaries:
        label = row["strategy"]
        grouped.setdefault(label, []).append(row)
    stats = {
        name: {
            "n_rows": len(group),
            "mean_total_reward": mean(x["total_reward"] for x in group),
            "std_total_reward": pstdev(x["total_reward"] for x in group),
            "mean_runtime_total_ms": mean(x["runtime_total_ms"] for x in group),
        }
        for name, group in grouped.items()
    }
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "test_files": [f.name for f in files],
                    "forecast": "exogenous public mean; no future realized traffic",
                    "mpc": "time-bounded beam approximation, not certified optimum",
                    "max_planning_ms": args.max_planning_ms or None,
                    "max_expansions": args.max_expansions or None,
                    "training_seed_is_independent_from_evaluation_seed": True,
                    "runtime": "wall clock per decision, excluding checkpoint load",
                    "ppo_checkpoints": [str(p) for p in args.checkpoints],
                    "eval_seeds": args.eval_seeds,
                    "stochastic": args.stochastic,
                },
                "summary": stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for name, row in stats.items():
        print(
            f"{name:20s} reward={row['mean_total_reward']:.2f} +/- {row['std_total_reward']:.2f}; "
            f"total inference={row['mean_runtime_total_ms']:.1f}ms"
        )
    print("Saved", args.output)


if __name__ == "__main__":
    main()
