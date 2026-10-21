"""A machine-readable, independently checked placement/routing/flow decision."""

from __future__ import annotations

from datetime import datetime, timezone

from .constraints import assess
from .paths import PathCatalog
from .problem import Problem, Solution


def decision_json(
    problem: Problem,
    solution: Solution,
    paths: PathCatalog,
    *,
    checkpoint: str,
    policy_steps: int,
    seed: int,
) -> dict:
    a = assess(problem, solution, paths)
    if not a.feasible:
        raise ValueError(
            "Cannot export an infeasible solution: " + "; ".join(a.violations)
        )
    total_requested = sum(d.max_mbps for d in problem.demands)
    return {
        "schema_version": 1,
        "instance": problem.id,
        "checkpoint": checkpoint,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "policy_steps": policy_steps,
        "feasible": True,
        "objective": "maximize_sum_served_mbps",
        "total_served_mbps": a.total_mbps,
        "max_requested_mbps": total_requested,
        "served_fraction": a.total_mbps / total_requested if total_requested else 1.0,
        "placements": [
            {
                "service": s.id,
                "node": problem.nodes[int(solution.placement[i])].id,
                "cpu_vcpu": s.cpu,
            }
            for i, s in enumerate(problem.services)
        ],
        "demands": [
            {
                "demand": dm.id,
                "service": problem.services[dm.service].id,
                "origin": problem.nodes[dm.origin].id,
                "route": [problem.nodes[n].id for n in a.paths[i]],
                "quantity_mbps": int(solution.quantities[i]),
                "min_mbps": dm.min_mbps,
                "max_mbps": dm.max_mbps,
                "latency_ms": float(a.demand_latency[i]),
                "latency_budget_ms": dm.max_latency_ms,
            }
            for i, dm in enumerate(problem.demands)
        ],
        "node_compute_loads": [
            {
                "node": node.id,
                "used_vcpu": float(a.node_load[i]),
                "available_vcpu": node.cpu,
            }
            for i, node in enumerate(problem.nodes)
        ],
        "directed_links": [
            {
                "source": problem.nodes[u].id,
                "target": problem.nodes[v].id,
                "load_mbps": float(a.edge_load[i]),
                "capacity_mbps": float(problem.arc_capacities[i]),
            }
            for i, (u, v) in enumerate(problem.arcs)
        ],
    }
