"""Export only physically validated solutions with human-readable node IDs."""
from __future__ import annotations

from datetime import datetime, timezone

from mecbench.domain.conversions import to_arc_solution
from mecbench.domain.loads import residual_arrays
from mecbench.domain.validation import validate


def format_result(instance, compact_solution, path_resolver, *, checkpoint, steps, seed):
    problem = instance.problem
    arc_solution = to_arc_solution(problem, compact_solution, path_resolver=path_resolver)
    report = validate(problem, arc_solution)
    if not report.feasible:
        raise ValueError("Final candidate fails feasibility checks: " + "; ".join(
            violation.message for violation in report.violations[:5]
        ))
    data = problem.data
    loads = residual_arrays(data, arc_solution.x, arc_solution.y, arc_solution.q)
    placements = []
    for s, node in enumerate(compact_solution.placement):
        placements.append({
            "service": instance.service_ids[s],
            "node": instance.node_ids[int(node)],
            "cpu_vcpu": float(data.service_resources[s]),
        })
    demands = []
    for d, qty in enumerate(compact_solution.quantities):
        service = int(data.demand_to_service[d])
        start = int(data.origins[d])
        target = int(compact_solution.placement[service])
        path = path_resolver(start, target, int(compact_solution.routing[d]))
        demands.append({
            "demand": instance.demand_ids[d],
            "service": instance.service_ids[service],
            "origin": instance.node_ids[start],
            "route": [instance.node_ids[int(n)] for n in path],
            "quantity_mbps": float(qty),
            "min_mbps": float(data.min_flows[d]),
            "max_mbps": float(data.max_flows[d]),
            "latency_ms": float(loads["demand_latency"][d]),
            "latency_budget_ms": float(data.latency_limits[d]),
        })
    node_loads = [{
        "node": instance.node_ids[n],
        "used_vcpu": float(loads["node_load"][n]),
        "available_vcpu": float(data.node_resources[n]),
    } for n in range(data.n_nodes)]
    links = [{
        "source": instance.node_ids[a], "target": instance.node_ids[b],
        "load_mbps": float(loads["edge_load"][index]),
        "capacity_mbps": float(data.arc_capacities[index]),
    } for index, (a, b) in enumerate(data.arcs)]
    return {
        "instance": instance.name,
        "source_file": str(instance.source),
        "checkpoint": str(checkpoint),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "policy_steps": steps,
        "feasible": True,
        "objective": "maximize_sum_served_Mbps",
        "total_served_mbps": float(arc_solution.q.sum()),
        "max_requested_mbps": float(data.max_flows.sum()),
        "served_fraction": float(arc_solution.q.sum() / max(float(data.max_flows.sum()), 1e-9)),
        "placements": placements,
        "node_compute_loads": node_loads,
        "demands": demands,
        "directed_links": links,
    }
