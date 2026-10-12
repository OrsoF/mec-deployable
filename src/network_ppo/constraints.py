"""Reference feasibility checker (node CPU, per-direction Mbps and load-dependent latency)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paths import PathCatalog
from .problem import Problem, Solution


@dataclass(frozen=True)
class Assessment:
    feasible: bool
    violations: tuple[str, ...]
    edge_load: np.ndarray
    node_load: np.ndarray
    demand_latency: np.ndarray
    paths: tuple[tuple[int, ...], ...]
    total_mbps: float


def assess(problem: Problem, solution: Solution, paths: PathCatalog, *, tol: float = 1e-7) -> Assessment:
    n, s, d, a = len(problem.nodes), len(problem.services), len(problem.demands), len(problem.arcs)
    violations = []
    vectors = (solution.placement, solution.routes, solution.quantities)
    if any(np.asarray(v).shape != (count,) for v, count in zip(vectors, (s, d, d))):
        raise ValueError("Solution dimensions differ from problem")
    placement = np.asarray(solution.placement)
    routes = np.asarray(solution.routes)
    q = np.asarray(solution.quantities)
    if not all(np.all(np.isfinite(v)) for v in vectors):
        raise ValueError("Non-finite solution components")
    if not all(np.all(v == np.floor(v)) for v in vectors):
        violations.append("placement, routes and quantities must be integer-valued")
    if np.any((placement < 0) | (placement >= n)) or np.any((routes < 0) | (routes >= paths.k)):
        violations.append("invalid placement node or route index")
        # Never clip an invalid decision and call it feasible.
        return Assessment(False, tuple(violations), np.zeros(a), np.zeros(n), np.zeros(d), (), float(q.sum()))
    node_load = np.zeros(n)
    for i, service in enumerate(problem.services):
        node_load[int(placement[i])] += service.cpu
    incidence = np.zeros((d, a))
    route_nodes: list[tuple[int, ...]] = []
    for i, demand in enumerate(problem.demands):
        lo, hi = demand.min_mbps, demand.max_mbps
        if q[i] < lo - tol or q[i] > hi + tol:
            violations.append(f"{demand.id}: quantity outside [{lo}, {hi}] Mbps")
        path = paths.route(demand.origin, int(placement[demand.service]), int(routes[i]))
        route_nodes.append(path)
        for edge in zip(path, path[1:]):
            incidence[i, paths.arc_index[edge]] = 1.0
    edge_load = incidence.T @ q
    demand_latency = incidence @ (problem.arc_base_latencies + problem.arc_latency_slopes * edge_load)
    for i, node in enumerate(problem.nodes):
        if node_load[i] > node.cpu + tol:
            violations.append(f"{node.id}: CPU {node_load[i]:g} > {node.cpu:g} vCPU")
    for i, (u, v) in enumerate(problem.arcs):
        if edge_load[i] > problem.arc_capacities[i] + tol:
            violations.append(f"{problem.nodes[u].id}->{problem.nodes[v].id}: capacity exceeded")
    for i, demand in enumerate(problem.demands):
        if demand_latency[i] > demand.max_latency_ms + tol:
            violations.append(f"{demand.id}: latency {demand_latency[i]:g} > {demand.max_latency_ms:g} ms")
    return Assessment(not violations, tuple(violations), edge_load, node_load, demand_latency,
                      tuple(route_nodes), float(q.sum()))
