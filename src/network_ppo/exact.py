"""Optional exact small-instance solver by enumeration + integer linear programming.

For a fixed placement and routes, capacity and congestion-dependent latency
constraints are linear in Mbps quantities. Enumerating all placements/routings
and solving the resulting MILPs is globally exact only when the search finishes.
Exponential search: intended for tiny test instances, NOT deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np

from .constraints import assess
from .heuristic import solve_baseline
from .paths import PathCatalog
from .problem import Problem, Solution


@dataclass
class ExactResult:
    solution: Solution
    certified: bool
    configurations_checked: int
    upper_bound_mbps: int
    reason: str


def solve_exact_small(problem: Problem, paths: PathCatalog, *,
                      max_configs: int = 5000, time_limit_s: float = 20) -> ExactResult:
    """Return best feasible candidate; `certified` false if search truncated.

    Requires optional scipy; per-configuration MILP solver status is checked.
    No certificate is claimed if any configuration was skipped or timed out.
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise ImportError("Exact small-instance solver requires scipy") from exc
    if max_configs < 1 or time_limit_s <= 0:
        raise ValueError("Positive max_configs and time_limit_s required")
    d, s, n = len(problem.demands), len(problem.services), len(problem.nodes)
    upper = int(sum(x.max_mbps for x in problem.demands))
    best = solve_baseline(problem, paths, "lookahead")
    if best.quantities.sum() == upper:
        return ExactResult(best, True, 0, upper, "demand upper bound attained")
    start = perf_counter()
    explored = 0
    lower = np.array([x.min_mbps for x in problem.demands], dtype=float)
    high = np.array([x.max_mbps for x in problem.demands], dtype=float)
    slopes = problem.arc_latency_slopes
    base = problem.arc_base_latencies
    incomplete = False

    for placement in product(range(n), repeat=s):
        cpu = np.zeros(n)
        for idx, node in enumerate(placement):
            cpu[node] += problem.services[idx].cpu
        if np.any(cpu > [x.cpu for x in problem.nodes]):
            continue
        for routes in product(range(paths.k), repeat=d):
            if explored >= max_configs or perf_counter() - start >= time_limit_s:
                incomplete = True
                break
            explored += 1
            incidence = np.zeros((d, len(problem.arcs)))
            for i, demand in enumerate(problem.demands):
                chosen = paths.route(demand.origin, placement[demand.service], routes[i])
                for edge in zip(chosen, chosen[1:]):
                    incidence[i, paths.arc_index[edge]] = 1
            # edge_load = incidence.T @ quantities
            edge_matrix = incidence.T
            # latency_i = incidence_i @ base + sum_j [incidence_i @ diag(slope) @ incidence_j] q_j
            latency_matrix = (incidence * slopes) @ incidence.T
            latency_rhs = np.array([x.max_latency_ms for x in problem.demands]) - incidence @ base
            matrix = np.vstack([edge_matrix, latency_matrix])
            rhs = np.concatenate([problem.arc_capacities, latency_rhs])
            result = milp(c=-np.ones(d), integrality=np.ones(d),
                          bounds=Bounds(lower, high),
                          constraints=LinearConstraint(matrix, -np.inf, rhs),
                          options={"time_limit": max(0.001, time_limit_s - (perf_counter() - start))})
            if result.status == 1:
                incomplete = True
                break
            if result.status not in (0, 2):
                raise RuntimeError(f"MILP solver failed on configuration {explored}: {result.message}")
            if result.x is None:
                continue
            candidate = Solution(np.array(placement, dtype=np.int64),
                                 np.array(routes, dtype=np.int64),
                                 np.rint(result.x).astype(np.int64))
            if not assess(problem, candidate, paths).feasible:
                raise RuntimeError("MILP returned infeasible integer solution")
            if candidate.quantities.sum() > best.quantities.sum():
                best = candidate
            if int(best.quantities.sum()) == upper:
                return ExactResult(best, True, explored, upper, "demand upper bound attained")
        if incomplete:
            break
    return ExactResult(best, not incomplete, explored, upper,
                       "exhaustive enumeration" if not incomplete else "search budget exhausted")
