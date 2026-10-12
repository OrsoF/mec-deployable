"""Shared traffic, latency, resource and residual calculations."""

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray, ProblemData
from mecbench.domain.problem_arrays import ProblemArrays, problem_arrays


def traffic_arrays(
    problem: ProblemData | ProblemArrays, x: ArrayLike, q: ArrayLike
) -> dict[str, FloatArray]:
    """Calculate each arc's total traffic and the resulting demand latencies."""
    routing = np.asarray(x, dtype=float)
    quantities = np.asarray(q, dtype=float)
    data = problem_arrays(problem)
    edge_load = routing.T @ quantities
    arc_latency = data.latency_offsets + data.latency_slopes * edge_load
    return {
        "edge_load": edge_load,
        "arc_latency": arc_latency,
        "demand_latency": routing @ arc_latency,
    }


def residual_arrays(
    problem: ProblemData | ProblemArrays, x: ArrayLike, y: ArrayLike, q: ArrayLike
) -> dict[str, FloatArray]:
    data = problem_arrays(problem)
    traffic = traffic_arrays(data, x, q)
    y = np.asarray(y, dtype=float)
    node_load = y.T @ data.service_resources
    capacities = data.arc_capacities
    resources = data.node_resources
    return {
        **traffic,
        "node_load": node_load,
        "remaining_capacity": capacities - traffic["edge_load"],
        "remaining_resources": resources - node_load,
    }
