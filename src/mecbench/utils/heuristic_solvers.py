"""Shared initial candidates, clipping, repair and quantity search."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from mecbench.domain.contracts import FloatArray, ProblemData
from mecbench.domain.conversions import make_state, split_state, to_int_array
from mecbench.domain.loads import traffic_arrays
from mecbench.domain.problem_arrays import ProblemArrays, problem_arrays
from mecbench.domain.solution import Solution
from mecbench.domain.validation import validate_arc_solution

# Mutable search candidates are separate from published CompactSolution snapshots.
CompactCandidate: TypeAlias = tuple[list[int], list[int], list[float]]
StateArray: TypeAlias = NDArray[np.int_]


class SolutionChecker(Protocol):
    def is_feasible(
        self, placement: ArrayLike, routing: ArrayLike, quantities: ArrayLike
    ) -> bool: ...


def baseline_solution(problem: ProblemData) -> Solution:
    data = problem_arrays(problem)
    network = problem.network_graph
    graph = network.nx_graph
    placement_node = int(np.argmax(data.node_resources))
    x = np.zeros((data.n_demands, data.n_arcs))
    for d, origin in enumerate(data.origins):
        nodes = nx.shortest_path(graph, origin, placement_node)
        for edge in zip(nodes, nodes[1:], strict=False):
            x[d, data.arc_index[edge]] = 1.0
    y = np.zeros((data.n_services, data.n_nodes))
    y[:, placement_node] = 1.0
    return Solution(x, y, np.asarray(data.min_flows, dtype=float))


def basic_solution(problem: ProblemData | ProblemArrays, graph: nx.Graph) -> CompactCandidate:
    data = problem_arrays(problem)
    cloud = int(np.argmax(data.node_resources))
    return (
        [cloud] * data.n_services,
        [0] * data.n_demands,
        data.min_flows.tolist(),
    )


def basic_state(problem: ProblemData, graph: nx.Graph) -> StateArray:
    return make_state(*basic_solution(problem, graph))


def clip_solution(
    problem: ProblemData,
    graph: nx.Graph,
    max_path: int,
    placement: ArrayLike,
    routing: ArrayLike,
    quantities: ArrayLike,
) -> CompactCandidate:
    data = problem_arrays(problem)
    n = graph.number_of_nodes()
    p: list[int] = np.clip(placement, 0, n - 1).astype(int).tolist()
    r: list[int] = np.clip(routing, 0, max_path - 1).astype(int).tolist()
    q: list[float] = np.clip(quantities, data.min_flows, data.max_flows).astype(int).tolist()
    return p, r, q


def clip_state(problem: ProblemData, graph: nx.Graph, max_path: int, s: ArrayLike) -> StateArray:
    s = to_int_array(s)
    p, r, q = split_state(problem, s)
    p[:], r[:], q[:] = clip_solution(problem, graph, max_path, p, r, q)
    return s


def greedy_quantity_lift(
    problem: ProblemData,
    checker: SolutionChecker,
    rng: np.random.Generator,
    placement: Sequence[int],
    routing: Sequence[int],
    quantities: Sequence[float],
    steps: int,
    increment: float,
) -> tuple[Sequence[int], Sequence[int], list[float]]:
    data = problem_arrays(problem)
    q = list(quantities)
    D = data.n_demands
    for _ in range(steps):
        d = int(rng.integers(D))
        if q[d] >= data.max_flows[d]:
            continue
        delta = min(increment, data.max_flows[d] - q[d])
        q[d] += delta
        if not checker.is_feasible(placement, routing, q):
            q[d] -= delta
    return placement, routing, q


def repair_solution(
    problem: ProblemData,
    graph: nx.Graph,
    checker: SolutionChecker,
    max_path: int,
    placement: ArrayLike,
    routing: ArrayLike,
    quantities: ArrayLike,
) -> CompactCandidate | None:
    data = problem_arrays(problem)
    p, r, q = clip_solution(problem, graph, max_path, placement, routing, quantities)
    if checker.is_feasible(p, r, q):
        return p, r, q
    q = data.min_flows.tolist()
    if checker.is_feasible(p, r, q):
        return p, r, q
    return None


def remaining_quantity(
    problem: ProblemData | ProblemArrays,
    x: FloatArray,
    y: FloatArray,
    q: FloatArray,
    demand_index: int,
) -> float:
    """Integer increase allowed by every affected demand's shared-arc latency."""
    data = problem_arrays(problem)
    validate_arc_solution(data, x, y, q)
    traffic = traffic_arrays(data, x, q)
    edge_load = traffic["edge_load"]
    alpha = data.latency_slopes
    latency_slack = np.asarray(data.latency_limits) - traffic["demand_latency"]
    sensitivity = x @ (alpha * x[demand_index])
    affected = sensitivity > 0
    latency_room = (
        np.min(latency_slack[affected] / sensitivity[affected]) if np.any(affected) else np.inf
    )
    used = x[demand_index] > 0
    capacity_room = np.min((data.arc_capacities - edge_load)[used]) if np.any(used) else np.inf
    flow_room = data.max_flows[demand_index] - q[demand_index]
    return float(max(0.0, np.floor(min(latency_room, capacity_room, flow_room))))
