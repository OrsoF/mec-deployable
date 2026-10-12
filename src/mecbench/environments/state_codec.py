import networkx as nx
import numpy as np

from mecbench.domain.conversions import compact_to_arc, split_state, to_int_array
from mecbench.domain.conversions import make_state as make_state
from mecbench.domain.network import NetworkGraph
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.problem_arrays import problem_arrays
from mecbench.utils.heuristic_solvers import basic_state, clip_state
from mecbench.utils.solver_paths import ShortestPathCache
from mecbench.utils.solver_workspace import SolvingInstance


def to_integer_state(s: np.ndarray | list) -> np.ndarray:
    return to_int_array(s)


def state_to_vectors(
    problem: ProblemInstance,
    s: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return split_state(problem, s)


def make_basic_state(problem: ProblemInstance, graph: nx.Graph) -> np.ndarray:
    return basic_state(problem, graph)


def clip_state_to_bounds(
    problem: ProblemInstance,
    graph: nx.Graph,
    max_path_candidates: int,
    s: np.ndarray | list,
) -> np.ndarray:
    return clip_state(problem, graph, max_path_candidates, s)


def is_feasible_state(
    problem: ProblemInstance,
    solution_checker: SolvingInstance,
    path_cache: ShortestPathCache,
    s: np.ndarray | list,
) -> bool:
    data = problem_arrays(problem)
    placement, routing, quantities = split_state(problem, to_int_array(s))
    for d in range(data.n_demands):
        k = data.demand_to_service[d]
        try:
            path_cache.nth_shortest_path(data.origins[d], placement[k], routing[d])
        except (IndexError, nx.NetworkXException):
            return False
    return solution_checker.is_feasible(placement, routing, quantities)


def state_to_variables(
    problem: ProblemInstance,
    graph: nx.Graph,
    network_graph: NetworkGraph,
    path_cache: ShortestPathCache,
    s: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, r, q = split_state(problem, to_int_array(s))
    return compact_to_arc(
        problem,
        graph,
        path_cache.nth_shortest_path,
        p,
        r,
        q,
    )
