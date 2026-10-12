from typing import Any

import networkx as nx
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete

from mecbench.domain.loads import residual_arrays
from mecbench.domain.problem import ProblemInstance
from mecbench.environments.state_codec import state_to_vectors
from mecbench.utils.solver_paths import ShortestPathCache
from mecbench.utils.solver_workspace import apply_residuals_to_graph


def make_action_space(
    problem_instance: ProblemInstance, graph: nx.Graph, max_path_candidates: int
) -> MultiDiscrete:
    """Build the discrete action space for the loaded problem dimensions."""
    data = problem_instance.data
    return MultiDiscrete(
        [
            3,
            data.n_services,
            data.n_nodes,
            data.n_demands,
            3,
            max_path_candidates,
        ]
    )


def make_vector_observation_space(
    problem_instance: ProblemInstance, graph: nx.Graph, max_path_candidates: int
) -> Box:
    """Build the vector observation bounds for the loaded problem."""
    data = problem_instance.data
    demand_count, service_count = data.n_demands, data.n_services
    node_count, edge_count = data.n_nodes, data.n_arcs
    total_max_flow = float(data.max_flows.sum())
    max_edge_latency = float(
        (data.latency_offsets + total_max_flow * data.latency_slopes).max(initial=0)
    )
    latency_ratio_high = [
        node_count * max_edge_latency / max(latency, 1e-9) for latency in data.latency_limits
    ]

    low_array = np.zeros(
        service_count
        + demand_count
        + demand_count
        + demand_count
        + edge_count
        + node_count
        + demand_count
        + demand_count
        + edge_count
        + node_count
        + service_count
        + demand_count
    )
    high_array = np.array(
        [max(node_count - 1, 0)] * service_count
        + [max(max_path_candidates - 1, 0)] * demand_count
        + data.max_flows.tolist()
        + latency_ratio_high
        + [1.0] * edge_count
        + [1.0] * node_count
        + data.max_flows.tolist()
        + data.latency_limits.tolist()
        + data.arc_capacities.tolist()
        + data.node_resources.tolist()
        + data.service_resources.tolist()
        + [node_count] * demand_count
    )
    return Box(
        low=low_array.astype(np.float32),
        high=high_array.astype(np.float32),
        dtype=np.float32,
    )


def make_vector_observation(env: Any, state: np.ndarray | list) -> np.ndarray:
    """Build the flat observation vector used by current RL agents."""
    data = env.data
    x, y, q = env._state_to_variables(state)
    residual = residual_arrays(data, x, y, q)
    apply_residuals_to_graph(data, env.graph, x, y, q)

    latency_proportion = (residual["demand_latency"] / data.latency_limits).tolist()

    arc_saturation_proportion = [
        residual["remaining_capacity"][edge_index] / data.arc_capacities[edge_index]
        for edge_index in range(data.n_arcs)
    ]

    server_saturation = [
        residual["remaining_resources"][node] / (data.node_resources[node] + 0.01)
        for node in range(data.n_nodes)
    ]

    path_lengths = [
        len(get_demand_path(env.problem_instance, env.path_cache, demand_index, state))
        for demand_index in range(data.n_demands)
    ]

    return np.array(
        np.asarray(state).tolist()
        + latency_proportion
        + arc_saturation_proportion
        + server_saturation
        + env.data.max_flows.tolist()
        + env.data.latency_limits.tolist()
        + env.data.arc_capacities.tolist()
        + env.data.node_resources.tolist()
        + env.data.service_resources.tolist()
        + path_lengths,
        dtype=np.float32,
    )


def get_demand_path(
    problem: ProblemInstance,
    path_cache: ShortestPathCache,
    demand_index: int,
    state: np.ndarray | list,
) -> list:
    """Return the selected path for a demand in the given state."""
    service_placement, path_indices, _ = state_to_vectors(problem, state)
    service_index = problem.data.demand_to_service[demand_index]
    origin = problem.data.origins[demand_index]
    target = service_placement[service_index]
    return path_cache.nth_shortest_path(origin, target, path_indices[demand_index])
