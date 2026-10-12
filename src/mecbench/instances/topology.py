"""Topology builders and GML instance loading."""

from pathlib import Path

import networkx as nx
import numpy as np

from mecbench.domain.network import NetworkGraph
from mecbench.domain.problem import ProblemInstance
from mecbench.instances.services import build_gml_service_package


def build_tree_graph(
    n_users: int,
    neighbors: int = 1,
) -> nx.Graph:
    """
    Build a small edge/cloud tree-like graph.

    Topology:
        users -- local neighbors
           |
          wap
           |
         cloud
    """
    assert n_users >= 1
    assert neighbors >= 0

    graph = nx.Graph()

    users = list(range(n_users))
    wap = n_users
    cloud = n_users + 1

    graph.add_nodes_from(range(n_users + 2))

    # Local user-user connectivity
    for i in range(n_users):
        for delta in range(-neighbors, neighbors + 1):
            j = i + delta

            if i == j:
                continue

            if 0 <= j < n_users:
                graph.add_edge(users[i], users[j])

    # User <-> WAP
    for user in users:
        graph.add_edge(user, wap)

    # WAP <-> cloud
    graph.add_edge(wap, cloud)

    return graph


def build_star_graph(
    n_groups: int,
    users_per_group: int,
    neighbors: int = 1,
) -> nx.Graph:
    """
    Build a grouped star/mesh topology.

    Topology:
        user mesh -> wap -> cloud
    repeated for each group.
    """
    assert n_groups >= 1
    assert users_per_group >= 1
    assert neighbors >= 0

    graph = nx.Graph()

    n_users = n_groups * users_per_group
    waps = [n_users + group_idx for group_idx in range(n_groups)]
    cloud = n_users + n_groups

    graph.add_nodes_from(range(n_users + n_groups + 1))

    for group_idx in range(n_groups):
        users = [group_idx * users_per_group + user_idx for user_idx in range(users_per_group)]
        wap = waps[group_idx]

        # Circular local neighborhood
        for i in range(users_per_group):
            for delta in range(-neighbors, neighbors + 1):
                if delta == 0:
                    continue

                j = (i + delta) % users_per_group
                if j == i:
                    continue
                graph.add_edge(users[i], users[j])

        # Users <-> WAP
        for user in users:
            graph.add_edge(user, wap)

        # WAP <-> cloud
        graph.add_edge(wap, cloud)

    return graph


def build_graph_from_gml(gml_path: str | Path) -> nx.Graph:
    gml_path = Path(gml_path)

    if not gml_path.exists():
        raise FileNotFoundError(gml_path)

    return nx.read_gml(gml_path)


def build_large_graph(
    base_graph: nx.Graph,
    edge_per_core: int = 10,
    users_per_edge: int = 2,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    """
    Build a hierarchical large-scale graph.

    Topology:
        core nodes
            ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬Å“
        edge servers
            ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬Å“
        users
    """
    rng = rng or np.random.default_rng(0)

    graph = base_graph.copy()

    core_nodes = list(graph.nodes)

    edge_nodes = []
    user_nodes = []

    # ----- Edge layer -----

    for core in core_nodes:
        for _edge_idx in range(edge_per_core):
            edge_node = graph.number_of_nodes()

            graph.add_node(edge_node, layer="edge")
            graph.add_edge(core, edge_node)

            edge_nodes.append(edge_node)

    # ----- User layer -----

    for edge_node in edge_nodes:
        for _user_idx in range(users_per_edge):
            user_node = graph.number_of_nodes()

            graph.add_node(user_node, layer="user")
            graph.add_edge(edge_node, user_node)

            user_nodes.append(user_node)

    return graph


def _constant_edge_list(
    graph: nx.Graph,
    value: float,
) -> list[float]:
    return [value] * graph.number_of_edges()


def _make_network_graph(
    graph: nx.Graph,
    node_resources: list[float],
    edge_capacities: list[float],
    alpha: float,
    beta: float,
) -> NetworkGraph:
    return NetworkGraph(
        base_graph=graph,
        node_resources=node_resources,
        edge_capacity=edge_capacities,
        latency_alpha_factor=_constant_edge_list(graph, alpha),
        latency_beta_factor=_constant_edge_list(graph, beta),
    )


def create_gml_problem_instance(
    gml_path: str | Path,
    n_services: int,
    demands_per_service: int,
    default_node_resources: float = 1e4,
    default_edge_capacity: float = 1e3,
    rng: np.random.Generator | None = None,
) -> ProblemInstance:
    rng = rng or np.random.default_rng(0)

    graph = build_graph_from_gml(gml_path)

    node_resources = [default_node_resources] * graph.number_of_nodes()

    edge_capacities = [default_edge_capacity] * graph.number_of_edges()

    network_graph = _make_network_graph(
        graph=graph,
        node_resources=node_resources,
        edge_capacities=edge_capacities,
        alpha=0.2,
        beta=5.0,
    )

    graph_name = Path(gml_path).stem

    instance_name = f"gml_{graph_name}_{n_services}_{demands_per_service}"

    service_package = build_gml_service_package(
        network_graph=network_graph,
        n_services=n_services,
        demands_per_service=demands_per_service,
        instance_name=instance_name,
        rng=rng,
    )

    return ProblemInstance(
        network_graph=network_graph,
        service_package=service_package,
        problem_name=instance_name,
    )
