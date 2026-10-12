import networkx as nx
import numpy as np

from mecbench.domain.network import NetworkGraph
from mecbench.domain.services import Demand, Service, ServicePackage

# ============================================================
# Generic helpers
# ============================================================


def _mean_edge_attribute(
    graph: nx.Graph,
    attribute: str,
    default: float,
) -> float:
    values = [graph.edges[edge].get(attribute, default) for edge in graph.edges]

    return float(np.mean(values))


def _generic_service_package(
    network_graph: NetworkGraph,
    n_services: int,
    demands_per_service: int,
    flow_min: float,
    flow_max: float,
    latency: float,
    resources_needed: float,
    origin_nodes: list,
    rng: np.random.Generator,
    instance_name: str,
) -> ServicePackage:
    services = []

    for _ in range(n_services):
        demands = []

        for _ in range(demands_per_service):
            origin = rng.choice(origin_nodes)

            demand = Demand(
                origin_node_index=origin,
                flow_min=flow_min,
                flow_max=flow_max,
                max_latency=latency,
            )

            demands.append(demand)

        service = Service(
            task_list=demands,
            network_graph=network_graph,
            resources_needed=resources_needed,
        )

        services.append(service)

    return ServicePackage(
        services=services,
        instance_name=instance_name,
    )


# ============================================================
# GML services
# ============================================================


def build_gml_service_package(
    network_graph: NetworkGraph,
    n_services: int,
    demands_per_service: int,
    instance_name: str,
    rng: np.random.Generator | None = None,
) -> ServicePackage:
    rng = rng or np.random.default_rng(0)

    nx_graph = network_graph.nx_graph

    alpha_mean = _mean_edge_attribute(nx_graph, "alpha", 1.0)
    beta_mean = _mean_edge_attribute(nx_graph, "beta", 1.0)

    latency = 7.0 * (
        alpha_mean * 5.0 * n_services**2 / max(1, nx_graph.number_of_edges()) + beta_mean
    )

    return _generic_service_package(
        network_graph=network_graph,
        n_services=n_services,
        demands_per_service=demands_per_service,
        flow_min=1.0,
        flow_max=100.0,
        latency=latency,
        resources_needed=5000 // n_services,
        origin_nodes=list(nx_graph.nodes),
        rng=rng,
        instance_name=instance_name,
    )


# ============================================================
# Tree services
# ============================================================


def build_tree_service_package(
    network_graph: NetworkGraph,
    n_services: int,
    instance_name: str,
) -> ServicePackage:
    cloud_node = n_services + 1

    services = []

    for user_idx in range(n_services):
        user_demand = Demand(
            origin_node_index=user_idx,
            flow_min=1,
            flow_max=5,
            max_latency=10 * n_services,
        )

        cloud_demand = Demand(
            origin_node_index=cloud_node,
            flow_min=1,
            flow_max=2,
            max_latency=10 * n_services,
        )

        service = Service(
            task_list=[user_demand, cloud_demand],
            network_graph=network_graph,
            resources_needed=1,
        )

        services.append(service)

    return ServicePackage(
        services=services,
        instance_name=instance_name,
    )


# ============================================================
# Star/grouped services
# ============================================================


def build_star_service_package(
    network_graph: NetworkGraph,
    n_groups: int,
    users_per_group: int,
    instance_name: str,
) -> ServicePackage:
    services = []

    for group_idx in range(n_groups):
        for user_idx in range(users_per_group):
            user_node = group_idx * users_per_group + user_idx

            user_demand = Demand(
                origin_node_index=user_node,
                flow_min=1,
                flow_max=5,
                max_latency=5 * n_groups * users_per_group,
            )

            cloud_demand = Demand(
                origin_node_index=n_groups * users_per_group + n_groups,
                flow_min=1,
                flow_max=2,
                max_latency=5 * n_groups * users_per_group,
            )

            service = Service(
                task_list=[user_demand, cloud_demand],
                network_graph=network_graph,
                resources_needed=1,
            )

            services.append(service)

    return ServicePackage(
        services=services,
        instance_name=instance_name,
    )


# ============================================================
# Large hierarchical services
# ============================================================


def build_large_service_package(
    network_graph: NetworkGraph,
    n_services: int,
    demands_per_service: int,
    instance_name: str,
    rng: np.random.Generator | None = None,
) -> ServicePackage:
    rng = rng or np.random.default_rng(0)

    nx_graph = network_graph.nx_graph

    alpha_mean = _mean_edge_attribute(nx_graph, "alpha", 1.0)
    beta_mean = _mean_edge_attribute(nx_graph, "beta", 1.0)

    latency = 25.0 * (
        alpha_mean * 5.0 * n_services**2 / max(1, nx_graph.number_of_edges()) + beta_mean
    )

    origin_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get("layer") != "edge"]

    return _generic_service_package(
        network_graph=network_graph,
        n_services=n_services,
        demands_per_service=demands_per_service,
        flow_min=1.0,
        flow_max=100.0,
        latency=latency,
        resources_needed=5000 // n_services,
        origin_nodes=origin_nodes,
        rng=rng,
        instance_name=instance_name,
    )
