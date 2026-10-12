"""Problem data; mutable historical solution access is isolated in compat."""

from dataclasses import replace
from typing import Any

from mecbench.compat.problem_state import LegacyProblemState
from mecbench.domain.contracts import FloatArray, IntArray
from mecbench.domain.network import NetworkGraph as NetworkGraph
from mecbench.domain.network import NetworkSnapshot
from mecbench.domain.network import check_network_graph as check_network_graph
from mecbench.domain.network import is_graph_symmetric as is_graph_symmetric
from mecbench.domain.problem_arrays import ProblemArrays
from mecbench.domain.services import Demand as Demand
from mecbench.domain.services import Service as Service
from mecbench.domain.services import ServicePackage as ServicePackage


class ProblemInstance(LegacyProblemState):
    def __init__(
        self, network_graph: NetworkGraph, service_package: ServicePackage, problem_name: str
    ) -> None:
        self._validate_source_components(network_graph, service_package, problem_name)
        self._data = ProblemArrays.from_components(network_graph, service_package)
        self._network_graph = NetworkSnapshot.from_data(self._data, network_graph.nx_graph)
        self.service_package = service_package  # historical construction metadata
        self.problem_name = problem_name
        self.seed: int = 0
        self._legacy_state = None
        self.validate_components()

    @property
    def data(self) -> ProblemArrays:
        return self._data

    @property
    def network_graph(self) -> NetworkSnapshot:
        return self._network_graph

    def validate_components(self) -> None:
        self.data.validate()
        data = self.data
        if min(data.n_nodes, data.n_arcs, data.n_services, data.n_demands) <= 0:
            raise ValueError("A problem must contain nodes, arcs, services and demands.")
        for name in (
            "min_flows",
            "max_flows",
            "latency_limits",
            "service_resources",
            "latency_slopes",
            "latency_offsets",
            "node_resources",
        ):
            if (getattr(data, name) < 0).any():
                raise ValueError(f"{name} must be non-negative.")
        if (data.arc_capacities <= 0).any():
            raise ValueError("Arc capacities must be positive.")
        if (data.min_flows > data.max_flows).any():
            raise ValueError("Minimum flows must not exceed maximum flows.")

    def with_data(self, **changes: Any) -> "ProblemInstance":
        """Create a new numerical problem; preserve name, topology metadata and seed."""
        result = object.__new__(type(self))
        result._data = replace(self.data, **changes)
        result._network_graph = NetworkSnapshot.from_data(result.data, self.network_graph.nx_graph)
        result.service_package = self.service_package
        result.problem_name = self.problem_name
        result.seed = self.seed
        result._legacy_state = None
        result.validate_components()
        return result

    @property
    def number_of_service(self) -> int:
        return self.data.n_services

    @property
    def total_number_of_demands(self) -> int:
        return self.data.n_demands

    @property
    def demand_per_service(self) -> IntArray:
        return self.data.demand_counts

    @property
    def origins(self) -> IntArray:
        return self.data.origins

    @property
    def demand_to_service(self) -> IntArray:
        return self.data.demand_to_service

    @property
    def min_flows(self) -> FloatArray:
        return self.data.min_flows

    @property
    def max_flows(self) -> FloatArray:
        return self.data.max_flows

    @property
    def latencies(self) -> FloatArray:
        return self.data.latency_limits

    @property
    def needed_resources(self) -> FloatArray:
        return self.data.service_resources

    @staticmethod
    def _validate_source_components(
        network_graph: NetworkGraph, service_package: ServicePackage, problem_name: str
    ) -> None:
        assert isinstance(network_graph, NetworkGraph)
        assert isinstance(service_package, ServicePackage)
        assert problem_name

        graph = network_graph.nx_graph
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        assert n > 0
        assert m > 0
        assert len(network_graph.available_resources) == n
        assert len(network_graph.capacities) == m
        assert len(network_graph.alphas) == m
        assert len(network_graph.betas) == m
        assert all(resource >= 0 for resource in network_graph.available_resources)
        assert all(capacity > 0 for capacity in network_graph.capacities)
        assert all(alpha >= 0 for alpha in network_graph.alphas)
        assert all(beta >= 0 for beta in network_graph.betas)

        services = service_package.services
        assert services
        assert service_package.nx_graph is graph
        assert service_package.number_of_services == len(services)
        assert all(
            service.task_number == service_package.number_of_demand_per_service
            for service in services
        )
        for service in services:
            assert service.network_graph is network_graph
            assert service.task_number == len(service.task_list)
            assert service.needed_resources >= 0
            assert len(service.origin_nodes) == service.task_number
            assert len(service.min_flows) == service.task_number
            assert len(service.max_flows) == service.task_number
            assert len(service.max_latencies) == service.task_number
            assert all(0 <= origin < n for origin in service.origin_nodes)
            assert all(flow >= 0 for flow in service.min_flows)
            assert all(flow >= 0 for flow in service.max_flows)
            assert all(
                lo <= hi for lo, hi in zip(service.min_flows, service.max_flows, strict=False)
            )
            assert all(latency >= 0 for latency in service.max_latencies)
