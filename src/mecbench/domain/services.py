"""Demand, service and service-package data."""

from numbers import Integral, Real

import networkx as nx

from mecbench.domain.network import NetworkGraph


class Demand:
    """
    A demand is a certain amount of resources needed at a destination
    node and a flow in [flow_min, flow_max] that need to transit.
    """

    def __init__(
        self,
        origin_node_index: int,
        flow_min: float,
        flow_max: float,
        max_latency: float,
    ):
        if not isinstance(origin_node_index, Integral) or isinstance(origin_node_index, bool):
            raise TypeError("origin_node_index must be an int.")
        if not isinstance(flow_min, Real):
            raise TypeError("flow_min must be numeric.")
        if not isinstance(flow_max, Real):
            raise TypeError("flow_max must be numeric.")
        if not isinstance(max_latency, Real):
            raise TypeError("max_latency must be numeric.")
        if origin_node_index < 0:
            raise ValueError("origin_node_index must be non-negative.")
        if flow_min < 0:
            raise ValueError("flow_min must be non-negative.")
        if flow_max < 0:
            raise ValueError("flow_max must be non-negative.")
        if flow_min > flow_max:
            raise ValueError("flow_min must be <= flow_max.")
        if max_latency < 0:
            raise ValueError("max_latency must be non-negative.")

        self.origin_node_index = origin_node_index
        self.flow_min = flow_min
        self.flow_max = flow_max
        self.max_latency = max_latency


class Service:
    def __init__(
        self,
        task_list: list[Demand],
        network_graph: NetworkGraph,
        resources_needed: float,
    ):
        if not isinstance(network_graph, NetworkGraph):
            raise TypeError("network_graph must be a NetworkGraph.")
        if resources_needed < 0:
            raise ValueError("resources_needed must be non-negative.")
        if not task_list:
            raise ValueError("task_list must not be empty.")
        if not all(isinstance(task, Demand) for task in task_list):
            raise TypeError("task_list must contain only Demand instances.")

        self.network_graph = network_graph
        self.node_number = self.network_graph.nx_graph.number_of_nodes()
        self.edge_number = self.network_graph.nx_graph.number_of_edges()
        self.needed_resources = resources_needed

        self.task_number = 0
        self.task_list = []

        self.origin_nodes = []
        self.min_flows = []
        self.max_flows = []
        self.max_latencies = []

        for task in task_list:
            self.add_task(task)

    def add_task(self, task_element: Demand):
        self._check_task(task_element)

        self.current_task_element = task_element
        self.task_list.append(self.current_task_element)
        self.task_number += 1

        self.origin_nodes.append(self.current_task_element.origin_node_index)
        self.min_flows.append(self.current_task_element.flow_min)
        self.max_flows.append(self.current_task_element.flow_max)
        self.max_latencies.append(self.current_task_element.max_latency)

    def _check_task(self, task: Demand):
        if not 0 <= task.origin_node_index < self.node_number:
            raise ValueError(f"Demand origin node {task.origin_node_index} is outside graph nodes.")


class ServicePackage:
    """A list of Services."""

    def __init__(self, services: list[Service], instance_name: str) -> None:
        if not services:
            raise ValueError("services must not be empty.")
        if not instance_name:
            raise ValueError("instance_name must not be empty.")
        if not all(isinstance(service, Service) for service in services):
            raise TypeError("services must contain only Service instances.")

        first_network_graph = services[0].network_graph
        first_task_number = services[0].task_number
        for service in services:
            if service.network_graph is not first_network_graph:
                raise ValueError("All services must use the same NetworkGraph instance.")
            if service.task_number != first_task_number:
                raise ValueError("All services must have the same number of demands.")

        self.nx_graph: nx.Graph = services[0].network_graph.nx_graph

        self.services: list[Service] = []
        for service in services:
            self.services.append(service)

        self.instance_name = instance_name

        self.number_of_services: int = len(services)
        self.number_of_demand_per_service: int = services[0].task_number
        self.number_of_nodes = self.nx_graph.number_of_nodes()
        self.number_of_edges = self.nx_graph.number_of_edges()
