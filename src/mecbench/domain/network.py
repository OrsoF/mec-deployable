"""Network data and construction helpers."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from mecbench.domain.contracts import FloatArray

if TYPE_CHECKING:
    from mecbench.domain.problem_arrays import ProblemArrays


@dataclass(frozen=True)
class NetworkSnapshot:
    """Compatibility view of canonical numbers and a stable copied topology."""

    data: "ProblemArrays"
    nx_graph: nx.DiGraph

    @classmethod
    def from_data(cls, data: "ProblemArrays", template: nx.Graph) -> "NetworkSnapshot":
        snapshot = cls(data, nx.freeze(cls.make_graph(data, template)))
        return snapshot

    @staticmethod
    def make_graph(data: "ProblemArrays", template: nx.Graph) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.graph.update(deepcopy(template.graph))
        for node in range(data.n_nodes):
            graph.add_node(node, **deepcopy(template.nodes[node]) if node in template else {})
            graph.nodes[node]["available_resources"] = float(data.node_resources[node])
        for a, arc in enumerate(data.arcs):
            graph.add_edge(*arc, **deepcopy(template.edges[arc]) if template.has_edge(*arc) else {})
            graph.edges[arc].update(
                capacity=float(data.arc_capacities[a]),
                alpha=float(data.latency_slopes[a]),
                beta=float(data.latency_offsets[a]),
            )
        return graph

    @property
    def alphas(self) -> FloatArray:
        return self.data.latency_slopes

    @property
    def betas(self) -> FloatArray:
        return self.data.latency_offsets

    @property
    def capacities(self) -> FloatArray:
        return self.data.arc_capacities

    @property
    def available_resources(self) -> FloatArray:
        return self.data.node_resources

    @property
    def arc_list(self) -> tuple[tuple[int, int], ...]:
        return self.data.arcs

    @property
    def arc_index_dictionary(self) -> Mapping[tuple[int, int], int]:
        return self.data.arc_index


def check_network_graph(nx_graph: nx.DiGraph):
    """
    Checks if the graph is a DiGraph and is symmetric.
    """
    assert isinstance(nx_graph, nx.DiGraph), "Considered graph is not a DiGraph instance."
    assert is_graph_symmetric(nx_graph), "Considered graph is not symmetric."


def is_graph_symmetric(nx_graph: nx.DiGraph) -> bool:
    """
    Check if a given DiGraph is symmetric.
    """
    for edge in nx_graph.edges:
        if (edge[1], edge[0]) not in nx_graph.edges:
            return False
    return True


class NetworkGraph:
    def __init__(
        self,
        base_graph: nx.Graph,
        node_resources: list[float],
        edge_capacity: list[float],
        latency_alpha_factor: list[float],
        latency_beta_factor: list[float],
    ):
        # The graph needs to be symmetric. We label nodes with integers.
        if not isinstance(base_graph, nx.DiGraph):
            self.nx_graph = nx.DiGraph(base_graph)
            self.nx_graph = nx.relabel.relabel_nodes(
                self.nx_graph, {node: i for i, node in enumerate(self.nx_graph.nodes())}
            )
            self.make_graph_symmetric()
            self.remove_all_attributes()
        else:
            self.nx_graph = base_graph
            self.make_graph_symmetric()

        check_network_graph(self.nx_graph)

        self.set_node_attribute(node_resources, "available_resources")
        self.set_edge_attribute(edge_capacity, "capacity")
        self.set_edge_attribute(latency_alpha_factor, "alpha")
        self.set_edge_attribute(latency_beta_factor, "beta")

        self.alphas: list[float] = [
            self.nx_graph.edges[edge]["alpha"] for edge in self.nx_graph.edges
        ]
        self.betas: list[float] = [
            self.nx_graph.edges[edge]["beta"] for edge in self.nx_graph.edges
        ]
        self.capacities = [self.nx_graph.edges[edge]["capacity"] for edge in self.nx_graph.edges]
        self.available_resources = [
            self.nx_graph.nodes[node]["available_resources"] for node in self.nx_graph.nodes
        ]
        self.arc_index_dictionary = {arc: i for i, arc in enumerate(self.nx_graph.edges)}
        self.arc_list = list(self.nx_graph.edges)

    def set_node_attribute(self, attributes: list[float], name: str):
        assert len(attributes) == self.nx_graph.number_of_nodes(), (
            "There not as many node parameters as the number of nodes in the graph."
        )
        attribute_map = {
            node_index: attributes[node_index]
            for node_index in range(self.nx_graph.number_of_nodes())
        }
        nx.classes.function.set_node_attributes(self.nx_graph, attribute_map, name)

    def set_edge_attribute(self, attributes: list[float], name: str):
        edges = list(self.nx_graph.edges)

        if len(attributes) == len(edges) // 2:
            undirected_edges = []
            seen = set()
            for u, v in edges:
                if (v, u) in seen:
                    continue
                undirected_edges.append((u, v))
                seen.add((u, v))

            assert len(attributes) == len(undirected_edges), (
                f"{len(attributes)} parameters not equal to {len(undirected_edges)} undirected edges in graph."
            )
            attribute_map = {}
            for edge, value in zip(undirected_edges, attributes, strict=False):
                u, v = edge
                attribute_map[(u, v)] = value
                attribute_map[(v, u)] = value
        else:
            assert len(attributes) == len(edges), (
                f"{len(attributes)} parameters not equal to {len(edges)} edges in graph."
            )
            attribute_map = {edge: attributes[edge_index] for edge_index, edge in enumerate(edges)}
        nx.classes.function.set_edge_attributes(self.nx_graph, attribute_map, name)

    def make_graph_symmetric(self):
        for arc in self.nx_graph.edges:
            if (arc[1], arc[0]) not in self.nx_graph.edges:
                self.nx_graph.add_edge(arc[1], arc[0])

    def remove_all_attributes(self):
        for node in self.nx_graph.nodes:
            keys = list(self.nx_graph.nodes[node].keys())
            for key in keys:
                del self.nx_graph.nodes[node][key]
        for edge in self.nx_graph.edges:
            keys = list(self.nx_graph.edges[edge].keys())
            for key in keys:
                del self.nx_graph.edges[edge][key]
        return self.nx_graph

    def draw_graph(self):
        nx.draw(self.nx_graph, with_labels=True)
