"""Mutable graphs, candidate state and routing caches owned by one solver."""

import networkx as nx
import numpy as np

from mecbench.compat.workspace import LegacyWorkspaceOperations
from mecbench.domain.contracts import IntArray
from mecbench.domain.conversions import to_arc_solution
from mecbench.domain.loads import residual_arrays
from mecbench.domain.network import NetworkSnapshot
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.problem_arrays import problem_arrays
from mecbench.domain.solution import ArcSolution, CompactSolution
from mecbench.domain.validation import validate, validate_arc_solution
from mecbench.utils.heuristic_solvers import basic_solution, remaining_quantity
from mecbench.utils.solver_paths import ShortestPathCache


def apply_residuals_to_graph(problem, graph: nx.Graph, x, y, q) -> nx.Graph:
    residual = residual_arrays(problem, x, y, q)
    for u in graph.nodes:
        graph.nodes[u]["remaining_resources"] = residual["remaining_resources"][u]

    for a, e in enumerate(problem_arrays(problem).arcs):
        if not graph.has_edge(*e):
            continue
        graph.edges[e]["remaining_capacity"] = residual["remaining_capacity"][a]
        graph.edges[e]["latency"] = residual["arc_latency"][a]
    return graph


class SolverWorkspace(LegacyWorkspaceOperations):
    def __init__(self, problem_instance: ProblemInstance, verbose: bool = True, *, seed: int = 0):
        self.problem_instance = problem_instance
        self.data = problem_arrays(problem_instance)
        self.graph = NetworkSnapshot.make_graph(
            self.data, self.problem_instance.network_graph.nx_graph
        )
        self.path_cache = ShortestPathCache(self.graph)
        self.saved_shortest_path = self.path_cache._paths
        self._edge_index_cache: dict[tuple[int, int, int], IntArray] = {}
        self.rng = np.random.default_rng(seed)
        self.current_solution: CompactSolution | ArcSolution | None = None
        self.initialize_graph_attributes(self.graph)

    def initialize_graph_attributes(self, graph: nx.Graph) -> nx.Graph:
        for v in graph.nodes:
            graph.nodes[v]["remaining_resources"] = graph.nodes[v]["available_resources"]
        for e in graph.edges:
            graph.edges[e]["remaining_capacity"] = graph.edges[e]["capacity"]
            graph.edges[e]["latency"] = graph.edges[e]["beta"]
        return graph

    def update_graph(
        self, solution: CompactSolution | ArcSolution, *, graph: nx.Graph | None = None
    ) -> nx.Graph:
        """Apply residuals from either representation to the selected working graph."""
        arc = to_arc_solution(self.data, solution, path_resolver=self.compute_nth_shortest_path)
        target = self.graph if graph is None else graph
        return apply_residuals_to_graph(self.data, target, arc.x, arc.y, arc.q)

    def basic_placement_and_routing(self) -> tuple:
        return basic_solution(self.data, self.graph)

    @property
    def current_arc_solution(self) -> ArcSolution | None:
        if self.current_solution is None:
            return None
        return to_arc_solution(
            self.data, self.current_solution, path_resolver=self.compute_nth_shortest_path
        )

    def render_solution(
        self,
        placement: list = None,
        routing: list = None,
        quantities: list = None,
        **plot_kwargs,
    ):
        from mecbench.benchmarks.reporting import plot_instance_solution

        provided = (placement is not None, routing is not None, quantities is not None)
        if any(provided) and not all(provided):
            raise ValueError("placement, routing, and quantities must be provided together.")

        if not any(provided):
            solution = self.current_arc_solution
            if solution is None:
                raise ValueError("No solution has been stored or provided.")
            return plot_instance_solution(self.problem_instance, *solution, **plot_kwargs)

        return plot_instance_solution(
            self.problem_instance,
            *self.solution_to_arc_variables(placement, routing, quantities),
            **plot_kwargs,
        )

    def solution_to_arc_variables(self, placement, routing, quantities):
        solution = to_arc_solution(
            self.data,
            CompactSolution(placement, routing, quantities),
            path_resolver=self.compute_nth_shortest_path,
        )
        return solution.x, solution.y, solution.q

    def path_edge_indices(self, source: int, destination: int, index: int) -> IntArray:
        key = (int(source), int(destination), int(index))
        cached = self._edge_index_cache.get(key)
        if cached is not None:
            return cached
        path = self.compute_nth_shortest_path(source, destination, index)
        arc_index = self.data.arc_index
        indices = np.fromiter(
            (arc_index[(path[i], path[i + 1])] for i in range(len(path) - 1)),
            dtype=np.int64,
            count=max(0, len(path) - 1),
        )
        cached = np.frombuffer(indices.tobytes(), dtype=np.int64)
        self._edge_index_cache[key] = cached
        return cached

    def compact_path_indices(self, placement, routing) -> list[IntArray]:
        return [
            self.path_edge_indices(
                self.data.origins[d],
                placement[self.data.demand_to_service[d]],
                routing[d],
            )
            for d in range(self.data.n_demands)
        ]

    def compact_loads(self, placement, routing, quantities, *, paths=None):
        """Calculate loads; reuse supplied paths while placement/routing stay fixed."""
        if paths is None:
            paths = self.compact_path_indices(placement, routing)
        q = np.asarray(quantities, dtype=float)
        edge_load = np.zeros(self.data.n_arcs)
        for d, idx in enumerate(paths):
            edge_load[idx] += q[d]

        node_load = np.bincount(
            np.asarray(placement, dtype=int),
            weights=self.data.service_resources,
            minlength=self.data.n_nodes,
        )
        return paths, edge_load, node_load

    def is_feasible(self, placement, routing, quantities) -> bool:
        try:
            candidate = CompactSolution(placement, routing, quantities)
            return validate(
                self.data, candidate, path_resolver=self.compute_nth_shortest_path
            ).feasible
        except (IndexError, ValueError, TypeError, KeyError, nx.NetworkXException):
            return False

    # Historical names share the implementation, without additional wrappers.
    test_solution = is_feasible
    test_compact_solution = is_feasible

    def get_latency_from_graph(self, path_edge: list) -> float:
        return sum(self.graph.edges[e]["latency"] for e in path_edge)

    def maximize_quantities(self, placement, routing, quantities, addten=True):
        assert self.is_feasible(placement, routing, quantities)
        step = 10 * self.data.n_demands * self.data.max_flows[0]
        q = list(quantities)
        for _ in range(int(step)):
            q_new = q.copy()
            d = self.rng.integers(self.data.n_demands)
            q_new[d] += 10 if addten else 1
            if self.is_feasible(placement, routing, q_new):
                q = q_new
        return placement, routing, q

    def _maximize_quantities_greedy(self):
        placement, routing, quantities = self.basic_placement_and_routing()
        solution = CompactSolution(
            *self.maximize_quantities(placement, routing, quantities, addten=False)
        )
        return self.set_solution(solution)

    def reset_instance(self):
        self.graph = self.initialize_graph_attributes(self.graph)
        return self.graph

    def set_variable_solution(self, x_variable, y_variable, q_variable):
        return self.set_solution(ArcSolution(x_variable, y_variable, q_variable))

    def set_solution(
        self, solution: CompactSolution | ArcSolution
    ) -> CompactSolution | ArcSolution:
        """Store exactly the supplied representation after reference validation."""
        arc = to_arc_solution(self.data, solution, path_resolver=self.compute_nth_shortest_path)
        validate_arc_solution(self.data, arc.x, arc.y, arc.q)
        self.update_graph(arc)
        self.current_solution = solution
        return solution

    def _get_remaining_quantity_per_demand(self, placement, routing, quantities, demand_index):
        x, y, q = self.solution_to_arc_variables(placement, routing, quantities)
        return remaining_quantity(self.data, x, y, q, demand_index)

    def compute_nth_shortest_path(self, source: int, destination: int, index: int) -> list:
        return self.path_cache.nth_shortest_path(int(source), int(destination), int(index))

    def test_basic_placement(self, print_solution=False):
        placement, routing, quantities = self.basic_placement_and_routing()
        works = self.is_feasible(placement, routing, quantities)
        if works:
            print("Basic placement and routing works.")
        if print_solution:
            self.render_solution(placement, routing, quantities)
        assert works, "Basic placement and routing does not work."


SolvingInstance = SolverWorkspace
