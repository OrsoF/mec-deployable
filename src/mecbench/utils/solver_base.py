"""Explicit constructor and output contracts shared by every solver."""

import time
from collections.abc import Callable
from typing import Protocol

import networkx as nx
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray
from mecbench.domain.conversions import PathResolver, to_arc_solution
from mecbench.domain.network import NetworkSnapshot
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.problem_arrays import problem_arrays
from mecbench.domain.solution import ArcSolution, CompactSolution


class SolverBackend(Protocol):
    score: float | None
    runtime: float | None
    build_time: float
    termination_condition: str | None

    @property
    def x_var(self) -> FloatArray | None: ...

    @property
    def y_var(self) -> FloatArray | None: ...

    @property
    def q_var(self) -> FloatArray | None: ...

    @property
    def solution(self) -> ArcSolution | CompactSolution | None: ...

    @property
    def path_resolver(self) -> PathResolver | None: ...

    @property
    def run(self) -> Callable[..., object]: ...


class GenericSolver:
    REQUIRED_OUTPUT_ATTRIBUTES = (
        "solution",
        "path_resolver",
        "x_var",
        "y_var",
        "q_var",
        "score",
        "runtime",
        "build_time",
    )

    def __init__(self, problem_instance: ProblemInstance) -> None:
        self.problem_instance = problem_instance
        self.data = problem_arrays(problem_instance)
        network = problem_instance.network_graph
        self._graph = (
            network.nx_graph
            if getattr(network, "data", None) is self.data
            else NetworkSnapshot.from_data(self.data, network.nx_graph).nx_graph
        )
        self.solver_name = "generic_solver"

        self._solution: ArcSolution | CompactSolution | None = None
        self._arc_solution: ArcSolution | None = None
        self._path_resolver: PathResolver | None = None

        self.score: float | None = None
        self.runtime: float | None = None
        self.build_time: float = 0.0
        self.termination_condition: str | None = None

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def solution(self) -> ArcSolution | CompactSolution | None:
        return self._solution

    @property
    def path_resolver(self) -> PathResolver | None:
        return self._path_resolver

    @property
    def arc_solution(self) -> ArcSolution | None:
        if self._arc_solution is None and self.solution is not None:
            self._arc_solution = to_arc_solution(
                self.data, self.solution, path_resolver=self.path_resolver
            )
        return self._arc_solution

    @property
    def x_var(self) -> FloatArray | None:
        solution = self.arc_solution
        return None if solution is None else solution.x

    @property
    def y_var(self) -> FloatArray | None:
        solution = self.arc_solution
        return None if solution is None else solution.y

    @property
    def q_var(self) -> FloatArray | None:
        solution = self.arc_solution
        return None if solution is None else solution.q

    # Historical array names delegate to the typed result too.
    x_variable = x_var
    y_variable = y_var
    q_variable = q_var

    def _set_solution(
        self,
        solution: ArcSolution | CompactSolution | None,
        *,
        path_resolver: PathResolver | None = None,
    ) -> None:
        if isinstance(solution, CompactSolution) and path_resolver is None:
            raise ValueError("Publishing a CompactSolution requires its path_resolver.")
        self._solution = solution
        self._path_resolver = path_resolver
        self._arc_solution = solution if isinstance(solution, ArcSolution) else None

    def _set_compact_solution(
        self,
        placement: ArrayLike,
        routing: ArrayLike,
        quantities: ArrayLike,
        *,
        path_resolver: PathResolver,
    ) -> None:
        self._set_solution(
            CompactSolution(placement, routing, quantities), path_resolver=path_resolver
        )

    def _set_output_variables(self, x_var: ArrayLike, y_var: ArrayLike, q_var: ArrayLike) -> None:
        self._set_solution(ArcSolution(x_var, y_var, q_var))

    def run(self) -> object:
        raise NotImplementedError("Subclasses must implement run().")

    def _start_solving(self) -> None:
        self.start_time = time.time()

    def _end_solving(self) -> None:
        if self.solution is None:
            raise RuntimeError("Result variables not set.")
        self.runtime = time.time() - self.start_time
