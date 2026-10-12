"""Pure compact/state/arc representation conversions."""

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Any, TypeAlias

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray

from mecbench.domain.contracts import ArcArrays, IntArray, ProblemData
from mecbench.domain.problem_arrays import ProblemArrays, problem_arrays
from mecbench.domain.solution import ArcSolution, CompactSolution

PathResolver: TypeAlias = Callable[[int, int, int], Sequence[int]]


class SolutionConversionError(ValueError):
    """A compact candidate cannot be represented by arc variables."""

    def __init__(
        self,
        message: str,
        *,
        constraint: str = "routing",
        variable: str | None = None,
        demand: int | None = None,
        observed: tuple[int, ...] | None = None,
        limit: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.constraint = constraint
        self.variable = variable
        self.demand = demand
        self.observed = observed
        self.limit = limit


def _check_compact_vectors(data: ProblemArrays, solution: CompactSolution) -> None:
    for name, values, size in (
        ("placement", solution.placement, data.n_services),
        ("routing", solution.routing, data.n_demands),
        ("quantities", solution.quantities, data.n_demands),
    ):
        if values.shape != (size,):
            raise SolutionConversionError(
                f"{name} shape must be ({size},), got {values.shape}.",
                constraint="dimensions",
                variable=name,
                observed=values.shape,
                limit=(size,),
            )
        if not np.all(np.isfinite(values)):
            raise SolutionConversionError(
                f"{name} must contain only finite values.", constraint="finite", variable=name
            )
    if np.any((solution.placement < 0) | (solution.placement >= data.n_nodes)):
        raise SolutionConversionError(
            "placement contains an unknown node.", constraint="placement", variable="placement"
        )
    if np.any(solution.routing < 0):
        raise SolutionConversionError("routing contains a negative path index.", variable="routing")


def to_arc_solution(
    problem: ProblemData | ProblemArrays,
    solution: CompactSolution | ArcSolution,
    *,
    path_resolver: PathResolver | None = None,
) -> ArcSolution:
    """Normalize a candidate without rounding quantities or inferring path indices."""
    if isinstance(solution, ArcSolution):
        return solution
    if not isinstance(solution, CompactSolution):
        raise TypeError("solution must be a CompactSolution or an ArcSolution.")
    data = problem_arrays(problem)
    _check_compact_vectors(data, solution)
    if path_resolver is None:
        raise SolutionConversionError(
            "A CompactSolution requires its path_resolver.", constraint="path_resolution"
        )
    y = np.zeros((data.n_services, data.n_nodes))
    y[np.arange(data.n_services), solution.placement] = 1.0
    x = np.zeros((data.n_demands, data.n_arcs))
    for d in range(data.n_demands):
        source = int(data.origins[d])
        destination = int(solution.placement[data.demand_to_service[d]])
        index = int(solution.routing[d])
        try:
            path = list(path_resolver(source, destination, index))
            if (
                not path
                or any(not isinstance(node, Integral) for node in path)
                or path[0] != source
                or path[-1] != destination
                or len(set(path)) != len(path)
            ):
                raise ValueError(
                    "The resolver must return a simple path with the requested endpoints."
                )
            for arc in zip(path, path[1:], strict=False):
                x[d, data.arc_index[arc]] = 1.0
        except (IndexError, KeyError, TypeError, ValueError, nx.NetworkXException) as error:
            raise SolutionConversionError(
                f"Cannot resolve routing[{d}]={index} from {source} to {destination}: {error}",
                variable="routing",
                demand=d,
            ) from error
    return ArcSolution(x, y, solution.quantities)


def compact_from_state(problem: ProblemData | ProblemArrays, state: ArrayLike) -> CompactSolution:
    """Snapshot an encoded state without rounding its indices or quantities."""
    data = problem_arrays(problem)
    values = np.asarray(state)
    size = data.n_services + 2 * data.n_demands
    if values.shape != (size,):
        raise SolutionConversionError(
            f"state shape must be ({size},), got {values.shape}.",
            constraint="dimensions",
            variable="state",
            observed=values.shape,
            limit=(size,),
        )
    start = data.n_services
    return CompactSolution(
        values[:start], values[start : start + data.n_demands], values[start + data.n_demands :]
    )


def to_int_array(x: ArrayLike) -> IntArray:
    return np.asarray(np.rint(np.asarray(x, dtype=float)), dtype=np.int64)


def split_state(
    problem: ProblemData | ProblemArrays, s: ArrayLike
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    data = problem_arrays(problem)
    s = np.asarray(s)
    S = data.n_services
    D = data.n_demands
    return s[:S], s[S : S + D], s[S + D : S + 2 * D]


def make_state(placement: ArrayLike, routing: ArrayLike, quantities: ArrayLike) -> IntArray:
    return to_int_array(np.concatenate((placement, routing, quantities)))


def validate_compact_shapes(
    problem: ProblemData | ProblemArrays,
    graph: nx.Graph,
    placement: ArrayLike,
    routing: ArrayLike,
    quantities: ArrayLike,
) -> None:
    _check_compact_vectors(problem_arrays(problem), CompactSolution(placement, routing, quantities))


def compact_to_arc(
    problem: ProblemData | ProblemArrays,
    graph: nx.Graph,
    path_fn: PathResolver,
    placement: ArrayLike,
    routing: ArrayLike,
    quantities: ArrayLike,
) -> ArcArrays:
    solution = to_arc_solution(
        problem, CompactSolution(placement, routing, quantities), path_resolver=path_fn
    )
    return solution.x, solution.y, solution.q
