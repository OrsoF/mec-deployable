"""Reference feasibility checks, independent of solver implementations.

Binary routing/placement and continuous quantities match the arc MILP.
Flow conservation permits balanced circulations, as does the formulation.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mecbench.domain.contracts import FloatArray, ProblemData, SolutionData
from mecbench.domain.conversions import PathResolver, SolutionConversionError, to_arc_solution
from mecbench.domain.loads import residual_arrays
from mecbench.domain.problem_arrays import ProblemArrays, problem_arrays
from mecbench.domain.solution import CompactSolution


class SolutionValidationError(ValueError):
    """A solution violates its dimensions, variable domains, or constraints."""


@dataclass(frozen=True)
class Violation:
    """A constraint violation with zero-based entity indices and measured values."""

    constraint: str
    message: str
    observed: float | tuple[int, ...] | None = None
    limit: float | tuple[int, ...] | None = None
    demand: int | None = None
    service: int | None = None
    node: int | None = None
    arc: int | None = None
    variable: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    violations: tuple[Violation, ...] = ()

    @property
    def feasible(self) -> bool:
        return not self.violations

    def raise_if_infeasible(self) -> None:
        if self.violations:
            raise SolutionValidationError(self.violations[0].message)

    @classmethod
    def from_conversion_error(cls, error: SolutionConversionError) -> "ValidationReport":
        return cls(
            (
                Violation(
                    error.constraint,
                    str(error),
                    error.observed,
                    error.limit,
                    demand=error.demand,
                    variable=error.variable,
                ),
            )
        )


def validate(
    problem: ProblemData | ProblemArrays,
    solution: SolutionData | CompactSolution | None,
    *,
    atol: float = 1e-6,
    path_resolver: PathResolver | None = None,
) -> ValidationReport:
    """Report all violations; malformed arrays stop dependent physical checks.

    Compact candidates require their path resolver; arc candidates expose x, y
    and q. ``None`` represents absence of a candidate. Inputs are never changed.
    """
    _check_tolerance(atol)
    if solution is None:
        return ValidationReport((Violation("no_solution", "No solution available."),))
    if isinstance(solution, CompactSolution):
        try:
            solution = to_arc_solution(problem, solution, path_resolver=path_resolver)
        except SolutionConversionError as error:
            return ValidationReport.from_conversion_error(error)
    if not all(hasattr(solution, name) for name in ("x", "y", "q")):
        raise TypeError("solution must be compact, expose x, y and q arrays, or be None.")
    return _validate_arrays(problem, solution.x, solution.y, solution.q, atol=atol)


def _check_tolerance(atol: object) -> None:
    if not isinstance(atol, Real) or not isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")


def _validate_arrays(
    problem: ProblemData | ProblemArrays,
    x_values: ArrayLike,
    y_values: ArrayLike,
    q_values: ArrayLike,
    *,
    atol: float = 1e-6,
) -> ValidationReport:
    _check_tolerance(atol)
    data = problem_arrays(problem)
    demands, services, nodes, arcs = data.n_demands, data.n_services, data.n_nodes, data.n_arcs
    violations = []
    arrays: dict[str, FloatArray] = {}
    for name, raw_values, shape in (
        ("x", x_values, (demands, arcs)),
        ("y", y_values, (services, nodes)),
        ("q", q_values, (demands,)),
    ):
        try:
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError):
            violations.append(
                Violation("numeric", f"{name} must be a numeric array.", variable=name)
            )
            continue
        if values.shape != shape:
            violations.append(
                Violation(
                    "dimensions",
                    f"{name} shape must be {shape}, got {values.shape}.",
                    values.shape,
                    shape,
                    variable=name,
                )
            )
        elif not np.all(np.isfinite(values)):
            violations.append(
                Violation("finite", f"{name} must contain only finite values.", variable=name)
            )
        else:
            arrays[name] = values
    if violations:
        return ValidationReport(tuple(violations))
    x, y, q = (arrays[name] for name in ("x", "y", "q"))
    for name, values in (("x", x), ("y", y)):
        rounded = np.rint(values)
        if np.any(np.abs(values - rounded) > atol) or np.any((rounded < 0) | (rounded > 1)):
            violations.append(
                Violation(
                    "binary", f"{name} must be binary within tolerance {atol}.", variable=name
                )
            )
        arrays[name] = rounded
    if violations:
        return ValidationReport(tuple(violations))
    x, y = arrays["x"], arrays["y"]

    def record(
        mask: NDArray[np.bool_],
        constraint: str,
        message: str,
        observed: FloatArray,
        limit: FloatArray,
        entity: Literal["demand", "service", "node", "arc"],
    ) -> None:
        for index in np.flatnonzero(mask):
            i = int(index)
            violations.append(
                Violation(
                    constraint,
                    message,
                    float(observed[i]),
                    float(limit[i]),
                    demand=i if entity == "demand" else None,
                    service=i if entity == "service" else None,
                    node=i if entity == "node" else None,
                    arc=i if entity == "arc" else None,
                )
            )

    placements = y.sum(axis=1)
    record(
        placements != 1,
        "placement",
        "Each service must be placed on exactly one node.",
        placements,
        np.ones(services),
        "service",
    )
    minimum = np.maximum(0, data.min_flows)
    maximum = data.max_flows
    record(
        q < minimum - atol,
        "minimum_flow",
        "Demand quantity below minimum flow.",
        q,
        minimum,
        "demand",
    )
    record(
        q > maximum + atol,
        "maximum_flow",
        "Demand quantity above maximum flow.",
        q,
        maximum,
        "demand",
    )
    loads = residual_arrays(data, x, y, q)
    node_load = loads["node_load"]
    resources = data.node_resources
    record(
        node_load > resources + atol,
        "resources",
        "Node resources exceeded.",
        node_load,
        resources,
        "node",
    )
    edge_load = loads["edge_load"]
    capacities = data.arc_capacities
    record(
        edge_load > capacities + atol,
        "capacity",
        "Arc capacity exceeded.",
        edge_load,
        capacities,
        "arc",
    )

    balance = np.zeros((demands, nodes))
    for a, (source, destination) in enumerate(data.arcs):
        balance[:, source] += x[:, a]
        balance[:, destination] -= x[:, a]
    expected = -y[data.demand_to_service].copy()
    expected[np.arange(demands), data.origins] += 1
    for d, node in np.argwhere(np.abs(balance - expected) > atol):
        violations.append(
            Violation(
                "conservation",
                "Flow conservation violated.",
                float(balance[d, node]),
                float(expected[d, node]),
                demand=int(d),
                node=int(node),
            )
        )

    demand_latency = loads["demand_latency"]
    limits = data.latency_limits
    record(
        demand_latency > limits + atol,
        "latency",
        "Demand latency exceeded under total arc load.",
        demand_latency,
        limits,
        "demand",
    )
    return ValidationReport(tuple(violations))


def validate_arc_solution(
    problem: ProblemData | ProblemArrays,
    x: ArrayLike,
    y: ArrayLike,
    q: ArrayLike,
    *,
    atol: float = 1e-6,
) -> None:
    """Strict compatibility entry point using the same checks as ``validate``."""
    _validate_arrays(problem, x, y, q, atol=atol).raise_if_infeasible()
