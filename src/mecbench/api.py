"""Small public interface for building, solving and validating instances."""

import inspect
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from time import perf_counter
from typing import Any

import numpy as np

from mecbench.domain.conversions import SolutionConversionError, to_arc_solution
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.solution import ArcSolution, CompactSolution
from mecbench.domain.validation import ValidationReport, validate
from mecbench.errors import SolverExecutionError
from mecbench.instances.synthetic import create_star_problem_instance, create_tree_problem_instance
from mecbench.utils.solver_registry import REGISTRY, canonical_name, dependency_error, make_solver


class SolveStatus(str, Enum):
    FEASIBLE = "feasible"
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    TIME_LIMIT = "time_limit"
    NO_SOLUTION = "no_solution"
    INVALID_SOLUTION = "invalid_solution"
    UNBOUNDED = "unbounded"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SolveResult:
    solution: ArcSolution | None
    objective: float | None
    status: SolveStatus
    runtime: float
    build_time: float
    solver: str
    seed: int
    validation: ValidationReport
    termination: str | None = None


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def make_tree_instance(n_users: int = 20, neighbors: int = 1, *, seed: int = 0) -> ProblemInstance:
    """Build the deterministic research tree topology; seed is recorded as metadata.

    The current tree generator has no random choices. Changing seed therefore
    does not change its topology or demands; solve's seed controls the search.
    """
    n_users = _integer(n_users, "n_users", minimum=1)
    neighbors = _integer(neighbors, "neighbors")
    seed = _integer(seed, "seed")
    problem = create_tree_problem_instance(n_users, neighbors)
    problem.seed = seed
    return problem


def make_star_instance(
    n_groups: int, users_per_group: int, neighbors: int = 1, *, seed: int = 0
) -> ProblemInstance:
    """Build a deterministic grouped star topology with validated parameters."""
    n_groups = _integer(n_groups, "n_groups", minimum=1)
    users_per_group = _integer(users_per_group, "users_per_group", minimum=1)
    neighbors = _integer(neighbors, "neighbors")
    seed = _integer(seed, "seed")
    problem = create_star_problem_instance(n_groups, users_per_group, neighbors)
    problem.seed = seed
    return problem


def _termination_status(termination: str | None) -> SolveStatus | None:
    normalized = str(termination or "").lower().replace("_", "").replace(" ", "")
    return {
        "optimal": SolveStatus.OPTIMAL,
        "globallyoptimal": SolveStatus.OPTIMAL,
        "infeasible": SolveStatus.INFEASIBLE,
        "unbounded": SolveStatus.UNBOUNDED,
        "maxtimelimit": SolveStatus.TIME_LIMIT,
        "timelimit": SolveStatus.TIME_LIMIT,
    }.get(normalized)


def solve(
    problem: ProblemInstance,
    *,
    solver: str = "fast_heuristic",
    seed: int = 0,
    time_limit: float | None = None,
    solver_options: dict[str, Any] | None = None,
    run_options: dict[str, Any] | None = None,
) -> SolveResult:
    """Run a backend lazily and validate its candidate with the reference checks.

    ``solver_options`` configures construction, ``run_options`` configures run.
    ``time_limit`` is in seconds and is supported by mathematical backends only.
    An invalid candidate is retained for diagnosis but has no objective. Missing
    dependencies and execution failures raise actionable, typed exceptions.
    """
    name = canonical_name(solver)
    seed = _integer(seed, "seed")
    if time_limit is not None and (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, int | float)
        or not np.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive.")
    if time_limit is not None and not REGISTRY[name].mathematical:
        raise ValueError(f"Solver {name!r} does not support time_limit.")
    constructor_options = dict(solver_options or {})
    if "seed" in constructor_options:
        raise ValueError("Pass seed directly to solve(), not in solver_options.")
    start = perf_counter()
    backend = make_solver(name, problem, seed=seed, **constructor_options)
    kwargs = dict(run_options or {})
    if REGISTRY[name].mathematical:
        if "options" not in inspect.signature(backend.run).parameters:
            raise ValueError(f"Solver {name!r} cannot receive optimizer options through this API.")
        optimizer_options = dict(kwargs.get("options") or {})
        optimizer_options["Seed"] = seed
        if time_limit is not None:
            optimizer_options["TimeLimit"] = float(time_limit)
        kwargs["options"] = optimizer_options
    inspect.signature(backend.run).bind(**kwargs)
    try:
        backend.run(**kwargs)
    except ModuleNotFoundError as error:
        raise dependency_error(name, error) from error
    except Exception as error:
        hint = " Check your Gurobi installation and license." if REGISTRY[name].mathematical else ""
        raise SolverExecutionError(f"Solver {name!r} failed: {error}.{hint}") from error

    termination = getattr(
        backend, "termination_condition", getattr(backend, "solving_status", None)
    )
    termination = str(termination) if termination is not None else None
    status = _termination_status(termination)
    # An infeasibility proof/unbounded model has no usable incumbent. Some legacy
    # backends supply a baseline in these cases; do not present it as a result.
    solution: ArcSolution | None = None
    candidate: ArcSolution | CompactSolution | None = None
    report: ValidationReport | None = None
    if status not in (SolveStatus.INFEASIBLE, SolveStatus.UNBOUNDED):
        output = getattr(backend, "solution", None)
        if output is not None:
            if not isinstance(output, ArcSolution | CompactSolution):
                raise SolverExecutionError(
                    f"Solver {name!r} produced an unsupported solution type."
                )
            candidate = output
        else:
            # Compatibility with external backends that still publish three arrays.
            x, y, q = backend.x_var, backend.y_var, backend.q_var
            if x is not None and y is not None and q is not None:
                try:
                    candidate = ArcSolution(x, y, q)
                except (TypeError, ValueError) as error:
                    raise SolverExecutionError(
                        f"Solver {name!r} produced non-numeric solution arrays: {error}"
                    ) from error
        if candidate is not None:
            try:
                solution = to_arc_solution(
                    problem, candidate, path_resolver=getattr(backend, "path_resolver", None)
                )
            except SolutionConversionError as error:
                report = ValidationReport.from_conversion_error(error)
            if (
                solution is not None
                and REGISTRY[name].mathematical
                and not all(np.all(np.isfinite(v)) for v in solution)
            ):
                solution = candidate = None
    if report is None:
        report = validate(problem, solution)
    if report.feasible:
        status = status or SolveStatus.FEASIBLE
    elif candidate is not None:
        status = SolveStatus.INVALID_SOLUTION
    else:
        status = (
            status
            if status in (SolveStatus.INFEASIBLE, SolveStatus.UNBOUNDED, SolveStatus.TIME_LIMIT)
            else SolveStatus.NO_SOLUTION
        )
    return SolveResult(
        solution,
        float(np.sum(solution.q)) if report.feasible and solution is not None else None,
        status,
        perf_counter() - start,
        max(0.0, float(getattr(backend, "build_time", 0) or 0)),
        name,
        seed,
        report,
        termination,
    )
