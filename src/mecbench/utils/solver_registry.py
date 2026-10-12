"""One lazy registry for the API, CLI and historical benchmark helpers."""

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from mecbench.domain.problem import ProblemInstance
from mecbench.errors import MissingSolverDependencyError, UnknownSolverError
from mecbench.utils.solver_base import SolverBackend


@dataclass(frozen=True)
class SolverSpec:
    module: str
    extra: str | None = None
    mathematical: bool = False


REGISTRY = {
    "ilp_arc": SolverSpec("mecbench.solvers.ilp_arc", extra="exact", mathematical=True),
    "ilp_arc_gap_5pct": SolverSpec(
        "mecbench.solvers.ilp_arc_gap_5pct", extra="exact", mathematical=True
    ),
    "ilp_path": SolverSpec("mecbench.solvers.ilp_path", extra="exact", mathematical=True),
    "milp_arc": SolverSpec("mecbench.solvers.milp_arc", extra="exact", mathematical=True),
    "milp_path": SolverSpec("mecbench.solvers.milp_path", extra="exact", mathematical=True),
    "milp_arc_full": SolverSpec("mecbench.solvers.milp_arc_full", extra="exact", mathematical=True),
    "ilp_arc_heuristic_placement": SolverSpec(
        "mecbench.solvers.ilp_arc_heuristic_placement", extra="exact", mathematical=True
    ),
    "ilp_arc_heuristic_warmstart": SolverSpec(
        "mecbench.solvers.ilp_arc_heuristic_warmstart", extra="exact", mathematical=True
    ),
    "ilp_arc_rl_warmstart": SolverSpec(
        "mecbench.solvers.ilp_arc_rl_warmstart", extra="exact,rl", mathematical=True
    ),
    "lp_arc": SolverSpec("mecbench.solvers.lp_arc", extra="exact", mathematical=True),
    "lp_path": SolverSpec("mecbench.solvers.lp_path", extra="exact", mathematical=True),
    "lp_path_latency_tightened": SolverSpec(
        "mecbench.solvers.lp_path_latency_tightened", extra="exact", mathematical=True
    ),
    "a2c": SolverSpec("mecbench.solvers.a2c", extra="rl"),
    "ppo": SolverSpec("mecbench.solvers.ppo", extra="rl"),
    "ppo_best_state_reset": SolverSpec("mecbench.solvers.ppo_best_state_reset", extra="rl"),
    "hierarchical_ppo": SolverSpec("mecbench.solvers.hierarchical_ppo", extra="rl"),
    "heuristic": SolverSpec("mecbench.solvers.heuristic"),
    "heuristic_unit_step": SolverSpec("mecbench.solvers.heuristic_unit_step"),
    "fast_heuristic": SolverSpec("mecbench.solvers.fast_heuristic"),
    "discrete_evolutionary": SolverSpec("mecbench.solvers.discrete_evolutionary"),
    "cma_es": SolverSpec("mecbench.solvers.cma_es", extra="evolutionary"),
}

SOLVER_MODULES = {name: spec.module for name, spec in REGISTRY.items()}
SOLVER_FILE_ALIASES = {
    "ILP_arc": "ilp_arc",
    "ILP_arc_95": "ilp_arc_gap_5pct",
    "ILP_path": "ilp_path",
    "MILP_arc": "milp_arc",
    "MILP_path": "milp_path",
    "milp_arc_exact": "milp_arc_full",
    "ILP_arc_heuristic_placement": "ilp_arc_heuristic_placement",
    "ILP_arc_heuristic_warmstart": "ilp_arc_heuristic_warmstart",
    "ILP_arc_rl_warmstart": "ilp_arc_rl_warmstart",
    "LP_arc": "lp_arc",
    "LP_path": "lp_path",
    "LP_path_strong_latency": "lp_path_latency_tightened",
    "rl_A2C": "a2c",
    "rl_PPO": "ppo",
    "rl_PPO_options": "ppo_best_state_reset",
    "base_heuristic": "heuristic",
    "rl_base": "a2c",
    "PPO": "ppo",
    "HRL": "hierarchical_ppo",
    "ilp_arc_95_latency": "ilp_arc_gap_5pct",
    "heuristicprecise": "heuristic_unit_step",
    "discrete": "discrete_evolutionary",
    "cma": "cma_es",
    "hrl": "hierarchical_ppo",
    "ppo_options": "ppo_best_state_reset",
}


def list_solvers() -> tuple[str, ...]:
    """List canonical names without importing optional dependencies."""
    return tuple(REGISTRY)


def canonical_name(name: str) -> str:
    if not isinstance(name, str):
        raise UnknownSolverError("solver must be a registered name; use list_solvers().")
    name = SOLVER_FILE_ALIASES.get(name, name)
    if name not in REGISTRY:
        raise UnknownSolverError(
            f"Unknown solver {name!r}. Available: {', '.join(list_solvers())}."
        )
    return name


def dependency_error(name: str, error: ModuleNotFoundError) -> MissingSolverDependencyError:
    extra = REGISTRY[name].extra
    install = f"mec-placement-benchmarks[{extra}]" if extra else "mec-placement-benchmarks"
    return MissingSolverDependencyError(
        f"Solver {name!r} needs {error.name!r}. Install with: python -m pip install '{install}'."
    )


def make_solver(
    name: str, problem: ProblemInstance, *, seed: int = 0, **options: Any
) -> SolverBackend:
    name = canonical_name(name)
    try:
        cls = cast(
            Callable[..., SolverBackend], importlib.import_module(REGISTRY[name].module).Solver
        )
        parameters = inspect.signature(cls).parameters
        if "seed" in parameters:
            options = {**options, "seed": seed}
        inspect.signature(cls).bind(problem, **options)
        return cls(problem, **options)
    except ModuleNotFoundError as error:
        raise dependency_error(name, error) from error
