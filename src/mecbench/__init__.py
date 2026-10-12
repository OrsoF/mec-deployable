"""Mobile edge computing service placement benchmarks."""

__version__ = "0.1.0"

from mecbench.api import SolveResult, SolveStatus, make_star_instance, make_tree_instance, solve
from mecbench.domain.conversions import SolutionConversionError, to_arc_solution
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.problem_arrays import ProblemArrays
from mecbench.domain.solution import ArcSolution, CompactSolution, Solution
from mecbench.domain.validation import (
    SolutionValidationError,
    ValidationReport,
    Violation,
    validate,
)
from mecbench.errors import MissingSolverDependencyError, SolverExecutionError, UnknownSolverError
from mecbench.utils.solver_registry import list_solvers

__all__ = [
    "make_tree_instance",
    "make_star_instance",
    "solve",
    "validate",
    "list_solvers",
    "ProblemInstance",
    "ProblemArrays",
    "Solution",
    "ArcSolution",
    "CompactSolution",
    "to_arc_solution",
    "SolutionConversionError",
    "SolveResult",
    "SolveStatus",
    "ValidationReport",
    "Violation",
    "SolutionValidationError",
    "UnknownSolverError",
    "MissingSolverDependencyError",
    "SolverExecutionError",
]
