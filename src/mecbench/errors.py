"""Actionable errors raised by the public solver interface."""


class UnknownSolverError(ValueError):
    """The requested solver is not registered."""


class MissingSolverDependencyError(ImportError):
    """An optional solver dependency is missing."""


class SolverExecutionError(RuntimeError):
    """A backend could not execute (for example, an unavailable optimizer/license)."""
