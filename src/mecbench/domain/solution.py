"""Canonical solution data with lazy historical import compatibility."""

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray, IntArray


def _readonly_floats(source: ArrayLike) -> FloatArray:
    values = np.asarray(source, dtype=np.float64)
    return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


def _readonly_indices(source: ArrayLike, name: str) -> IntArray:
    values = np.asarray(source)
    if values.dtype.kind not in "biuf" or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite integer indices.")
    if values.dtype.kind == "f" and (
        np.any(values != np.rint(values)) or np.any(values < -(2**63)) or np.any(values >= 2**63)
    ):
        raise ValueError(f"{name} must contain int64 indices without rounding.")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain int64 indices without overflow.")
    indices = np.asarray(values, dtype=np.int64)
    return np.frombuffer(indices.tobytes(), dtype=np.int64).reshape(indices.shape)


@dataclass(frozen=True, eq=False, init=False)
class ArcSolution:
    """Owned, read-only arc routing, service placement and demand quantities.

    Columns follow ``problem.network_graph.arc_list`` and integer node indices.
    Construction copies the arrays; feasibility is checked by ``validate``.
    """

    x: FloatArray
    y: FloatArray
    q: FloatArray

    def __init__(self, x: ArrayLike, y: ArrayLike, q: ArrayLike) -> None:
        for name, source in (("x", x), ("y", y), ("q", q)):
            object.__setattr__(self, name, _readonly_floats(source))

    def __iter__(self) -> Iterator[FloatArray]:
        return iter((self.x, self.y, self.q))

    def __getitem__(self, index: int) -> FloatArray:
        return (self.x, self.y, self.q)[index]

    def __reduce__(self) -> tuple[Any, tuple[FloatArray, FloatArray, FloatArray]]:
        return type(self), (self.x, self.y, self.q)


@dataclass(frozen=True, eq=False, init=False)
class CompactSolution:
    """Immutable node placements, path indices and continuous quantities.

    Routing indices are interpreted by an explicit path resolver at conversion.
    Shapes and feasibility depend on the problem and are checked by validation.
    """

    placement: IntArray
    routing: IntArray
    quantities: FloatArray

    def __init__(self, placement: ArrayLike, routing: ArrayLike, quantities: ArrayLike) -> None:
        object.__setattr__(self, "placement", _readonly_indices(placement, "placement"))
        object.__setattr__(self, "routing", _readonly_indices(routing, "routing"))
        object.__setattr__(self, "quantities", _readonly_floats(quantities))

    def __iter__(self) -> Iterator[IntArray | FloatArray]:
        return iter((self.placement, self.routing, self.quantities))

    def __getitem__(self, index: int) -> IntArray | FloatArray:
        return (self.placement, self.routing, self.quantities)[index]

    def __reduce__(self) -> tuple[Any, tuple[IntArray, IntArray, FloatArray]]:
        return type(self), (self.placement, self.routing, self.quantities)


Solution = ArcSolution


_LEGACY_EXPORTS: dict[str, str] = {
    "to_int_array": "mecbench.domain.conversions",
    "split_state": "mecbench.domain.conversions",
    "make_state": "mecbench.domain.conversions",
    "validate_compact_shapes": "mecbench.domain.conversions",
    "compact_to_arc": "mecbench.domain.conversions",
    "basic_solution": "mecbench.utils.heuristic_solvers",
    "basic_state": "mecbench.utils.heuristic_solvers",
    "clip_solution": "mecbench.utils.heuristic_solvers",
    "clip_state": "mecbench.utils.heuristic_solvers",
    "greedy_quantity_lift": "mecbench.utils.heuristic_solvers",
    "repair_solution": "mecbench.utils.heuristic_solvers",
    "SolvingInstance": "mecbench.utils.solver_workspace",
    "residual_arrays": "mecbench.domain.loads",
    "apply_residuals_to_graph": "mecbench.utils.solver_workspace",
    "validate_arc_solution": "mecbench.domain.validation",
}

__all__ = ["ArcSolution", "CompactSolution", "Solution", *_LEGACY_EXPORTS]


def __getattr__(name: str) -> Any:
    module = _LEGACY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)
