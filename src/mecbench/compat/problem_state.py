"""Lazy mutable solution state for the historical ProblemInstance interface."""

from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import ArcArrays, FloatArray, ProblemData
from mecbench.domain.validation import validate_arc_solution

if TYPE_CHECKING:
    from mecbench.domain.problem import ProblemInstance


class LegacyProblemState:
    """Compatibility methods; modern solve() owns its state in a workspace."""

    _legacy_state: ArcArrays | None = None

    def _legacy_arrays(self) -> ArcArrays:
        if self._legacy_state is None:
            self._initialize_baseline()
        assert self._legacy_state is not None
        return self._legacy_state

    @property
    def x_variable(self) -> FloatArray:
        return self._legacy_arrays()[0]

    @x_variable.setter
    def x_variable(self, values: ArrayLike) -> None:
        _, y, q = self._legacy_arrays()
        self._set_arc_solution(values, y, q)

    @property
    def y_variable(self) -> FloatArray:
        return self._legacy_arrays()[1]

    @y_variable.setter
    def y_variable(self, values: ArrayLike) -> None:
        x, _, q = self._legacy_arrays()
        self._set_arc_solution(x, values, q)

    @property
    def q_variable(self) -> FloatArray:
        return self._legacy_arrays()[2]

    @q_variable.setter
    def q_variable(self, values: ArrayLike) -> None:
        x, y, _ = self._legacy_arrays()
        self._set_arc_solution(x, y, values)

    def _set_arc_solution(
        self, x_variable: ArrayLike, y_variable: ArrayLike, q_variable: ArrayLike
    ) -> None:
        problem = cast("ProblemInstance", self)
        x, y, q = (
            np.array(values, dtype=float, copy=True)
            for values in (x_variable, y_variable, q_variable)
        )
        shapes = (
            (problem.total_number_of_demands, problem.network_graph.nx_graph.number_of_edges()),
            (problem.number_of_service, problem.network_graph.nx_graph.number_of_nodes()),
            (problem.total_number_of_demands,),
        )
        for name, values, shape in zip(("x", "y", "q"), (x, y, q), shapes, strict=False):
            if values.shape != shape:
                raise ValueError(f"{name} shape must be {shape}, got {values.shape}.")
        self._legacy_state = x, y, q

    def check_current_solution(self) -> None:
        validate_arc_solution(cast(ProblemData, self), *self._legacy_arrays())

    def validate_instance(self) -> None:
        """Validate the historical baseline; this is not an infeasibility proof."""
        self._initialize_baseline()
        self.check_current_solution()

    def _initialize_baseline(self) -> None:
        from mecbench.utils.heuristic_solvers import baseline_solution

        candidate = baseline_solution(cast(ProblemData, self))
        self._set_arc_solution(candidate.x, candidate.y, candidate.q)
