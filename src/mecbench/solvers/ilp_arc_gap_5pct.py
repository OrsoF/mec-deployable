"""Arc ILP with a default relative optimality gap of five percent."""

from mecbench.domain.problem import ProblemInstance
from mecbench.solvers.ilp_arc import Solver as ILPArcSolver
from mecbench.utils.solver_base import GenericSolver


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        super().__init__(problem_instance)
        self.solver_name = "ilp_arc_gap_5pct"

    def run(self, options: dict | None = None):
        solver = ILPArcSolver(self.problem_instance)
        solver.run(options={"MIPGap": 0.05, **(options or {})})
        self._set_output_variables(solver.x_var, solver.y_var, solver.q_var)
        self.termination_condition = solver.termination_condition
        self.score = solver.score
        self.runtime = solver.runtime
        self.build_time = solver.build_time
        return self
