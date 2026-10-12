"""Heuristic variant with a default quantity increment of one."""

from mecbench.domain.problem import ProblemInstance
from mecbench.solvers.heuristic import Solver as HeuristicSolver


class Solver(HeuristicSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        limit_shortest_path: int = 20,
        step_no_progression: int = 20,
        quantity_increment: int = 1,
        seed: int = 0,
        max_iter: int | None = None,
        patience: int | None = None,
    ):
        super().__init__(
            problem_instance=problem_instance,
            limit_shortest_path=limit_shortest_path,
            step_no_progression=step_no_progression,
            quantity_increment=quantity_increment,
            seed=seed,
            max_iter=max_iter,
            patience=patience,
        )
        self.solver_name = "heuristic_unit_step"
