"""PPO reinforcement-learning solver."""

from mecbench.domain.problem import ProblemInstance
from mecbench.utils.rl_solvers import RLSolver


class Solver(RLSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int | None = None,
        train_step: int | None = None,
        evaluation_step: int = 100,
        method: str = "PPO",
        reuse_best_on_reset: bool = False,
        seed: int = 0,
    ):
        """Initialize the default PPO RL solver."""
        problem_instance = problem[0] if isinstance(problem, list) else problem
        max_step = 1000 + problem_instance.data.n_services * 10 if max_step is None else max_step
        train_step = (
            10000 + problem_instance.data.n_services * 20 if train_step is None else train_step
        )
        super().__init__(
            problem=problem,
            max_step=max_step,
            train_step=train_step,
            evaluation_step=evaluation_step,
            method=method,
            reuse_best_on_reset=reuse_best_on_reset,
            seed=seed,
        )
