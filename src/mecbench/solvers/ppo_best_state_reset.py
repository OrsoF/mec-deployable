"""PPO solver that reuses the best state on reset."""

from mecbench.domain.problem import ProblemInstance
from mecbench.solvers.ppo import Solver as PPOSolver


class Solver(PPOSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int | None = None,
        train_step: int | None = None,
        evaluation_step: int = 100,
        method: str = "PPO",
        reuse_best_on_reset: bool = True,
        seed: int = 0,
    ):
        """Initialize the PPO solver variant that reuses the best reset state."""
        super().__init__(
            problem=problem,
            max_step=max_step,
            train_step=train_step,
            evaluation_step=evaluation_step,
            method=method,
            reuse_best_on_reset=reuse_best_on_reset,
            seed=seed,
        )
        self.solver_name = "ppo_best_state_reset"
