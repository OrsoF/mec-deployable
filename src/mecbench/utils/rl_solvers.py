"""Shared reinforcement-learning training and evaluation wrapper."""

import time

from stable_baselines3 import A2C, PPO

from mecbench.domain.problem import ProblemInstance
from mecbench.environments.spp_env import SPPEnv
from mecbench.utils.solver_base import GenericSolver

MODEL_CLASSES = {"A2C": A2C, "PPO": PPO}


class RLSolver(GenericSolver):
    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int,
        train_step: int,
        evaluation_step: int,
        method: str,
        reuse_best_on_reset: bool = False,
        seed: int = 0,
    ):
        """Initialize the RL solver around one problem or compatible problems."""
        if isinstance(problem, list) and not problem:
            raise ValueError("RL Solver requires at least one problem instance.")
        problem_instance = problem[0] if isinstance(problem, list) else problem
        super().__init__(problem_instance)

        self.train_step = train_step
        self.evaluation_step = evaluation_step
        self.method = method.upper()
        self.seed = seed
        self.solver_name = f"rl_{self.method}"

        self.env = self._make_env(
            problem,
            max_step,
            reuse_best_on_reset,
        )
        self.model = self._make_model()

    def _make_env(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int,
        reuse_best_on_reset: bool,
    ) -> SPPEnv:
        """Create the Gym environment used by the SB3 model."""
        env = SPPEnv(
            problem=problem,
            max_step=max_step,
            reuse_best_on_reset=reuse_best_on_reset,
        )
        env.reset(seed=self.seed)
        return env

    def _make_model(self):
        """Create the configured Stable-Baselines3 model."""
        if self.method not in MODEL_CLASSES:
            known_methods = ", ".join(sorted(MODEL_CLASSES))
            raise ValueError(
                f"Unknown RL method '{self.method}'. Expected one of: {known_methods}."
            )

        model_kwargs = {
            "policy": "MlpPolicy",
            "env": self.env,
            "verbose": 0,
            "seed": self.seed,
        }
        if self.method == "PPO":
            model_kwargs["ent_coef"] = 0.01

        return MODEL_CLASSES[self.method](**model_kwargs)

    def run(self):
        """Train the model and record aggregate solver metrics."""
        start_time = time.time()
        self._set_solution(None)
        self.score = self.runtime = self.termination_condition = None
        self.build_time = 0.0

        self.model.learn(total_timesteps=self.train_step, progress_bar=False)

        # Evaluate on the solver's target, independently of training-instance resets.
        self.evaluation_env = SPPEnv(
            problem=self.problem_instance,
            max_step=self.env.max_steps,
            max_path_candidates=self.env.max_path_candidates,
            warm_start_heuristic=self.env.warm_start_heuristic,
            reuse_best_on_reset=self.env.reuse_best_on_reset,
        )
        if self.env.problem_instance is self.problem_instance:
            trained_state = self.env.best_state_record[0]
            if trained_state is not None:
                self.evaluation_env.state = trained_state.copy()
                self.evaluation_env._record_best_state()
        observation, _ = self.evaluation_env.reset(seed=self.seed)
        for _ in range(self.evaluation_step):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = self.evaluation_env.step(action)
            if terminated or truncated:
                observation, _ = self.evaluation_env.reset()

        best_state = self.evaluation_env.best_state_record[0]
        if best_state is not None:
            solution = self.evaluation_env.solution_from_state(best_state)
            self._set_solution(
                solution, path_resolver=self.evaluation_env.path_cache.nth_shortest_path
            )
            self.score = float(solution.quantities.sum())
        self.runtime = time.time() - start_time
        self.build_time = 0.0
        return self
