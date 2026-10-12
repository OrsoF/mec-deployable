from collections.abc import Callable

import gymnasium as gym
import numpy as np

from mecbench.domain.conversions import compact_from_state
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.solution import CompactSolution
from mecbench.environments.observation import (
    make_action_space,
    make_vector_observation,
    make_vector_observation_space,
)
from mecbench.environments.reward import served_quantity_score
from mecbench.environments.state_codec import (
    clip_state_to_bounds,
    is_feasible_state,
    make_basic_state,
    make_state,
    state_to_variables,
    state_to_vectors,
    to_integer_state,
)
from mecbench.solvers.heuristic_unit_step import Solver
from mecbench.utils.solver_workspace import SolverWorkspace


class SPPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        problem: ProblemInstance | list[ProblemInstance],
        max_step: int = 1000,
        max_path_candidates: int = 3,
        warm_start_heuristic: bool = True,
        observation_type: str = "vector",
        reuse_best_on_reset: bool = False,
        render_mode: str | None = None,
    ):
        """Initialize the environment for one problem or a compatible problem set."""
        self._validate_observation_type(observation_type)
        self._validate_render_mode(render_mode)
        self.render_mode = render_mode
        self.max_path_candidates = max_path_candidates
        self.max_steps = max_step
        self.problem_instances = self._make_problem_instances(problem)
        self._validate_problem_instances_compatibility(self.problem_instances)
        self.warm_start_heuristic = warm_start_heuristic
        self.reuse_best_on_reset = reuse_best_on_reset
        self.base_problem_instance = self.problem_instances[0]
        self.served_quantity_history = [0]
        self.steps_done = 0
        self.episode_start_score_index = 0
        self._load_problem_instance(self.base_problem_instance)

    @staticmethod
    def _make_problem_instances(
        problem: ProblemInstance | list[ProblemInstance],
    ) -> list[ProblemInstance]:
        """Normalize the problem input to a non-empty instance list."""
        if isinstance(problem, list):
            if not problem:
                raise ValueError("SPPEnv requires at least one problem instance.")
            return problem
        return [problem]

    @staticmethod
    def _validate_observation_type(observation_type: str) -> None:
        """Reject unsupported observation modes while preserving the public option."""
        if observation_type != "vector":
            raise ValueError("Only vector observations are implemented.")

    @classmethod
    def _validate_render_mode(cls: type["SPPEnv"], render_mode: str | None) -> None:
        """Reject unsupported Gym render modes."""
        if render_mode is not None and render_mode not in cls.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render mode {render_mode!r}. "
                f"Expected one of {cls.metadata['render_modes']}."
            )

    @staticmethod
    def _problem_compatibility_signature(problem_instance: ProblemInstance) -> tuple:
        """Return the dimensions that define Gym-space compatibility."""
        graph = problem_instance.network_graph.nx_graph
        return (
            problem_instance.number_of_service,
            problem_instance.total_number_of_demands,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    @classmethod
    def _validate_problem_instances_compatibility(
        cls: type["SPPEnv"], problem_instances: list[ProblemInstance]
    ) -> None:
        """Ensure every training instance preserves the Gym action/observation shape."""
        expected_signature = cls._problem_compatibility_signature(problem_instances[0])

        for instance_index, problem_instance in enumerate(problem_instances[1:], start=1):
            signature = cls._problem_compatibility_signature(problem_instance)
            if signature == expected_signature:
                continue

            raise ValueError(
                "All SPPEnv problem instances must share the same Gym dimensions. "
                f"Instance 0 has signature {expected_signature}, but instance "
                f"{instance_index} has signature {signature}. Signatures are "
                "(number_of_service, total_number_of_demands, number_of_nodes, "
                "number_of_edges)."
            )

    def _load_problem_instance(self, problem_instance: ProblemInstance) -> None:
        """Load problem-specific objects without rebuilding environment config."""
        self.problem_instance = problem_instance
        self.data = problem_instance.data
        self.network_graph = self.problem_instance.network_graph
        self.solution_checker = SolverWorkspace(self.problem_instance)
        self.graph = self.solution_checker.graph
        self.path_cache = self.solution_checker.path_cache
        self.saved_shortest_path = self.path_cache._paths
        self.initial_solution = None
        self._configure_spaces()
        self.state = to_integer_state(self._make_initial_state())
        self.best_state_record: tuple[np.ndarray | None, float] = (None, -np.inf)
        self._record_best_state()

    def _record_best_state(
        self, score: float | None = None, *, feasible: bool | None = None
    ) -> None:
        """Keep only feasible incumbents; an empty record denotes no solution."""
        if feasible is False:
            return
        if score is None:
            score = self._score_state(self.state)
        if score > self.best_state_record[1] and (
            feasible is True or self._is_feasible_state(self.state)
        ):
            self.best_state_record = (self.state.copy(), score)

    def _make_initial_state(self) -> np.ndarray:
        """Build the initial placement, routing, and quantity state."""
        if self.warm_start_heuristic and self.initial_solution is not None:
            return self.initial_solution.copy()

        basic_state = self._make_basic_state()

        if self.warm_start_heuristic:
            heuristic_solution = Solver(self.problem_instance, 3, 3)
            heuristic_solution.run()

            if heuristic_solution.solution is not None:
                self.initial_solution = make_state(*heuristic_solution.solution)
                if self._is_feasible_state(self.initial_solution):
                    self.served_quantity_history.append(self._score_state(self.initial_solution))
                    return self.initial_solution.copy()
                self.initial_solution = None

        return basic_state

    def _make_basic_state(self) -> np.ndarray:
        """Create the default basic state for the loaded problem."""
        return make_basic_state(self.problem_instance, self.graph)

    def reset(
        self,
        seed: int | None = None,
        initial_instance: bool = False,
        warm_start_heuristic: bool | None = None,
        observation_type: str | None = None,
        reuse_best_on_reset: bool | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment, optionally overriding reset-time configuration."""
        super().reset(seed=seed)

        warm_start_heuristic = (
            self.warm_start_heuristic if warm_start_heuristic is None else warm_start_heuristic
        )
        if observation_type is not None:
            self._validate_observation_type(observation_type)
        if reuse_best_on_reset is None:
            reuse_best_on_reset = self.reuse_best_on_reset
        self.warm_start_heuristic = warm_start_heuristic
        self.reuse_best_on_reset = reuse_best_on_reset

        next_problem_instance = self.problem_instance

        if len(self.problem_instances) > 1:
            instance_index = int(self.np_random.integers(len(self.problem_instances)))
            next_problem_instance = self.problem_instances[instance_index]

        if initial_instance:
            next_problem_instance = self.base_problem_instance

        if next_problem_instance is not self.problem_instance:
            self._load_problem_instance(next_problem_instance)
        else:
            self._configure_spaces()

        if not self.reuse_best_on_reset or self.best_state_record[0] is None:
            self.state = to_integer_state(self._make_initial_state())
        else:
            self.state = to_integer_state(self.best_state_record[0])

        self._record_best_state()
        self.steps_done = 0
        self.episode_start_score_index = len(self.served_quantity_history)
        return self._observe(self.state), {}

    def _try_state_update(
        self,
        state: np.ndarray | list,
        update_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], None],
    ) -> tuple:
        """Apply one state mutation and keep the old state when infeasible."""
        old_state = to_integer_state(state)
        new_state = old_state.copy()
        update_fn(*self._state_to_vectors(new_state))
        new_state = to_integer_state(new_state)
        is_feasible = self._is_feasible_state(new_state)
        if is_feasible:
            return new_state, is_feasible
        return old_state, is_feasible

    def _update_placement(
        self, state: np.ndarray | list, selected_service: int, chosen_node: int
    ) -> tuple:
        """Try moving one service, keeping the old state when infeasible."""

        def move_service(placement: np.ndarray, _: np.ndarray, __: np.ndarray) -> None:
            """Move the selected service placement inside a candidate state."""
            placement[selected_service] = chosen_node

        return self._try_state_update(state, move_service)

    def _update_route(
        self, state: np.ndarray | list, selected_demand: int, chosen_path_index: int
    ) -> tuple:
        """Try rerouting one demand through the selected path index."""

        def reroute_demand(_: np.ndarray, routing: np.ndarray, __: np.ndarray) -> None:
            """Update the selected demand route inside a candidate state."""
            routing[selected_demand] = chosen_path_index

        return self._try_state_update(state, reroute_demand)

    def _update_quantity(
        self, state: np.ndarray | list, selected_demand: int, quantity_action: int
    ) -> tuple:
        """For one demand, apply -1, 0, or +1 according to quantity_action."""
        quantity_delta = quantity_action - 1

        def resize_demand(_: np.ndarray, __: np.ndarray, quantities: np.ndarray) -> None:
            """Update the selected demand quantity inside a candidate state."""
            quantities[selected_demand] += quantity_delta

        return self._try_state_update(state, resize_demand)

    def render(self, mode: str | None = None):
        """Render the current solution state."""
        from mecbench.benchmarks.reporting import plot_instance_solution

        render_mode = mode or self.render_mode or "human"
        self._validate_render_mode(render_mode)
        x, y, q = self._state_to_variables(self.state)
        fig, _ = plot_instance_solution(
            self.problem_instance,
            x=x,
            y=y,
            q=q,
            show=render_mode == "human",
        )

        if render_mode == "rgb_array":
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            import matplotlib.pyplot as plt

            plt.close(fig)
            return image

        return None

    def step(self, action: tuple) -> tuple:
        """Apply one environment action and return the Gym step tuple."""
        terminated = False
        info = {}

        state = to_integer_state(self.state)

        # Action layout: [operation, service, node, demand, quantity delta, path index].
        (
            operation,
            selected_service,
            chosen_node_for_this_service,
            selected_demand,
            quantity_action,
            chosen_path_index,
        ) = action

        if operation == 0:
            state, done = self._update_placement(
                state, selected_service, chosen_node_for_this_service
            )
            reward = -1 if done else -10
        elif operation == 1:
            state, done = self._update_route(state, selected_demand, chosen_path_index)
            reward = -1 if done else -2
        else:
            state, _ = self._update_quantity(state, selected_demand, quantity_action)
            current_served_quantity = sum(self._state_to_vectors(state)[2])
            best_episode_quantity = self._best_episode_score()
            reward = max(0, current_served_quantity - best_episode_quantity)

        self.state = to_integer_state(state)
        observation = self._observe(self.state)
        self.served_quantity_history.append(sum(self._state_to_vectors(self.state)[2]))

        current_score = self._score_state(self.state)
        self._record_best_state(current_score)

        self.steps_done += 1
        truncated = self.steps_done >= self.max_steps
        return observation, reward / 100, terminated, truncated, info

    def _is_feasible_state(self, state: np.ndarray | list | None = None) -> bool:
        """Test if a state is feasible."""
        if state is None:
            state = self.state
        return is_feasible_state(
            self.problem_instance,
            self.solution_checker,
            self.path_cache,
            state,
        )

    def _configure_spaces(self) -> None:
        """Build the Gym action and observation spaces."""
        # State layout: [service placements | path indices | demand quantities].
        self.action_space = make_action_space(
            self.problem_instance, self.graph, self.max_path_candidates
        )
        self.observation_space = make_vector_observation_space(
            self.problem_instance, self.graph, self.max_path_candidates
        )

    def clip_state(self) -> np.ndarray:
        """Clip the current flat state to action-compatible integer bounds."""
        self.state = self._canonical_state(self.state)
        return self.state

    def _canonical_state(self, state: np.ndarray | list) -> np.ndarray:
        """Return an integer state rounded and clipped to valid bounds."""
        return clip_state_to_bounds(
            self.problem_instance,
            self.graph,
            self.max_path_candidates,
            state,
        )

    def _best_episode_score(self) -> float:
        """Return the best served quantity seen since the episode started."""
        episode_scores = self.served_quantity_history[self.episode_start_score_index :]
        return max(episode_scores) if episode_scores else 0

    def _state_to_variables(self, state: list) -> tuple:
        """Convert state to arc, placement, and quantity variables."""
        return state_to_variables(
            self.problem_instance,
            self.graph,
            self.network_graph,
            self.path_cache,
            state,
        )

    def solution_from_state(self, state: np.ndarray | list) -> CompactSolution:
        return compact_from_state(self.data, state)

    def _state_to_vectors(self, state: list) -> tuple:
        """Split state into service placement, path indices, and quantities."""
        return state_to_vectors(self.problem_instance, state)

    def _observe(self, state: np.ndarray | list) -> np.ndarray:
        """Transform a state into the flat observation vector."""
        # Observation features depend on mutable graph capacities and resources.
        return make_vector_observation(self, state)

    def _score_state(self, state: np.ndarray | list) -> float:
        """Compute the scalar score for a state."""
        return served_quantity_score(self.problem_instance, state)
