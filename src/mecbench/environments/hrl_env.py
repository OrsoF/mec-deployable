import numpy as np
from gymnasium.spaces import MultiDiscrete

from mecbench.domain.conversions import make_state, split_state
from mecbench.environments.hrl_reward import (
    normalized_delta,
    placement_routing_score,
    served_quantity,
)
from mecbench.environments.options import move_service, reroute_demand, shift_quantity
from mecbench.environments.spp_env import SPPEnv


class PlacementRoutingEnv(SPPEnv):
    def __init__(
        self,
        *args,
        quantity_increment: int = 1,
        score_rollouts: int | None = None,
        **kwargs,
    ):
        self.quantity_increment = quantity_increment
        self.score_rollouts = score_rollouts
        super().__init__(*args, **kwargs)

    def _configure_spaces(self) -> None:
        super()._configure_spaces()
        self.action_space = MultiDiscrete(
            [
                2,
                self.data.n_services,
                self.graph.number_of_nodes(),
                self.data.n_demands,
                self.max_path_candidates,
            ]
        )

    def _make_initial_state(self) -> np.ndarray:
        p, r, _ = split_state(self.problem_instance, super()._make_initial_state())
        return make_state(p, r, self.data.min_flows)

    def _score_pr_state(self, s) -> float:
        steps = self.score_rollouts
        if steps is None:
            steps = int(
                self.data.n_demands
                * max(
                    1,
                    max(self.data.max_flows),
                )
            )
        return placement_routing_score(
            self.problem_instance,
            self.solution_checker,
            self.np_random,
            s,
            steps,
            self.quantity_increment,
        )

    def step(self, action: tuple) -> tuple:
        op, k, v, d, path_index = action
        old_state = self.state.copy()
        old_score = self._score_pr_state(old_state)

        if op == 0:
            candidate = move_service(self.problem_instance, old_state, k, v)
        else:
            candidate = reroute_demand(self.problem_instance, old_state, d, path_index)
        candidate = self._canonical_state(candidate)

        feasible = self._is_feasible_state(candidate)
        self.state = candidate if feasible else old_state
        new_score = self._score_pr_state(self.state)
        reward = normalized_delta(new_score, old_score) if feasible else -0.05

        self._record_best_state(new_score, feasible=feasible)

        self.served_quantity_history.append(new_score)
        self.steps_done += 1
        truncated = self.steps_done >= self.max_steps
        return self._observe(self.state), reward, False, truncated, {"feasible": feasible}


class FlowSetupEnv(SPPEnv):
    def __init__(
        self,
        *args,
        fixed_state: np.ndarray | None = None,
        quantity_increment: int = 1,
        **kwargs,
    ):
        self.fixed_state = fixed_state
        self.quantity_increment = quantity_increment
        super().__init__(*args, **kwargs)

    def _configure_spaces(self) -> None:
        super()._configure_spaces()
        self.action_space = MultiDiscrete(
            [
                self.data.n_demands,
                3,
            ]
        )

    def _make_initial_state(self) -> np.ndarray:
        if self.fixed_state is None:
            p, r, _ = split_state(self.problem_instance, super()._make_initial_state())
        else:
            p, r, _ = split_state(self.problem_instance, self.fixed_state)
        return make_state(p, r, self.data.min_flows)

    def step(self, action: tuple) -> tuple:
        d, quantity_action = action
        dq = (int(quantity_action) - 1) * self.quantity_increment
        old_state = self.state.copy()
        old_score = served_quantity(self.problem_instance, old_state)
        candidate = self._canonical_state(shift_quantity(self.problem_instance, old_state, d, dq))

        feasible = self._is_feasible_state(candidate)
        self.state = candidate if feasible else old_state
        new_score = served_quantity(self.problem_instance, self.state)
        reward = normalized_delta(new_score, old_score) if feasible else -0.05

        self._record_best_state(new_score, feasible=feasible)

        self.served_quantity_history.append(new_score)
        self.steps_done += 1
        truncated = self.steps_done >= self.max_steps
        return self._observe(self.state), reward, False, truncated, {"feasible": feasible}
