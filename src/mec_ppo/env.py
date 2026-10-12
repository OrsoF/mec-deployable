"""Feasible-action PPO environment with common observation space across instances."""
import numpy as np
from gymnasium.spaces import Box

from mecbench.environments.spp_env import SPPEnv
from mecbench.environments.observation import make_vector_observation
from mecbench.environments.state_codec import to_integer_state
from mecbench.domain.conversions import make_state
from mecbench.solvers.fast_heuristic import Solver as FastHeuristic


class NetworkPPOEnv(SPPEnv):
    def __init__(self, *args, **kwargs):
        # Cache the per-instance feasible seed: the old heuristic_unit_step can
        # return no candidate when the default all-cloud placement is infeasible.
        self._seeds: dict[int, np.ndarray] = {}
        super().__init__(*args, **kwargs)

    def _make_initial_state(self):
        if not self.warm_start_heuristic:
            return self._make_basic_state()
        key = id(self.problem_instance)
        if key not in self._seeds:
            solver = FastHeuristic(
                self.problem_instance,
                limit_shortest_path=self.max_path_candidates,
                candidate_nodes=min(8, self.data.n_nodes),
                max_iter=30,
                patience=6,
                seed=0,
            ).run()
            if solver.placement is None:
                raise ValueError(
                    f"No feasible warm-start for instance {self.problem_instance.problem_name}. "
                    "Check capacity, compute and latency constraints."
                )
            # Start PPO at min flow so a throughput improvement can be learned.
            state = make_state(solver.placement, solver.routing, self.data.min_flows)
            if not self._is_feasible_state(state):
                raise ValueError("Fast heuristic returned an infeasible placement/routing seed")
            self._seeds[key] = state
        return self._seeds[key].copy()

    def _configure_spaces(self) -> None:
        super()._configure_spaces()
        # Add demand origins and link delay coefficients (absent from base observation).
        size = int(self.observation_space.shape[0] + self.data.n_demands + 2 * self.data.n_arcs)
        # Fixed bounds across a scenario family; VecNormalize learns actual scale.
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32
        )

    def _observe(self, state) -> np.ndarray:
        base = make_vector_observation(self, state)
        return np.concatenate(
            (base, self.data.origins, self.data.latency_slopes, self.data.latency_offsets)
        ).astype(np.float32)

    def step(self, action):
        """Reward feasible changes in throughput; reject invalid actions.

        Every accepted action preserves physical feasibility, including quantity actions.
        Placement/routing can give a zero immediate reward while enabling later gains.
        """
        state = to_integer_state(self.state)
        old_score = float(self._score_state(state))
        operation, service, node, demand, quantity_action, path_index = map(int, action)
        if operation == 0:
            candidate, feasible = self._update_placement(state, service, node)
        elif operation == 1:
            candidate, feasible = self._update_route(state, demand, path_index)
        else:
            candidate, feasible = self._update_quantity(state, demand, quantity_action)
        if feasible:
            self.state = to_integer_state(candidate)
        new_score = float(self._score_state(self.state))
        self._record_best_state(new_score, feasible=feasible)
        self.served_quantity_history.append(new_score)
        self.steps_done += 1
        truncated = self.steps_done >= self.max_steps
        max_total = max(1.0, float(self.data.max_flows.sum()))
        reward = (new_score - old_score) / max_total
        if not feasible:
            reward -= 0.02
        reward -= 0.0005
        return self._observe(self.state), float(reward), False, truncated, {
            "feasible_action": bool(feasible), "total_mbps": new_score,
        }
