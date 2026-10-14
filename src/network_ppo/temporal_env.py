"""One PPO action = one complete decision for one physical time period.

Action = MultiDiscrete([node]*services + [path]*demands). A single valid
allocation is generated from these choices; its cost is charged ONCE. Invalid
proposals are rejected, with an explicit fallback and penalty. The next state
is a new demand/capacity observation, not another edit of the same instance.
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete
from .dynamics import TemporalScenario
from .paths import PathCatalog
from .problem import validate_compatible
from .temporal_costs import stage_score
from .temporal_optimization import allocate, initial_allocation, TemporalPlanner


class TemporalNetworkEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenarios: TemporalScenario | list[TemporalScenario], *, paths: int = 3,
                 forecast_steps: int = 4, invalid_penalty: float = 2.0):
        super().__init__()
        self.scenarios = ([scenarios] if isinstance(scenarios, TemporalScenario) else list(scenarios))
        if not self.scenarios or paths < 1 or forecast_steps < 1 or invalid_penalty < 0:
            raise ValueError("Invalid temporal environment configuration")
        validate_compatible([s.base for s in self.scenarios])
        self.paths = paths
        self.forecast_steps = forecast_steps
        self.invalid_penalty = invalid_penalty
        base = self.scenarios[0].base
        self.action_space = MultiDiscrete([len(base.nodes)] * len(base.services)
                                           + [paths] * len(base.demands))
        self.scenario = self.scenarios[0]
        self.t = 0
        self.previous = initial_allocation(self.scenario, paths=paths)
        self.last_invalid = False
        obs = self._observation()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = 0 if len(self.scenarios) == 1 else int(self.np_random.integers(len(self.scenarios)))
        self.scenario = self.scenarios[idx]
        self.t = 0
        self.previous = initial_allocation(self.scenario, paths=self.paths)
        self.last_invalid = False
        return self._observation(), {"scenario": self.scenario.id}

    def _observation(self):
        sc = self.scenario
        t = self.t
        d, s, n = len(sc.base.demands), len(sc.base.services), len(sc.base.nodes)
        max_d = np.maximum(1, np.array([dm.max_mbps for dm in sc.base.demands]))
        p = sc.problem_at(t)
        # Previous solution may be infeasible under changed capacities/demand bounds.
        # Do not hide it; the features describe the previous physical action.
        base = [self.previous.placement / max(1, n - 1),
                self.previous.routes / max(1, self.paths - 1),
                self.previous.quantities / max_d,
                sc.realized_mbps[t] / max_d,
                sc.capacity_scale[t],
                np.array([(sc.initial_time + t) / max(1, (sc.full_horizon or sc.horizon) - 1), float(self.last_invalid)]),
                np.array([sc.migration_cost, sc.route_change_cost, sc.active_node_cost])]
        for k in range(1, self.forecast_steps + 1):
            target = min(t + k, sc.horizon - 1)
            base.append(sc.forecast_mbps[target] / max_d)
        return np.concatenate(base).astype(np.float32)

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Invalid temporal action")
        sc = self.scenario
        p = sc.problem_at(self.t)
        catalog = PathCatalog(p, self.paths)
        s = len(p.services)
        placement = np.asarray(action[:s], dtype=int)
        routes = np.asarray(action[s:], dtype=int)
        candidate = allocate(p, catalog, placement, routes)
        invalid = candidate is None
        if invalid:
            candidate = allocate(p, catalog, self.previous.placement, self.previous.routes)
            if candidate is None:
                candidate = TemporalPlanner(strategy="myopic", paths=self.paths).decide(
                    sc, self.t, self.previous).solution
        metrics = stage_score(sc, p, self.previous, candidate, catalog)
        reward = metrics.reward - (self.invalid_penalty if invalid else 0.0)
        self.previous = candidate
        self.last_invalid = invalid
        self.t += 1
        terminated = self.t >= sc.horizon
        info = {"scenario": sc.id, "period": self.t - 1, "reward_without_invalid_penalty": metrics.reward,
                "served_mbps": metrics.served_mbps, "migrations": metrics.migrations,
                "route_changes": metrics.route_changes, "active_nodes": metrics.active_nodes,
                "valid_proposal": not invalid, "fallback": bool(invalid)}
        return (np.zeros(self.observation_space.shape, dtype=np.float32) if terminated
                else self._observation()), float(reward), terminated, False, info
