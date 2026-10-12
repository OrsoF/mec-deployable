"""One PPO/Gymnasium environment for placement + path + whole-Mbps allocation.

Discrete actions:
 [0, S*N)                set placement(service, node)
 [S*N, S*N+D*K)          set route(demand, kth path)
 [S*N+D*K, ...+2*D)      change demand Mbps by -1 or +1

Infeasible actions are rejected, rather than projected into an unreported state.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from .constraints import assess
from .heuristic import feasible_seed
from .paths import PathCatalog
from .problem import Problem, Solution, validate_compatible


class NetworkPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, problems: Problem | list[Problem], *, episode_steps: int = 96,
                 paths: int = 3):
        super().__init__()
        self.problems = [problems] if isinstance(problems, Problem) else list(problems)
        validate_compatible(self.problems)
        if episode_steps <= 0 or paths <= 0:
            raise ValueError("episode_steps and paths must be positive")
        self.episode_steps = episode_steps
        self.n_paths = paths
        self.catalogues = [PathCatalog(p, paths) for p in self.problems]
        # Precompute feasible states ONCE for all training cases.
        self.seeds = [feasible_seed(p, c) for p, c in zip(self.problems, self.catalogues)]
        first = self.problems[0]
        s, n, d = len(first.services), len(first.nodes), len(first.demands)
        self.placement_actions = s * n
        self.routing_actions = d * paths
        self.action_space = Discrete(self.placement_actions + self.routing_actions + 2 * d)
        self.case_index = 0
        self.problem = first
        self.catalogue = self.catalogues[0]
        self.state = self.seeds[0].copy()
        self.steps = 0
        self.best_episode = self.state.copy()
        self.best_overall = self.state.copy() if len(self.problems) == 1 else None
        observation = self._observe()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=observation.shape,
                                     dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Gym reset(seed) seeds RNG; sampling via this RNG ensures repeatability.
        if len(self.problems) == 1:
            index = 0
        else:
            index = int(self.np_random.integers(len(self.problems)))
        self.case_index = index
        self.problem = self.problems[index]
        self.catalogue = self.catalogues[index]
        self.state = self.seeds[index].copy()
        self.best_episode = self.state.copy()
        self.steps = 0
        if len(self.problems) == 1 and self.best_overall is None:
            self.best_overall = self.state.copy()
        return self._observe(), {"instance_id": self.problem.id}

    def _observe(self) -> np.ndarray:
        p = self.problem
        evaluation = assess(p, self.state, self.catalogue)
        n, s, d = len(p.nodes), len(p.services), len(p.demands)
        max_flow = np.maximum(1, [dm.max_mbps for dm in p.demands])
        cpu = np.array([node.cpu for node in p.nodes])
        caps = p.arc_capacities
        limits = np.array([dm.max_latency_ms for dm in p.demands])
        features = [
            self.state.placement / max(n - 1, 1),
            self.state.routes / max(self.n_paths - 1, 1),
            self.state.quantities / max_flow,
            (cpu - evaluation.node_load) / np.maximum(cpu, 1.0),
            (caps - evaluation.edge_load) / caps,
            evaluation.demand_latency / limits,
            np.array([dm.min_mbps for dm in p.demands], dtype=float),
            np.array([dm.max_mbps for dm in p.demands], dtype=float),
            limits,
            np.array([dm.origin / max(n - 1, 1) for dm in p.demands]),
            np.array([dm.service / max(s - 1, 1) for dm in p.demands]),
            np.array([service.cpu for service in p.services]),
            cpu, caps, p.arc_base_latencies, p.arc_latency_slopes,
            np.array([self.steps / self.episode_steps]),
        ]
        return np.concatenate(features).astype(np.float32)

    def step(self, action: int):
        index = int(action)
        if not self.action_space.contains(index):
            raise ValueError(f"Unknown action {action}")
        trial = self.state.copy()
        n = len(self.problem.nodes)
        if index < self.placement_actions:
            service, node = divmod(index, n)
            trial.placement[service] = node
        elif index < self.placement_actions + self.routing_actions:
            demand, route = divmod(index - self.placement_actions, self.n_paths)
            trial.routes[demand] = route
        else:
            demand, sign = divmod(index - self.placement_actions - self.routing_actions, 2)
            trial.quantities[demand] += -1 if sign == 0 else 1

        previous = float(self.state.quantities.sum())
        feasible = assess(self.problem, trial, self.catalogue).feasible
        if feasible:
            self.state = trial
        current = float(self.state.quantities.sum())
        if current > float(self.best_episode.quantities.sum()):
            self.best_episode = self.state.copy()
        if len(self.problems) == 1 and (
            self.best_overall is None or current > float(self.best_overall.quantities.sum())
        ):
            self.best_overall = self.state.copy()
        self.steps += 1
        truncated = self.steps >= self.episode_steps
        # Throughput change is the training objective. Reject invalid/no-op edits.
        reward = current - previous if current != previous else (-0.05 if not feasible else -0.01)
        return self._observe(), float(reward), False, truncated, {
            "instance_id": self.problem.id,
            "feasible_action": bool(feasible),
            "total_served_mbps": current,
            "best_episode_mbps": float(self.best_episode.quantities.sum()),
        }
