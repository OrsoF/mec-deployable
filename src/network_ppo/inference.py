"""Validated inference with explicit heuristic and optional PPO modes.

Importing this module does not import Gymnasium or PyTorch. PolicyRuntime
loads one trusted checkpoint once; metadata is checked before predictions.
Checkpoint files use pickle internally (SB3, VecNormalize): ONLY load trusted
local models, never an untrusted artifact supplied via HTTP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .constraints import assess
from .heuristic import solve_baseline
from .paths import PathCatalog
from .problem import Problem, Solution, validate_compatible
from .solution_io import decision_json


@dataclass
class Candidate:
    label: str
    solution: Solution
    steps: int = 0
    elapsed_ms: float = 0.0


class PolicyRuntime:
    """Cached checkpoint. One runtime handles fixed-topology requests sequentially."""

    def __init__(self, checkpoint: str | Path):
        from stable_baselines3 import PPO

        self.checkpoint = Path(checkpoint)
        metadata_path = self.checkpoint / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") not in (1, 2):
            raise ValueError("Unsupported checkpoint metadata version")
        for key in ("signature", "episode_steps", "paths"):
            if key not in self.metadata:
                raise ValueError(f"Checkpoint metadata missing {key}")
        self.model = PPO.load(str(self.checkpoint / "model.zip"), device="cpu")
        if not (self.checkpoint / "vecnormalize.pkl").exists():
            raise FileNotFoundError("VecNormalize statistics are required")

    def predict(
        self,
        problem: Problem,
        *,
        episodes: int = 1,
        stochastic: bool = False,
        seed: int = 123,
    ) -> Candidate:
        """Return the best feasible state visited, including the initial seed."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from .environment import NetworkPPOEnv

        if episodes < 1:
            raise ValueError("episodes must be positive")
        validate_compatible([problem], reference=self.metadata["signature"])
        modes = {1: "atomic", 2: self.metadata.get("action_mode", "atomic")}
        mode = modes[int(self.metadata["schema_version"])]
        env = NetworkPPOEnv(
            problem,
            episode_steps=int(self.metadata["episode_steps"]),
            paths=int(self.metadata["paths"]),
            action_mode=mode,
        )
        vec = DummyVecEnv([lambda: env])
        normalized = VecNormalize.load(str(self.checkpoint / "vecnormalize.pkl"), vec)
        normalized.training = False
        normalized.norm_reward = False
        self.model.set_random_seed(seed)
        normalized.seed(seed)
        begin = perf_counter()
        count = 0
        best = None
        try:
            for _ in range(episodes):
                obs = normalized.reset()
                # Track across episodes, not just terminal states.
                if best is None or env.state.quantities.sum() > best.quantities.sum():
                    best = env.state.copy()
                for _ in range(env.episode_steps):
                    action, _ = self.model.predict(obs, deterministic=not stochastic)
                    obs, _, dones, _ = normalized.step(action)
                    count += 1
                    if (
                        best is None
                        or env.state.quantities.sum() > best.quantities.sum()
                    ):
                        best = env.state.copy()
                    if dones[0]:
                        break
        finally:
            normalized.close()
        # DummyVecEnv automatically resets on truncation; the environment keeps
        # a separate best_overall, so include the last truncated transition.
        if env.best_overall is not None and (
            best is None or env.best_overall.quantities.sum() > best.quantities.sum()
        ):
            best = env.best_overall.copy()
        assert best is not None
        if not assess(
            problem, best, PathCatalog(problem, int(self.metadata["paths"]))
        ).feasible:
            raise RuntimeError("PPO returned infeasible solution")
        return Candidate("ppo", best, count, (perf_counter() - begin) * 1000)


class DecisionEngine:
    """Decision API: `auto` returns best feasible baseline/PPO (if available).

    `ppo` does not silently switch to heuristics. `auto` records failures and
    selects the best feasible candidate, with an explicit selected_method.
    """

    def __init__(
        self,
        *,
        strategy: str = "lookahead",
        checkpoint: str | Path | None = None,
        paths: int = 3,
        episodes: int = 1,
        stochastic: bool = False,
        seed: int = 123,
    ):
        if strategy not in ("greedy", "lookahead", "multistart", "ppo", "auto"):
            raise ValueError("Unknown strategy")
        if episodes < 1 or paths < 1:
            raise ValueError("episodes and paths must be positive")
        if strategy == "ppo" and checkpoint is None:
            raise ValueError("--checkpoint is required for PPO")
        self.strategy, self.paths = strategy, paths
        self.episodes, self.stochastic, self.seed = episodes, stochastic, seed
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.runtime = None
        self.checkpoint_error = None
        if checkpoint is not None:
            try:
                self.runtime = PolicyRuntime(checkpoint)
            except (
                OSError,
                ValueError,
                ImportError,
                ModuleNotFoundError,
                RuntimeError,
            ) as exc:
                if strategy != "auto":
                    raise
                self.checkpoint_error = f"{type(exc).__name__}: {exc}"

    def decide(self, problem: Problem) -> dict:
        catalogue = PathCatalog(problem, self.paths)
        t0 = perf_counter()
        problems: list[Candidate] = []
        failures: list[str] = []
        if self.checkpoint_error is not None:
            failures.append("checkpoint: " + self.checkpoint_error)
        if self.strategy in ("greedy", "lookahead", "multistart", "auto"):
            method = "lookahead" if self.strategy == "auto" else self.strategy
            start = perf_counter()
            try:
                solution = solve_baseline(problem, catalogue, method)
                problems.append(
                    Candidate(method, solution, 0, (perf_counter() - start) * 1000)
                )
            except (ValueError, RuntimeError) as exc:
                failures.append(f"{method}: {type(exc).__name__}: {exc}")
                if self.strategy != "auto":
                    raise
        if self.strategy in ("ppo", "auto"):
            if self.runtime is None:
                if self.strategy == "ppo":
                    raise ValueError("PPO checkpoint unavailable")
            else:
                try:
                    # Must not silently assess an incompatible checkpoint or paths count.
                    if self.paths != int(self.runtime.metadata["paths"]):
                        raise ValueError("paths mismatch with PPO checkpoint")
                    validate_compatible(
                        [problem], reference=self.runtime.metadata["signature"]
                    )
                    problems.append(
                        self.runtime.predict(
                            problem,
                            episodes=self.episodes,
                            stochastic=self.stochastic,
                            seed=self.seed,
                        )
                    )
                except (ValueError, RuntimeError, OSError) as exc:
                    if self.strategy == "ppo":
                        raise
                    failures.append(f"ppo: {type(exc).__name__}: {exc}")
        if not problems:
            raise RuntimeError("No feasible decision: " + "; ".join(failures))
        # Tie-break deterministically in favor of the baseline (no false PPO credit).
        selected = max(problems, key=lambda c: c.solution.quantities.sum())
        decision = decision_json(
            problem,
            selected.solution,
            catalogue,
            checkpoint=self.checkpoint or "none",
            policy_steps=selected.steps,
            seed=self.seed,
        )
        decision.update(
            {
                "algorithm": selected.label,
                "requested_strategy": self.strategy,
                "selected_method": selected.label,
                "candidate_scores_mbps": {
                    c.label: float(c.solution.quantities.sum()) for c in problems
                },
                "candidate_run_ms": {c.label: round(c.elapsed_ms, 3) for c in problems},
                "decision_wall_ms": round((perf_counter() - t0) * 1000, 3),
                "stochastic_actions": bool(self.stochastic),
                "episodes": self.episodes if selected.label == "ppo" else 0,
                "warnings": failures,
                "demand_upper_bound_mbps": sum(d.max_mbps for d in problem.demands),
                "certified_optimal": int(selected.solution.quantities.sum())
                == sum(d.max_mbps for d in problem.demands),
            }
        )
        return decision
