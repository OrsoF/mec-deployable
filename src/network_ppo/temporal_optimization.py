"""Causal receding-horizon beam MPC with feasible allocation candidates.

Beam MPC is an APPROXIMATION, not an exact MPC optimum. Candidate enumeration,
beam pruning and solver runtime are reported and comparable across policies.
No future realized demand or capacity enters the optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .constraints import assess
from .dynamics import TemporalScenario
from .heuristic import feasible_seed, greedy_uplift, routing_lookahead_greedy
from .paths import PathCatalog
from .problem import Problem, Solution
from .temporal_costs import StepScore, stage_score


def allocate(problem: Problem, catalog: PathCatalog, placement: np.ndarray,
             routes: np.ndarray) -> Solution | None:
    """Keep requested placement/routing, reset demands to minima and greedily fill."""
    q = np.array([d.min_mbps for d in problem.demands], dtype=int)
    candidate = Solution(np.asarray(placement, dtype=int).copy(),
                         np.asarray(routes, dtype=int).copy(), q)
    if not assess(problem, candidate, catalog).feasible:
        return None
    return greedy_uplift(problem, catalog, candidate)


def candidate_allocations(problem: Problem, catalog: PathCatalog, previous: Solution,
                          *, max_candidates: int = 9) -> list[Solution]:
    """Bounded collection of independently checked configurations.

    Includes 'keep current configuration', a fresh seed, locally improved
    routing, limited single-demand reroutes, and one-service relocations.
    Candidate limit makes MPC latency controllable and auditable.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    output: list[Solution] = []
    seen: set[tuple] = set()

    def add(sol: Solution | None):
        if sol is None or len(output) >= max_candidates:
            return
        key = (*sol.placement.tolist(), -1, *sol.routes.tolist())
        if key not in seen and assess(problem, sol, catalog).feasible:
            seen.add(key)
            output.append(sol)

    add(allocate(problem, catalog, previous.placement, previous.routes))
    try:
        default = feasible_seed(problem, catalog)
        add(greedy_uplift(problem, catalog, default))
        # This candidate is a strong static optimizer, not just a weak strawman.
        add(routing_lookahead_greedy(problem, catalog, default))
    except ValueError:
        default = None
    roots = [previous]
    if default is not None:
        roots.append(default)
    # Include relocation choices before spending the entire candidate budget on
    # route changes. This makes temporal service-migration decisions reachable.
    for service in range(len(problem.services)):
        for node in range(len(problem.nodes)):
            if node == int(previous.placement[service]):
                continue
            placement = previous.placement.copy()
            placement[service] = node
            add(allocate(problem, catalog, placement, previous.routes))
            if len(output) >= max_candidates:
                return output
    for root in roots:
        if len(output) >= max_candidates:
            break
        for d in range(len(problem.demands)):
            for r in range(catalog.k):
                if r == int(root.routes[d]):
                    continue
                routes = root.routes.copy()
                routes[d] = r
                add(allocate(problem, catalog, root.placement, routes))
                if len(output) >= max_candidates:
                    return output
    return output


def initial_allocation(scenario: TemporalScenario, t: int = 0, *, paths: int = 3) -> Solution:
    p = scenario.problem_at(t)
    catalog = PathCatalog(p, paths)
    return feasible_seed(p, catalog)


@dataclass(frozen=True)
class PlannedDecision:
    solution: Solution
    method: str
    runtime_ms: float
    candidate_count: int
    fallback: bool
    estimated_value: float
    lookahead_evaluated: int = 1
    budget_exhausted: bool = False


class TemporalPlanner:
    def __init__(self, *, strategy: str = "mpc", horizon: int = 4,
                 beam_width: int = 3, max_candidates: int = 9, paths: int = 3,
                 max_planning_ms: float | None = None, max_expansions: int | None = 40):
        if strategy not in ("keep", "greedy", "myopic", "mpc"):
            raise ValueError("Unknown temporal strategy")
        if min(horizon, beam_width, max_candidates, paths) <= 0 or (
            max_expansions is not None and max_expansions <= 0) or (
            max_planning_ms is not None and max_planning_ms <= 0):
            raise ValueError("Positive horizon, beam_width, max_candidates, paths required")
        self.strategy = strategy
        self.horizon = horizon
        self.beam_width = beam_width
        self.max_candidates = max_candidates
        self.paths = paths
        self.max_planning_ms = max_planning_ms
        self.max_expansions = max_expansions

    def decide(self, scenario: TemporalScenario, t: int, previous: Solution) -> PlannedDecision:
        start = perf_counter()
        current = scenario.problem_at(t)
        catalog = PathCatalog(current, self.paths)
        if self.strategy == "keep":
            # True no-reoptimization baseline: preserve last committed flows as
            # well as configuration (clamp only to current contract bounds).
            mins = np.array([d.min_mbps for d in current.demands], dtype=int)
            maxs = np.array([d.max_mbps for d in current.demands], dtype=int)
            q = np.clip(previous.quantities, mins, maxs)
            chosen = Solution(previous.placement.copy(), previous.routes.copy(), q)
            while not assess(current, chosen, catalog).feasible and np.any(chosen.quantities > mins):
                # Deterministic safe decrease for reduced capacities/latency.
                i = int(np.argmax(chosen.quantities - mins))
                chosen.quantities[i] -= 1
            if assess(current, chosen, catalog).feasible:
                return PlannedDecision(chosen, "keep", (perf_counter()-start)*1e3, 1, False, 0.0)
        if self.strategy == "greedy":
            chosen = allocate(current, catalog, previous.placement, previous.routes)
            if chosen is not None:
                return PlannedDecision(chosen, "greedy", (perf_counter()-start)*1e3, 1, False, 0.0)

        depth = 1 if self.strategy in ("myopic", "greedy", "keep") else self.horizon
        # Beam states: (accumulated discounted value, simulated allocation, initial action).
        beam: list[tuple[float, Solution, Solution | None]] = [(0.0, previous, None)]
        evaluated = 0
        reached = 0
        exhausted = False
        for offset in range(min(depth, scenario.horizon - t)):
            if offset > 0 and ((self.max_expansions is not None and evaluated >= self.max_expansions) or (
                self.max_planning_ms is not None and (perf_counter()-start)*1000 >= self.max_planning_ms)):
                exhausted = True
                break
            future = scenario.information_problem(t, t + offset)
            future_catalog = PathCatalog(future, self.paths)
            # Deduplicate identical configurations reached along different beam paths.
            branches: dict[tuple,tuple[float, Solution, Solution]] = {}
            incomplete = False
            for score, state, first in beam:
                if offset > 0 and (
                    (self.max_expansions is not None and evaluated >= self.max_expansions)
                    or (self.max_planning_ms is not None
                        and (perf_counter()-start)*1000 >= self.max_planning_ms)
                ):
                    incomplete=True
                    break
                candidates = candidate_allocations(future, future_catalog, state,
                                                    max_candidates=self.max_candidates)
                for sol in candidates:
                    reward = stage_score(scenario, future, state, sol, future_catalog).reward
                    evaluated += 1
                    row = (score + scenario.discount ** offset * reward,
                           sol, sol if first is None else first)
                    key = (*sol.placement.tolist(), -1, *sol.routes.tolist(), -1, *sol.quantities.tolist())
                    if key not in branches or row[0] > branches[key][0]:
                        branches[key] = row
            if incomplete:
                exhausted=True
                break   # retain the last COMPLETE beam depth
            if not branches:
                break
            beam = sorted(branches.values(), key=lambda z: z[0], reverse=True)[:self.beam_width]
            reached = offset + 1
        if not beam or beam[0][2] is None:
            # Fallback must be recorded and must be feasible, never silent.
            sol = allocate(current, catalog, previous.placement, previous.routes)
            if sol is None:
                sol = feasible_seed(current, catalog)
            return PlannedDecision(sol, "fallback", (perf_counter()-start)*1e3, evaluated, True, 0.0, reached, exhausted)
        best_val, _, selected = max(beam, key=lambda row: row[0])
        # Forecast may differ from the real demand at t only for future periods.
        # At t, the information problem has the exact contemporaneous observation.
        assert assess(current, selected, catalog).feasible
        return PlannedDecision(selected, self.strategy, (perf_counter()-start)*1e3,
                               evaluated, False, float(best_val), reached, exhausted)
