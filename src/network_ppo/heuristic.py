"""Small feasible-state constructor, not a competing optimization solver.

The seed is deliberately initialized at minimum requested Mbps. PPO must learn
which feasible edits increase throughput. A greedy uplift is used ONLY for
reporting a transparent, non-PPO reference score at evaluation time.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

from .constraints import assess
from .paths import PathCatalog
from .problem import Problem, Solution


def _route_choices(n_demands: int, n_paths: int, limit: int):
    """Yield bounded path choices without materializing K**D combinations."""
    issued = 0
    for edits in range(n_demands + 1):
        for positions in combinations(range(n_demands), edits):
            for choices in product(range(1, n_paths), repeat=edits):
                routes = [0] * n_demands
                for index, chosen in zip(positions, choices):
                    routes[index] = chosen
                yield routes
                issued += 1
                if issued >= limit:
                    return


def feasible_seed(
    problem: Problem,
    paths: PathCatalog,
    *,
    beam_width: int = 64,
    route_trials: int = 1024,
) -> Solution:
    """Find feasible placements/routes at minimum Mbps; raise if no seed found.

    CPU-constrained beam search considers nearest demand origins first, then
    route changes. The search is bounded, so failure does NOT prove infeasibility.
    """
    if beam_width < 1 or route_trials < 1:
        raise ValueError("beam_width and route_trials must be positive")
    n_services = len(problem.services)
    n_demands = len(problem.demands)
    min_q = np.array([d.min_mbps for d in problem.demands], dtype=np.int64)
    resources = np.array([node.cpu for node in problem.nodes], dtype=float)
    order = sorted(range(n_services), key=lambda s: -problem.services[s].cpu)
    # List of (estimated network cost, placement array, remaining CPU).
    beam: list[tuple[float, tuple[int, ...], np.ndarray]] = [
        (0.0, (-1,) * n_services, resources)
    ]
    for s in order:
        need = problem.services[s].cpu
        next_beam = []
        for cost, placed, remaining in beam:
            for node in range(len(problem.nodes)):
                if remaining[node] + 1e-9 < need:
                    continue
                candidate = list(placed)
                candidate[s] = node
                new_remaining = remaining.copy()
                new_remaining[node] -= need
                # Deterministic demand-weighted path length ranking.
                extra = sum(
                    max(1, min_q[d]) * (len(paths.route(dm.origin, node, 0)) - 1)
                    for d, dm in enumerate(problem.demands)
                    if dm.service == s
                )
                next_beam.append((cost + extra, tuple(candidate), new_remaining))
        if not next_beam:
            raise ValueError(
                f"No CPU-feasible placement found for service {problem.services[s].id}"
            )
        next_beam.sort(key=lambda row: (row[0], row[1]))
        beam = next_beam[:beam_width]
    for _, placement, _ in beam:
        for choice in _route_choices(n_demands, paths.k, route_trials):
            candidate = Solution(
                np.asarray(placement, dtype=np.int64),
                np.asarray(choice, dtype=np.int64),
                min_q.copy(),
            )
            if assess(problem, candidate, paths).feasible:
                return candidate
    raise ValueError(
        f"No feasible minimum-Mbps seed found for '{problem.id}' within the bounded "
        "placement/route search. Check CPU, minimum traffic and latency constraints; "
        "or increase seed search limits."
    )


def greedy_uplift(problem: Problem, paths: PathCatalog, initial: Solution) -> Solution:
    """Reference: repeatedly increase integer Mbps without changing placement/routes."""
    candidate = initial.copy()
    if not assess(problem, candidate, paths).feasible:
        raise ValueError("Greedy reference needs a feasible initial state")
    while True:
        changed = False
        for d, demand in enumerate(problem.demands):
            if candidate.quantities[d] >= demand.max_mbps:
                continue
            trial = candidate.copy()
            trial.quantities[d] += 1
            if assess(problem, trial, paths).feasible:
                candidate = trial
                changed = True
        if not changed:
            return candidate


def routing_lookahead_greedy(
    problem: Problem, paths: PathCatalog, initial: Solution, *, max_rounds: int = 30
) -> Solution:
    """Best improving one-demand reroute, followed by feasible greedy allocation.

    Starts from greedy_uplift(initial). No service migrations and no infeasible
    intermediate decisions. This is NOT a globally optimal algorithm.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    current = greedy_uplift(problem, paths, initial)
    for _ in range(max_rounds):
        best = current
        best_score = int(current.quantities.sum())
        for demand in range(len(problem.demands)):
            for route in range(paths.k):
                if route == int(current.routes[demand]):
                    continue
                trial = current.copy()
                trial.routes[demand] = route
                if not assess(problem, trial, paths).feasible:
                    continue
                trial = greedy_uplift(problem, paths, trial)
                score = int(trial.quantities.sum())
                if score > best_score:
                    best, best_score = trial, score
        if best_score == int(current.quantities.sum()):
            break
        current = best
    return current


def multi_start_lookahead(
    problem: Problem,
    paths: PathCatalog,
    initial: Solution,
    *,
    max_starts: int = 20,
    max_rounds: int = 30,
) -> Solution:
    """Stronger deterministic baseline over minimum-flow route perturbations.

    Does not optimize service placement; does not claim global optimality.
    Bounded to at most max_starts feasible initializations including 'initial'.
    """
    if max_starts < 1:
        raise ValueError("max_starts must be >= 1")
    base = initial.copy()
    base.quantities = np.array([d.min_mbps for d in problem.demands], dtype=np.int64)
    best = routing_lookahead_greedy(problem, paths, base, max_rounds=max_rounds)
    starts = 1
    for d in range(len(problem.demands)):
        for route in range(paths.k):
            if starts >= max_starts:
                return best
            if route == base.routes[d]:
                continue
            trial = base.copy()
            trial.routes[d] = route
            if not assess(problem, trial, paths).feasible:
                continue
            starts += 1
            candidate = routing_lookahead_greedy(
                problem, paths, trial, max_rounds=max_rounds
            )
            if candidate.quantities.sum() > best.quantities.sum():
                best = candidate
    return best


BASELINES = {
    "greedy": greedy_uplift,
    "lookahead": routing_lookahead_greedy,
    "multistart": multi_start_lookahead,
}


def solve_baseline(
    problem: Problem, paths: PathCatalog, method: str, initial: Solution | None = None
) -> Solution:
    """Compute checked reference solution, independent from optional RL libraries."""
    if method not in BASELINES:
        raise ValueError(f"Unknown baseline: {method!r}; choose {list(BASELINES)}")
    initial = feasible_seed(problem, paths) if initial is None else initial
    solution = BASELINES[method](problem, paths, initial)
    report = assess(problem, solution, paths)
    if not report.feasible:
        raise RuntimeError(
            "Baseline generated infeasible solution: " + "; ".join(report.violations)
        )
    return solution
