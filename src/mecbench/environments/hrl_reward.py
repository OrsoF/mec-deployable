import numpy as np

from mecbench.domain.conversions import split_state
from mecbench.domain.problem_arrays import problem_arrays


def served_quantity(problem, s) -> float:
    return float(np.sum(split_state(problem, s)[2]))


def placement_routing_score(problem, checker, rng, s, steps: int, increment: int) -> float:
    data = problem_arrays(problem)
    p, r, q = split_state(problem, s)
    p = p.astype(int).tolist()
    r = r.astype(int).tolist()
    q = list(data.min_flows)
    if not checker.is_feasible(p, r, q):
        return -1.0

    D = data.n_demands
    for _ in range(int(steps)):
        d = int(rng.integers(D))
        if q[d] >= data.max_flows[d]:
            continue
        dq = min(increment, data.max_flows[d] - q[d])
        q[d] += dq
        if not checker.is_feasible(p, r, q):
            q[d] -= dq
    return float(sum(q))


def normalized_delta(new_score: float, old_score: float) -> float:
    return float((new_score - old_score) / 100.0)
