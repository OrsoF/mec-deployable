import numpy as np

from mecbench.domain.conversions import make_state, split_state, to_int_array


def move_service(problem, s, k: int, v: int) -> np.ndarray:
    p, r, q = split_state(problem, to_int_array(s))
    p = p.copy()
    p[int(k)] = int(v)
    return make_state(p, r, q)


def reroute_demand(problem, s, d: int, path_index: int) -> np.ndarray:
    p, r, q = split_state(problem, to_int_array(s))
    r = r.copy()
    r[int(d)] = int(path_index)
    return make_state(p, r, q)


def shift_quantity(problem, s, d: int, dq: int) -> np.ndarray:
    p, r, q = split_state(problem, to_int_array(s))
    q = q.copy()
    q[int(d)] += int(dq)
    q = np.clip(q, problem.min_flows, problem.max_flows)
    return make_state(p, r, q)
