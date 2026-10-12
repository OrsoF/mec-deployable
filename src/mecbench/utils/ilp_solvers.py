"""Parameters, optimizer options and canonical extraction for mathematical solvers.

This module does not import Pyomo, so basic usage stays independent of exact extras.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import ArcArrays
from mecbench.domain.problem_arrays import ProblemArrays as ProblemArrays
from mecbench.domain.problem_arrays import problem_arrays as problem_arrays


def configure_gurobi(
    opt: Any,
    options: Mapping[str, Any] | None = None,
    time_limit: float | None = None,
    thread_limit: int | None = None,
) -> Any:
    if time_limit is not None:
        opt.options["TimeLimit"] = time_limit
    if thread_limit is not None:
        opt.options["threads"] = thread_limit
    for key, value in (options or {}).items():
        opt.options[key] = value
    return opt


def objective_value(q: ArrayLike) -> float:
    return float(np.sum(np.asarray(q, dtype=float)))


def variable_value(item: Any) -> float:
    value = getattr(item, "value", item)
    return float("nan") if value is None else float(value)


def arc_model_arrays(
    x: Any,
    y: Any,
    q: Any,
    demand_count: int,
    arc_count: int,
    service_count: int,
    node_count: int,
) -> ArcArrays:
    return (
        np.array(
            [[variable_value(x[d, a]) for a in range(arc_count)] for d in range(demand_count)],
            dtype=float,
        ).reshape(demand_count, arc_count),
        np.array(
            [[variable_value(y[s, u]) for u in range(node_count)] for s in range(service_count)],
            dtype=float,
        ).reshape(service_count, node_count),
        np.array([variable_value(q[d]) for d in range(demand_count)], dtype=float),
    )


def path_model_arrays(
    weights: Any,
    y: Any,
    q: Any,
    paths: Sequence[Sequence[int]],
    path_to_demand: Sequence[int],
    arc_index: Mapping[tuple[int, int], int],
    demand_count: int,
    arc_count: int,
    service_count: int,
    node_count: int,
) -> ArcArrays:
    x = np.zeros((demand_count, arc_count))
    for p, path in enumerate(paths):
        weight = variable_value(weights[p])
        d = path_to_demand[p]
        for source, destination in zip(path, path[1:], strict=False):
            x[d, arc_index[source, destination]] += weight
    placement = np.array(
        [[variable_value(y[s, u]) for u in range(node_count)] for s in range(service_count)],
        dtype=float,
    ).reshape(service_count, node_count)
    quantities = np.array([variable_value(q[d]) for d in range(demand_count)], dtype=float)
    return x, placement, quantities
