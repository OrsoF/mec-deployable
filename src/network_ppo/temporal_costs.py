"""Stage reward with *physical* service migrations and routing changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import assess
from .dynamics import TemporalScenario
from .paths import PathCatalog
from .problem import Problem, Solution


@dataclass(frozen=True)
class StepScore:
    reward: float
    served_mbps: float
    migrations: int
    route_changes: int
    active_nodes: int
    energy_cost: float


def stage_score(
    scenario: TemporalScenario,
    problem: Problem,
    previous: Solution,
    chosen: Solution,
    catalog: PathCatalog,
) -> StepScore:
    check = assess(problem, chosen, catalog)
    if not check.feasible:
        raise ValueError("Infeasible temporal decision: " + "; ".join(check.violations))
    migrations = int(np.count_nonzero(previous.placement != chosen.placement))
    reroutes = int(np.count_nonzero(previous.routes != chosen.routes))
    active = int(np.count_nonzero(check.node_load > 1e-9))
    energy = scenario.active_node_cost * active
    reward = (
        check.total_mbps
        - scenario.migration_cost * migrations
        - scenario.route_change_cost * reroutes
        - energy
    )
    return StepScore(
        float(reward), check.total_mbps, migrations, reroutes, active, float(energy)
    )
