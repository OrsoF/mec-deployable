"""Synthetic, causally forecastable temporal extensions of a static network instance.

A scenario stores realized demand trajectories and an *exogenous*, known-in-advance
mean-demand forecast. The forecast is NOT calculated from future realized traffic;
MPC and PPO receive the same public information. This is a research simulator,
not a learned traffic forecaster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .instance_io import parse_instance
from .problem import Problem, Solution


@dataclass(frozen=True)
class TemporalScenario:
    id: str
    base: Problem
    realized_mbps: np.ndarray  # [T, D] integer bounds per demand
    forecast_mbps: np.ndarray  # [T, D] known exogenous forecast, NOT realized future
    capacity_scale: np.ndarray  # [T, L], realized contemporaneously (not anticipated)
    migration_cost: float = 1.0  # utility units per migrated service
    route_change_cost: float = 0.3  # utility units per changed demand route
    active_node_cost: float = 0.1  # utility units per active compute node per step
    discount: float = 0.98
    split: str = "unspecified"
    initial_time: int = 0
    full_horizon: int | None = None

    def __post_init__(self):
        realized = np.asarray(self.realized_mbps)
        forecast = np.asarray(self.forecast_mbps)
        scales = np.asarray(self.capacity_scale)
        t, d, n_links = len(realized), len(self.base.demands), len(self.base.links)
        if (
            t < 1
            or realized.shape != (t, d)
            or forecast.shape != (t, d)
            or scales.shape != (t, n_links)
        ):
            raise ValueError(
                "Temporal arrays must be [T, demands], [T, demands], [T, links]"
            )
        if not (
            np.isfinite(realized).all()
            and np.isfinite(forecast).all()
            and np.isfinite(scales).all()
        ):
            raise ValueError("Non-finite temporal values")
        if not np.all(realized == np.floor(realized)):
            raise ValueError("Realized bandwidth bounds must be integer Mbps")
        mins = np.array([dm.min_mbps for dm in self.base.demands])
        if (realized < mins).any() or (forecast < mins).any() or (scales <= 0).any():
            raise ValueError("Demand below minimum or non-positive capacity scale")
        if min(self.migration_cost, self.route_change_cost, self.active_node_cost) < 0:
            raise ValueError("Costs must be nonnegative")
        if self.initial_time < 0 or (
            self.full_horizon is not None and self.full_horizon < self.initial_time + t
        ):
            raise ValueError("Invalid absolute time/episode length")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must lie in (0,1]")
        for name, a, dtype in (
            ("realized_mbps", realized, np.int64),
            ("forecast_mbps", forecast, float),
            ("capacity_scale", scales, float),
        ):
            frozen = np.array(a, dtype=dtype, copy=True)
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)

    @property
    def horizon(self) -> int:
        return len(self.realized_mbps)

    def problem_at(self, t: int, *, predicted: bool = False) -> Problem:
        """Predicted future capacity assumes CURRENT observed capacity persists.

        No future capacity or demand realizations are read when predicted=True.
        """
        if not 0 <= t < self.horizon:
            raise IndexError(t)
        levels = self.forecast_mbps[t] if predicted else self.realized_mbps[t]
        bounds = np.maximum(levels.astype(int), [d.min_mbps for d in self.base.demands])
        new_demands = tuple(
            replace(d, max_mbps=int(bounds[i])) for i, d in enumerate(self.base.demands)
        )
        # Used only for actually observed contemporaneous state; the MPC caller
        # overwrites future link factors with the latest observed factor.
        links = tuple(
            replace(
                link,
                capacity_mbps=float(link.capacity_mbps * self.capacity_scale[t, i]),
            )
            for i, link in enumerate(self.base.links)
        )
        return replace(self.base, id=f"{self.id}@{t}", demands=new_demands, links=links)

    def information_problem(self, now: int, target: int) -> Problem:
        """Current actual demand, otherwise public mean forecast; current link status only."""
        if not (0 <= now <= target < self.horizon):
            raise IndexError((now, target))
        levels = (
            self.realized_mbps[now] if target == now else self.forecast_mbps[target]
        )
        demands = tuple(
            replace(d, max_mbps=int(max(d.min_mbps, round(levels[i]))))
            for i, d in enumerate(self.base.demands)
        )
        links = tuple(
            replace(
                link,
                capacity_mbps=float(link.capacity_mbps * self.capacity_scale[now, i]),
            )
            for i, link in enumerate(self.base.links)
        )
        return replace(
            self.base,
            id=f"{self.id}@forecast_{now}_{target}",
            demands=demands,
            links=links,
        )


def parse_temporal_scenario(data: dict) -> TemporalScenario:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Temporal schema_version must be 1")
    if not isinstance(data.get("base_instance"), dict):
        raise ValueError("base_instance must be an embedded static JSON instance")
    base = parse_instance(data["base_instance"])
    realized = np.asarray(data["realized_mbps"])
    forecast = np.asarray(data["forecast_mbps"])
    capacity = np.asarray(
        data.get("capacity_scale", np.ones((len(realized), len(base.links))))
    )
    return TemporalScenario(
        id=str(data["id"]),
        base=base,
        realized_mbps=realized,
        forecast_mbps=forecast,
        capacity_scale=capacity,
        migration_cost=float(data.get("migration_cost", 1.0)),
        route_change_cost=float(data.get("route_change_cost", 0.3)),
        active_node_cost=float(data.get("active_node_cost", 0.1)),
        discount=float(data.get("discount", 0.98)),
        split=str(data.get("split", "unspecified")),
        initial_time=int(data.get("initial_time", 0)),
        full_horizon=data.get("full_horizon"),
    )


def load_temporal_scenario(path: str | Path) -> TemporalScenario:
    return parse_temporal_scenario(json.loads(Path(path).read_text(encoding="utf-8")))


def solution_to_state(solution: Solution) -> dict:
    return {
        "placement": solution.placement.astype(int).tolist(),
        "routes": solution.routes.astype(int).tolist(),
        "quantities": solution.quantities.astype(int).tolist(),
    }


def solution_from_state(raw: dict) -> Solution:
    if not isinstance(raw, dict):
        raise ValueError("previous must be an object")
    arrays = []
    for key in ("placement", "routes", "quantities"):
        values = np.asarray(raw[key], dtype=float)
        if (
            values.ndim != 1
            or not np.isfinite(values).all()
            or not np.all(values == np.floor(values))
        ):
            raise ValueError(f"previous.{key} must be a finite integer array")
        arrays.append(values.astype(int))
    return Solution(*arrays)


def parse_online_snapshot(raw: dict) -> tuple[TemporalScenario, Solution | None]:
    """Causal online request: current actual, *only* public future forecast.

    Future realized demand and future link state are impossible to supply through
    this contract. The stored future values are forecast placeholders used only
    to satisfy the fixed-size simulator representation.
    """
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Online snapshot schema_version must be 1")
    if "realized_mbps" in raw or "capacity_scale" in raw:
        raise ValueError("Online input must not contain future realized trajectories")
    base = parse_instance(raw["base_instance"])
    current_values = np.asarray(raw["observed_demand_mbps"], dtype=float)
    if not np.isfinite(current_values).all() or not np.all(
        current_values == np.floor(current_values)
    ):
        raise ValueError("observed_demand_mbps must contain integer Mbps")
    current = current_values.astype(int)
    forecasts = np.asarray(raw["forecast_mbps"], dtype=float)
    cap = np.asarray(raw["observed_capacity_scale"], dtype=float)
    d, n_links = len(base.demands), len(base.links)
    if (
        current.shape != (d,)
        or forecasts.ndim != 2
        or forecasts.shape[1] != d
        or cap.shape != (n_links,)
    ):
        raise ValueError(
            "Snapshot observed demand, forecast and capacity dimensions differ"
        )
    if len(forecasts) < 1:
        raise ValueError("Forecast must contain at least current period")
    # Future values are forecast-only placeholders; only t=0 is actually executed.
    future = np.maximum(forecasts[1:], [dm.min_mbps for dm in base.demands]).astype(int)
    realized = np.concatenate((current[None, :], future), axis=0)
    scenario = TemporalScenario(
        id=str(raw["id"]),
        base=base,
        realized_mbps=realized,
        forecast_mbps=forecasts,
        capacity_scale=np.tile(cap, (len(realized), 1)),
        migration_cost=float(raw.get("migration_cost", 1.0)),
        route_change_cost=float(raw.get("route_change_cost", 0.3)),
        active_node_cost=float(raw.get("active_node_cost", 0.1)),
        discount=float(raw.get("discount", 0.98)),
        split="online",
        initial_time=int(raw.get("absolute_period", 0)),
        full_horizon=int(raw.get("episode_length", len(realized))),
    )
    previous = (
        None if raw.get("previous") is None else solution_from_state(raw["previous"])
    )
    return scenario, previous
