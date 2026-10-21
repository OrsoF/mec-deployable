"""Domain data for a small, fixed-topology network-orchestration problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Node:
    id: str
    cpu: float


@dataclass(frozen=True)
class Link:
    source: int
    target: int
    capacity_mbps: float
    latency_base_ms: float
    latency_per_mbps_ms: float


@dataclass(frozen=True)
class Service:
    id: str
    cpu: float


@dataclass(frozen=True)
class Demand:
    id: str
    service: int
    origin: int
    min_mbps: int
    max_mbps: int
    max_latency_ms: float


@dataclass(frozen=True)
class Problem:
    id: str
    nodes: tuple[Node, ...]
    links: tuple[Link, ...]
    services: tuple[Service, ...]
    demands: tuple[Demand, ...]
    description: str = ""

    @property
    def arcs(self) -> tuple[tuple[int, int], ...]:
        # Each undirected physical link provides two independently capacitated arcs.
        return tuple(
            arc
            for e in self.links
            for arc in ((e.source, e.target), (e.target, e.source))
        )

    @property
    def signature(self) -> dict:
        """Mapping indices to identifiers; must match between train and inference."""
        return {
            "nodes": [n.id for n in self.nodes],
            "arcs": [[self.nodes[u].id, self.nodes[v].id] for u, v in self.arcs],
            "services": [s.id for s in self.services],
            "demands": [d.id for d in self.demands],
            "demand_to_service": [d.service for d in self.demands],
        }

    @property
    def arc_capacities(self) -> np.ndarray:
        return np.repeat([e.capacity_mbps for e in self.links], 2).astype(float)

    @property
    def arc_base_latencies(self) -> np.ndarray:
        return np.repeat([e.latency_base_ms for e in self.links], 2).astype(float)

    @property
    def arc_latency_slopes(self) -> np.ndarray:
        return np.repeat([e.latency_per_mbps_ms for e in self.links], 2).astype(float)


@dataclass
class Solution:
    """One chosen node per service, path index per demand, integer Mbps per demand."""

    placement: np.ndarray
    routes: np.ndarray
    quantities: np.ndarray

    def copy(self) -> "Solution":
        return Solution(
            self.placement.copy(), self.routes.copy(), self.quantities.copy()
        )


def validate_compatible(problems: list[Problem], reference: dict | None = None) -> dict:
    if not problems:
        raise ValueError("At least one instance is required")
    expected = reference if reference is not None else problems[0].signature
    for case in problems:
        if case.signature != expected:
            raise ValueError(
                f"Incompatible instance '{case.id}': network topology / indexed service or demand "
                "mapping differs from checkpoint. This fixed-topology V1 requires the same "
                "ordered node, link, service and demand identifiers."
            )
    return expected
