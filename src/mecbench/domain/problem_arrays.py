"""Owned immutable numerical inputs, shared by every consumer of a problem."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray, IntArray, ProblemData

if TYPE_CHECKING:
    from mecbench.domain.network import NetworkGraph
    from mecbench.domain.services import ServicePackage


def _floats(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _indices(values: ArrayLike) -> IntArray:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)) or np.any(array != np.rint(array)):
        raise ValueError("Problem indices must be finite integers.")
    array = np.asarray(array, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class ProblemArrays:
    origins: IntArray
    demand_to_service: IntArray
    min_flows: FloatArray
    max_flows: FloatArray
    latency_limits: FloatArray
    service_resources: FloatArray
    arc_capacities: FloatArray
    latency_slopes: FloatArray
    latency_offsets: FloatArray
    node_resources: FloatArray
    arcs: tuple[tuple[int, int], ...]
    arc_index: Mapping[tuple[int, int], int] = field(init=False)
    demand_counts: IntArray = field(init=False)

    def __post_init__(self) -> None:
        for name in ("origins", "demand_to_service"):
            object.__setattr__(self, name, _indices(getattr(self, name)))
        for name in (
            "min_flows",
            "max_flows",
            "latency_limits",
            "service_resources",
            "arc_capacities",
            "latency_slopes",
            "latency_offsets",
            "node_resources",
        ):
            object.__setattr__(self, name, _floats(getattr(self, name)))
        arcs = tuple(tuple(arc) for arc in self.arcs)
        object.__setattr__(self, "arcs", arcs)
        object.__setattr__(
            self, "arc_index", MappingProxyType({arc: a for a, arc in enumerate(arcs)})
        )
        self.validate()
        object.__setattr__(
            self,
            "demand_counts",
            _indices(
                np.bincount(
                    self.demand_to_service,
                    minlength=self.n_services,
                )
            ),
        )

    @property
    def n_services(self) -> int:
        return int(self.service_resources.size)

    @property
    def n_demands(self) -> int:
        return int(self.origins.size)

    @property
    def n_nodes(self) -> int:
        return int(self.node_resources.size)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    def validate(self) -> None:
        for name, size in (
            ("origins", self.n_demands),
            ("demand_to_service", self.n_demands),
            ("min_flows", self.n_demands),
            ("max_flows", self.n_demands),
            ("latency_limits", self.n_demands),
            ("service_resources", self.n_services),
            ("arc_capacities", self.n_arcs),
            ("latency_slopes", self.n_arcs),
            ("latency_offsets", self.n_arcs),
            ("node_resources", self.n_nodes),
        ):
            array = getattr(self, name)
            if array.shape != (size,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite vector of shape ({size},).")
        if np.any((self.origins < 0) | (self.origins >= self.n_nodes)):
            raise ValueError("Demand origins must refer to known nodes.")
        if np.any((self.demand_to_service < 0) | (self.demand_to_service >= self.n_services)):
            raise ValueError("Demand service indices must refer to known services.")
        if len(self.arc_index) != self.n_arcs:
            raise ValueError("Problem arcs must be unique.")
        if any(
            len(arc) != 2
            or any(
                not isinstance(node, int | np.integer) or not 0 <= node < self.n_nodes
                for node in arc
            )
            for arc in self.arcs
        ):
            raise ValueError("Arcs must connect known integer node indices.")

    @classmethod
    def from_components(cls, network: "NetworkGraph", package: "ServicePackage") -> "ProblemArrays":
        services = package.services
        return cls(
            origins=np.asarray([o for s in services for o in s.origin_nodes]),
            demand_to_service=np.asarray([k for k, s in enumerate(services) for _ in s.task_list]),
            min_flows=np.asarray([q for s in services for q in s.min_flows]),
            max_flows=np.asarray([q for s in services for q in s.max_flows]),
            latency_limits=np.asarray([limit for s in services for limit in s.max_latencies]),
            service_resources=np.asarray([s.needed_resources for s in services]),
            arc_capacities=np.asarray(network.capacities),
            latency_slopes=np.asarray(network.alphas),
            latency_offsets=np.asarray(network.betas),
            node_resources=np.asarray(network.available_resources),
            arcs=tuple(network.arc_list),
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # Reconstruct through __post_init__ to restore immutability after pickle/deepcopy.
        return type(self), tuple(getattr(self, f.name) for f in fields(self) if f.init)


def problem_arrays(problem: ProblemData | ProblemArrays) -> ProblemArrays:
    """Reuse canonical data, or snapshot a structural legacy problem at the boundary."""
    if isinstance(problem, ProblemArrays):
        return problem
    existing = getattr(problem, "data", None)
    if isinstance(existing, ProblemArrays):
        return existing
    network = problem.network_graph
    return ProblemArrays(
        origins=np.asarray(problem.origins),
        demand_to_service=np.asarray(problem.demand_to_service),
        min_flows=np.asarray(problem.min_flows),
        max_flows=np.asarray(problem.max_flows),
        latency_limits=np.asarray(problem.latencies),
        service_resources=np.asarray(problem.needed_resources),
        arc_capacities=np.asarray(network.capacities),
        latency_slopes=np.asarray(network.alphas),
        latency_offsets=np.asarray(network.betas),
        node_resources=np.asarray(network.available_resources),
        arcs=tuple(network.arc_list),
    )
