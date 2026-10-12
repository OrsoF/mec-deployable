"""Human-readable, validated industrial JSON -> mecbench ProblemInstance."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from mecbench.domain.network import NetworkGraph
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.services import Demand, Service, ServicePackage


@dataclass(frozen=True)
class LoadedInstance:
    problem: ProblemInstance
    name: str
    node_ids: tuple[str, ...]
    service_ids: tuple[str, ...]
    demand_ids: tuple[str, ...]
    source: Path

    @property
    def signature(self) -> dict:
        """Canonical ordering and action/observation layout; no arbitrary permutations."""
        arcs = self.problem.data.arcs
        return {
            "nodes": list(self.node_ids),
            "arcs": [[self.node_ids[a], self.node_ids[b]] for a, b in arcs],
            "services": list(self.service_ids),
            "demands": list(self.demand_ids),
            "demand_to_service": self.problem.data.demand_to_service.tolist(),
        }


def _finite_number(value, location: str, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    f = float(value)
    if not math.isfinite(f) or f < 0 or (strictly_positive and f == 0):
        raise ValueError(f"{location} must be finite and {'> 0' if strictly_positive else '>= 0'}")
    return f


def _read_items(obj: dict, key: str) -> list[dict]:
    v = obj.get(key)
    if not isinstance(v, list) or not v or not all(isinstance(item, dict) for item in v):
        raise ValueError(f"{key} must be a non-empty list of objects")
    return v


def _unique_name(item: dict, used: set[str], label: str) -> str:
    name = item.get("id")
    if not isinstance(name, str) or not name or name in used:
        raise ValueError(f"{label}.id must be a unique, nonempty string: {name!r}")
    used.add(name)
    return name


def load_instance(path: str | Path) -> LoadedInstance:
    path = Path(path)
    source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version=1")
    name = source.get("id")
    if not isinstance(name, str) or not name:
        raise ValueError("instance.id must be a nonempty string")
    # Semantic unit contract. Conversions, if needed, must happen upstream.
    if source.get("units") != {"flow": "Mbps", "latency": "ms", "compute": "vCPU"}:
        raise ValueError("units must be flow=Mbps, latency=ms, compute=vCPU")
    nodes = _read_items(source, "nodes")
    links = _read_items(source, "links")
    services = _read_items(source, "services")
    seen: set[str] = set()
    node_ids = tuple(_unique_name(x, seen, "node") for x in nodes)
    mapping = {name: i for i, name in enumerate(node_ids)}
    resources = [_finite_number(item.get("cpu"), f"nodes[{i}].cpu") for i, item in enumerate(nodes)]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    edge_caps: list[float] = []
    alphas: list[float] = []
    betas: list[float] = []
    for i, item in enumerate(links):
        a, b = item.get("source"), item.get("target")
        if a not in mapping or b not in mapping or a == b:
            raise ValueError(f"links[{i}] has invalid endpoints")
        u, v = mapping[a], mapping[b]
        if graph.has_edge(u, v):
            raise ValueError(f"links[{i}] duplicates an undirected link")
        graph.add_edge(u, v)
        edge_caps.append(_finite_number(item.get("capacity_mbps"), f"links[{i}].capacity_mbps", strictly_positive=True))
        betas.append(_finite_number(item.get("latency_base_ms"), f"links[{i}].latency_base_ms"))
        alphas.append(_finite_number(item.get("latency_per_mbps_ms"), f"links[{i}].latency_per_mbps_ms"))
    if not nx.is_connected(graph):
        raise ValueError("Network must be connected")
    network = NetworkGraph(graph, resources, edge_caps, alphas, betas)
    seen_services: set[str] = set()
    seen_demands: set[str] = set()
    service_objs: list[Service] = []
    service_ids: list[str] = []
    demand_ids: list[str] = []
    for s, item in enumerate(services):
        service_ids.append(_unique_name(item, seen_services, "service"))
        needs = _finite_number(item.get("cpu"), f"services[{s}].cpu")
        ds: list[Demand] = []
        for d, demand in enumerate(_read_items(item, "demands")):
            demand_ids.append(_unique_name(demand, seen_demands, "demand"))
            origin = demand.get("origin")
            if origin not in mapping:
                raise ValueError(f"services[{s}].demands[{d}].origin unknown: {origin}")
            minimum = _finite_number(demand.get("min_mbps"), "min_mbps")
            maximum = _finite_number(demand.get("max_mbps"), "max_mbps")
            latency = _finite_number(demand.get("max_latency_ms"), "max_latency_ms", strictly_positive=True)
            # Current PPO actions are +/-1 Mbps, so all bounds must be integer.
            if int(minimum) != minimum or int(maximum) != maximum:
                raise ValueError("This PPO MVP requires integer flow bounds (Mbps)")
            ds.append(Demand(mapping[origin], minimum, maximum, latency))
        service_objs.append(Service(ds, network, needs))
    package = ServicePackage(service_objs, name)
    problem = ProblemInstance(network, package, name)
    return LoadedInstance(problem, name, node_ids, tuple(service_ids), tuple(demand_ids), path)


def validate_compatible(instances: list[LoadedInstance], reference: dict | None = None) -> dict:
    if not instances:
        raise ValueError("At least one instance is required")
    target = reference if reference is not None else instances[0].signature
    for instance in instances:
        if instance.signature != target:
            raise ValueError(
                f"Incompatible instance {instance.source}: node/edge/service/demand order "
                "or topology differs from checkpoint. Use a shared template (V1)."
            )
    return target
