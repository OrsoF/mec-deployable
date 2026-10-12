"""Strict, human-readable industrial JSON input; no pickle or arbitrary code."""
from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx

from .problem import Demand, Link, Node, Problem, Service


def _items(obj: dict, field: str) -> list[dict]:
    value = obj.get(field)
    if not isinstance(value, list) or not value or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"'{field}' must be a non-empty list of objects")
    return value


def _name(data: dict, used: set[str], context: str) -> str:
    value = data.get("id")
    if not isinstance(value, str) or not value.strip() or value in used:
        raise ValueError(f"{context}.id must be a unique nonempty string")
    used.add(value)
    return value


def _number(data: dict, key: str, context: str, *, positive: bool = False, integer: bool = False):
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context}.{key} must be a finite number")
    if value < 0 or (positive and value <= 0):
        raise ValueError(f"{context}.{key} must be {'> 0' if positive else '>= 0'}")
    if integer and int(value) != value:
        raise ValueError(f"{context}.{key} must be an integer (whole Mbps)")
    return int(value) if integer else float(value)


def load_instance(path: str | Path) -> Problem:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError(f"{path}: expected schema_version=1")
    problem_id = raw.get("id")
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError("instance.id must be nonempty")
    if raw.get("units") != {"flow": "Mbps", "latency": "ms", "compute": "vCPU"}:
        raise ValueError("units must be flow=Mbps, latency=ms, compute=vCPU")
    seen_nodes: set[str] = set()
    nodes = tuple(Node(_name(x, seen_nodes, "node"), _number(x, "cpu", "node"))
                  for x in _items(raw, "nodes"))
    node_lookup = {n.id: i for i, n in enumerate(nodes)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    links: list[Link] = []
    for i, entry in enumerate(_items(raw, "links")):
        source, target = entry.get("source"), entry.get("target")
        if source not in node_lookup or target not in node_lookup or source == target:
            raise ValueError(f"links[{i}] invalid source/target")
        a, b = node_lookup[source], node_lookup[target]
        if graph.has_edge(a, b):
            raise ValueError(f"links[{i}] duplicate physical link")
        graph.add_edge(a, b)
        links.append(Link(
            a, b,
            _number(entry, "capacity_mbps", f"links[{i}]", positive=True),
            _number(entry, "latency_base_ms", f"links[{i}]"),
            _number(entry, "latency_per_mbps_ms", f"links[{i}]"),
        ))
    if not nx.is_connected(graph):
        raise ValueError("Network must be connected")
    seen_services: set[str] = set()
    seen_demands: set[str] = set()
    services: list[Service] = []
    demands: list[Demand] = []
    for i, entry in enumerate(_items(raw, "services")):
        services.append(Service(_name(entry, seen_services, f"services[{i}]"),
                                _number(entry, "cpu", f"services[{i}]", positive=True)))
        for j, d in enumerate(_items(entry, "demands")):
            context = f"services[{i}].demands[{j}]"
            origin = d.get("origin")
            if origin not in node_lookup:
                raise ValueError(f"{context}.origin unknown node: {origin!r}")
            lo = _number(d, "min_mbps", context, integer=True)
            hi = _number(d, "max_mbps", context, integer=True)
            if lo > hi:
                raise ValueError(f"{context}: min_mbps exceeds max_mbps")
            demands.append(Demand(
                _name(d, seen_demands, context), i, node_lookup[origin], lo, hi,
                _number(d, "max_latency_ms", context, positive=True),
            ))
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description must be a string")
    return Problem(problem_id, nodes, tuple(links), tuple(services), tuple(demands), description)
