"""Stable K-shortest-hop simple path catalogue with directed arcs."""
from __future__ import annotations

from functools import lru_cache

import networkx as nx

from .problem import Problem


class PathCatalog:
    def __init__(self, problem: Problem, k: int = 3):
        if k < 1:
            raise ValueError("Number of paths must be >= 1")
        self.problem = problem
        self.k = k
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(problem.nodes)))
        self.graph.add_edges_from((e.source, e.target) for e in problem.links)
        self.arc_index = {arc: i for i, arc in enumerate(problem.arcs)}

    @lru_cache(maxsize=None)
    def candidates(self, start: int, end: int) -> tuple[tuple[int, ...], ...]:
        if start == end:
            return ((start,),) * self.k
        # Ordering is fixed by the topology, NOT traffic or latency coefficients.
        paths = [tuple(p) for _, p in zip(range(self.k), nx.shortest_simple_paths(self.graph, start, end))]
        if not paths:
            raise ValueError(f"No path between {start} and {end}")
        return tuple(paths + [paths[-1]] * (self.k - len(paths)))

    def route(self, start: int, end: int, index: int) -> tuple[int, ...]:
        if not 0 <= index < self.k:
            raise ValueError(f"Path index {index} outside [0, {self.k})")
        return self.candidates(start, end)[index]
