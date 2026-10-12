"""Local placement/routing search with continuous quantity increases."""

import time
from numbers import Integral

import networkx as nx
import numpy as np

from mecbench.domain.contracts import FloatArray, IntArray
from mecbench.domain.problem import ProblemInstance
from mecbench.utils.solver_base import GenericSolver
from mecbench.utils.solver_workspace import SolverWorkspace


class Solver(GenericSolver):
    _quantity_tolerance = 1e-9
    _sensitivity_cache_limit = 256

    def __init__(
        self,
        problem_instance: ProblemInstance,
        limit_shortest_path: int = 4,
        candidate_nodes: int = 8,
        max_iter: int = 60,
        patience: int = 8,
        seed: int = 0,
    ):
        super().__init__(problem_instance)
        for name, value, minimum in (
            ("limit_shortest_path", limit_shortest_path, 1),
            ("candidate_nodes", candidate_nodes, 1),
            ("max_iter", max_iter, 1),
            ("patience", patience, 1),
            ("seed", seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.solver_name = "fast_heuristic"
        self.limit_shortest_path = int(limit_shortest_path)
        self.candidate_nodes = int(candidate_nodes)
        self.max_iter = int(max_iter)
        self.patience = int(patience)
        self.seed = int(seed)

        self.solving_instance = SolverWorkspace(problem_instance, seed=self.seed)

        self._dist = None
        self._service_nodes = None

    def run(self):
        start = time.time()
        self._set_solution(None)
        self.placement: list[int] | None = None
        self.routing: list[int] | None = None
        self.quantities: list[float] | None = None
        self.score = self.runtime = self.termination_condition = None
        self.build_time = 0.0
        self.rng = np.random.default_rng(self.seed)
        self._dist = dict(nx.all_pairs_shortest_path_length(nx.Graph(self.graph)))
        self._service_nodes = self._candidate_nodes()

        p = self._initial_placement()
        r = np.zeros(self.data.n_demands, dtype=np.int64)
        current = (p, r, self.data.min_flows.copy())
        best = self._repair_lift(*current)
        if best is not None:
            current = best
        best_score = None if best is None else float(best[2].sum())
        upper = float(self.data.max_flows.sum())
        no_progress = 0
        self.visited_objectives = [best_score]

        for _ in range(self.max_iter):
            if best_score is not None and best_score >= upper:
                break
            cand = self._best_neighbor(*current, minimum_score=-np.inf if best is None else None)
            if cand is None:
                cand = self._perturb(*current)

            no_progress += 1
            if cand is not None:
                # Only _repair_lift's reference-validated candidates can be incumbents.
                current = cand
                score = float(cand[2].sum())
                if best_score is None or score > best_score:
                    best = (cand[0].copy(), cand[1].copy(), cand[2].copy())
                    best_score = score
                    no_progress = 0

            self.visited_objectives.append(best_score)
            if no_progress >= self.patience:
                break

        if best is not None:
            self.placement = best[0].astype(int).tolist()
            self.routing = best[1].astype(int).tolist()
            self.quantities = best[2].tolist()
            self._set_compact_solution(
                self.placement,
                self.routing,
                self.quantities,
                path_resolver=self.solving_instance.compute_nth_shortest_path,
            )
            self.score = best_score
        self.runtime = time.time() - start
        self.build_time = 0.0
        return self

    def _candidate_nodes(self) -> list[IntArray]:
        top = np.argsort(-self.data.node_resources)[: self.candidate_nodes]
        nodes = []
        for s in range(self.data.n_services):
            d = np.flatnonzero(self.data.demand_to_service == s)
            origins = np.unique(self.data.origins[d])
            score = np.zeros(self.data.n_nodes)
            for u in origins:
                du = self._dist[int(u)]
                score += np.fromiter(
                    (du.get(v, self.data.n_nodes) for v in range(self.data.n_nodes)),
                    float,
                    self.data.n_nodes,
                )
            close = np.argsort(score)[: self.candidate_nodes]
            cand = np.unique(np.concatenate((origins, top, close)))
            cand = cand[self.data.node_resources[cand] >= self.data.service_resources[s]]
            nodes.append(cand.astype(np.int64))
        return nodes

    def _initial_placement(self) -> IntArray:
        p = np.empty(self.data.n_services, dtype=np.int64)
        rem = self.data.node_resources.copy()
        order = np.argsort(-self.data.service_resources)
        for s in order:
            d = np.flatnonzero(self.data.demand_to_service == s)
            best, best_score = None, np.inf
            for v in self._service_nodes[s]:
                if rem[v] < self.data.service_resources[s]:
                    continue
                dist = sum(
                    self.data.max_flows[j]
                    * self._dist[int(self.data.origins[j])].get(int(v), self.data.n_nodes)
                    for j in d
                )
                score = dist - 1e-6 * rem[v]
                if score < best_score:
                    best, best_score = int(v), score
            if best is None:
                best = int(np.argmax(rem))
            p[s] = best
            rem[best] -= self.data.service_resources[s]
        return p

    def _feasible(self, p, r, q) -> bool:
        return self.solving_instance.is_feasible(p, r, q)

    def _lift(self, p, r, q) -> FloatArray | None:
        q = np.minimum(np.maximum(q, self.data.min_flows), self.data.max_flows).astype(float)
        if not self._feasible(p, r, q):
            if np.array_equal(q, self.data.min_flows):
                return None
            q = self.data.min_flows.copy()
            if not self._feasible(p, r, q):
                return None
        initial = q.copy()
        changed = False

        paths = self.solving_instance.compact_path_indices(p, r)
        incidence = np.zeros((self.data.n_demands, self.data.n_arcs))
        for d, idx in enumerate(paths):
            incidence[d, idx] = 1
        # Entry (i, d) is the latency increase for demand i when q[d]
        # increases by one, including every arc shared by the two routes.
        # Cache the full response only for small problems. Large problems keep
        # one demand's response vector at a time instead of a D-by-D matrix.
        latency_sensitivity = (
            (incidence * self.data.latency_slopes) @ incidence.T
            if self.data.n_demands <= self._sensitivity_cache_limit
            else None
        )

        while True:
            _, edge_load, _ = self.solving_instance.compact_loads(p, r, q, paths=paths)
            latency_slack = self.data.latency_limits - incidence @ (
                self.data.latency_offsets + self.data.latency_slopes * edge_load
            )
            room = np.zeros(self.data.n_demands)
            for d, idx in enumerate(paths):
                flow_room = self.data.max_flows[d] - q[d]
                if flow_room <= self._quantity_tolerance:
                    continue
                cap_room = (
                    np.inf
                    if len(idx) == 0
                    else np.min(self.data.arc_capacities[idx] - edge_load[idx])
                )
                response = (
                    latency_sensitivity[:, d]
                    if latency_sensitivity is not None
                    else incidence[:, idx] @ self.data.latency_slopes[idx]
                )
                affected = response > 0
                lat_room = (
                    np.min(latency_slack[affected] / response[affected])
                    if np.any(affected)
                    else np.inf
                )
                room[d] = max(0.0, min(flow_room, cap_room, lat_room))

            d = int(np.argmax(room))
            delta = float(room[d])
            if delta <= self._quantity_tolerance:
                break
            previous = q[d]
            q[d] += delta
            if q[d] == previous:
                break
            changed = True
        # Analytical bounds preserve feasibility for fixed routes. Check once
        # after lifting and retain the validated starting point on numerical drift.
        return q if not changed or self._feasible(p, r, q) else initial

    def _repair_lift(self, p, r, q) -> tuple[IntArray, IntArray, FloatArray] | None:
        p = np.asarray(p, dtype=np.int64).copy()
        r = np.clip(np.asarray(r, dtype=np.int64), 0, self.limit_shortest_path - 1)
        q = self._lift(p, r, q)
        if q is None:
            return None
        return p, r, q

    def _best_neighbor(self, p, r, q, *, minimum_score: float | None = None):
        best_score = float(q.sum()) if minimum_score is None else minimum_score

        for s in self.rng.permutation(self.data.n_services)[: min(self.data.n_services, 8)]:
            old = p[s]
            nodes = self.rng.permutation(self._service_nodes[s])[
                : min(len(self._service_nodes[s]), 6)
            ]
            for v in nodes:
                if v == old:
                    continue
                pp = p.copy()
                pp[s] = v
                cand = self._repair_lift(pp, r, self.data.min_flows)
                if cand is None:
                    continue
                score = float(cand[2].sum())
                if score > best_score:
                    return cand

        for d in self.rng.permutation(self.data.n_demands)[: min(self.data.n_demands, 16)]:
            old = r[d]
            for k in range(self.limit_shortest_path):
                if k == old:
                    continue
                rr = r.copy()
                rr[d] = k
                cand = self._repair_lift(p, rr, q)
                if cand is None:
                    continue
                score = float(cand[2].sum())
                if score > best_score:
                    return cand

        return None

    def _perturb(self, p, r, q):
        pp, rr = p.copy(), r.copy()
        if self.rng.random() < 0.5:
            s = int(self.rng.integers(self.data.n_services))
            cand = self._service_nodes[s]
            if len(cand):
                pp[s] = int(self.rng.choice(cand))
        else:
            d = int(self.rng.integers(self.data.n_demands))
            rr[d] = int(self.rng.integers(self.limit_shortest_path))

        return self._repair_lift(pp, rr, self.data.min_flows)
