"""Evolutionary search over discrete placement, routing and quantity candidates."""

import time

import numpy as np

from mecbench.domain.problem import ProblemInstance
from mecbench.utils.heuristic_solvers import clip_solution, repair_solution
from mecbench.utils.solver_base import GenericSolver
from mecbench.utils.solver_workspace import SolvingInstance


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        max_pcc: int = 10,
        mu: int = 8,
        lambda_: int = 32,
        generations: int = 100,
        quantity_steps: int | None = None,
        seed: int = 0,
    ):
        super().__init__(problem_instance)
        if max_pcc < 1:
            raise ValueError("max_pcc must be positive.")
        if mu < 1 or lambda_ < 1 or generations < 1:
            raise ValueError("mu, lambda_, and generations must be positive.")
        if quantity_steps is not None and quantity_steps < 0:
            raise ValueError("quantity_steps must be nonnegative.")

        self.max_pcc = max_pcc
        self.mu = mu
        self.lambda_ = lambda_
        self.generations = generations
        self.quantity_steps = quantity_steps
        self.seed = seed
        self.solver_name = "discrete_evolutionary"

    def run(self):
        start = time.time()
        self._set_solution(None)
        self.placement: list[int] | None = None
        self.routing: list[int] | None = None
        self.quantities: list[float] | None = None
        self.score = self.runtime = self.termination_condition = None
        self.build_time = 0.0
        self.rng = np.random.default_rng(self.seed)
        self.sol_ins = SolvingInstance(self.problem_instance)

        base = self._initial_solution()
        pop = [base] if self.sol_ins.is_feasible(*base) else []
        for _ in range(self.mu * self.lambda_):
            if len(pop) >= self.mu:
                break
            child = self._repair(*self._mutate(*base))
            if child is not None:
                pop.append(child)

        for _ in range(self.generations):
            children = []
            for _ in range(self.lambda_):
                parent = pop[self.rng.integers(len(pop))] if pop else base
                child = self._repair(*self._mutate(*parent))
                if child is not None:
                    children.append(child)

            pool = pop + children
            pool.sort(key=lambda z: sum(z[2]), reverse=True)
            pop = pool[: self.mu]

        if pop:
            self.placement, self.routing, self.quantities = max(
                pop,
                key=lambda z: sum(z[2]),
            )
            self._set_compact_solution(
                self.placement,
                self.routing,
                self.quantities,
                path_resolver=self.sol_ins.compute_nth_shortest_path,
            )
            self.score = sum(self.quantities)
        self.runtime = time.time() - start
        self.build_time = 0.0
        return self

    def _initial_solution(self) -> tuple[list[int], list[int], list[int]]:
        placement, routing, quantities = self.sol_ins.basic_placement_and_routing()
        placement, routing, quantities = self._clip(placement, routing, quantities)
        if not self.sol_ins.is_feasible(placement, routing, quantities):
            return placement, routing, quantities
        return self._maximize_quantities(placement, routing, quantities)

    def _mutate(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        p, r, q = placement.copy(), routing.copy(), quantities.copy()
        S = self.data.n_services
        D = self.data.n_demands
        n = self.graph.number_of_nodes()

        for _ in range(self.rng.integers(1, 4)):
            op = self.rng.integers(3)
            if op == 0:
                p[self.rng.integers(S)] = self.rng.integers(n)
            elif op == 1:
                r[self.rng.integers(D)] = self.rng.integers(self.max_pcc)
            else:
                d = self.rng.integers(D)
                width = int(max(1, self.data.max_flows[d] // 10))
                q[d] += self.rng.integers(-width, width + 1)

        return p, r, q

    def _repair(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]] | None:
        repaired = repair_solution(
            self.problem_instance,
            self.graph,
            self.sol_ins,
            self.max_pcc,
            placement,
            routing,
            quantities,
        )
        if repaired is None:
            return None
        return self._maximize_quantities(*repaired)

    def _maximize_quantities(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        q = quantities.copy()
        D = self.data.n_demands
        steps = 4 * D if self.quantity_steps is None else self.quantity_steps

        for _ in range(steps):
            d = self.rng.integers(D)
            rem = self.sol_ins._get_remaining_quantity_per_demand(
                placement,
                routing,
                q,
                d,
            )
            inc = int(max(0, rem))
            if inc:
                q_new = q.copy()
                q_new[d] += inc
                if self.sol_ins.is_feasible(placement, routing, q_new):
                    q = q_new

        return placement, routing, q

    def _clip(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        return clip_solution(
            self.problem_instance,
            self.graph,
            self.max_pcc,
            placement,
            routing,
            quantities,
        )
