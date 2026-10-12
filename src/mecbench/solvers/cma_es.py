"""CMA-ES search over encoded service-placement candidates."""

import time

import numpy as np
from numpy.typing import ArrayLike

from mecbench.domain.contracts import FloatArray, IntArray
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.solution import CompactSolution
from mecbench.domain.validation import ValidationReport, validate
from mecbench.utils.solver_base import GenericSolver
from mecbench.utils.solver_workspace import SolverWorkspace


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        max_pcc: int = 10,
        sigma: float = 2.0,
        max_iter: int = 200,
        max_evals: int | None = None,
        seed: int = 0,
        verbose: bool = False,
    ):
        super().__init__(problem_instance)
        if max_pcc < 1:
            raise ValueError("max_pcc must be positive.")
        self.max_pcc = max_pcc
        self.sigma = sigma
        self.max_iter = max_iter
        self.max_evals = max_evals
        self.seed = seed
        self.verbose = verbose
        self.solver_name = "cma_es"
        self._flow_cap = float(self.data.max_flows.sum()) + 1.0

    def run(self):
        start_time = time.time()
        self._set_solution(None)
        self.placement: IntArray | None = None
        self.routing: IntArray | None = None
        self.quantities: FloatArray | None = None
        self.score = self.runtime = self.termination_condition = None
        self.build_time = 0.0
        self.sol_ins = SolverWorkspace(self.problem_instance, seed=self.seed)
        self.best_score = -np.inf
        self.best_solution: CompactSolution | None = None
        x0 = self.initial_value()
        self.objective(x0)

        import cma

        rng = np.random.default_rng(self.seed)
        lower, upper = self._bounds()
        free = np.flatnonzero(np.asarray(lower) != np.asarray(upper))

        def expand(candidate):
            value = x0.copy()
            value[free] = candidate
            return value

        opts = {
            "bounds": [np.asarray(bound)[free].tolist() for bound in (lower, upper)],
            "maxiter": self.max_iter,
            # CMA treats seed=0 as time-based randomness. Supply a local sampler
            # so every public seed is reproducible without changing NumPy globals.
            "seed": np.nan,
            "randn": lambda *shape: rng.standard_normal(shape),
            "verbose": 0 if self.verbose else -9,
            "verb_disp": 1 if self.verbose else 0,
            "verb_log": 1 if self.verbose else 0,
        }
        if self.max_evals is not None:
            opts["maxfevals"] = self.max_evals

        if free.size:
            es = cma.CMAEvolutionStrategy(x0[free], self.sigma, opts)
            while not es.stop():
                solutions = es.ask()
                es.tell(solutions, [self.objective(expand(x)) for x in solutions])
                if self.verbose:
                    es.logger.add()
                    es.disp()

        if self.best_solution is not None:
            self.placement = self.best_solution.placement
            self.routing = self.best_solution.routing
            self.quantities = self.best_solution.quantities
            self._set_solution(
                self.best_solution,
                path_resolver=self.sol_ins.compute_nth_shortest_path,
            )
            self.score = self.best_score

        self.runtime = time.time() - start_time
        return self

    def objective(self, value: ArrayLike) -> float:
        candidate = CompactSolution(*self.decode(value))
        report = validate(
            self.data, candidate, path_resolver=self.sol_ins.compute_nth_shortest_path
        )
        score = float(candidate.quantities.sum())
        if report.feasible:
            if score > self.best_score:
                self.best_score = score
                self.best_solution = candidate
            return -score

        return self._flow_cap + self._violation_penalty(report) - score / self._flow_cap

    def decode(self, value: ArrayLike) -> tuple[list[int], list[int], list[float]]:
        S = self.data.n_services
        D = self.data.n_demands
        z = np.asarray(value, dtype=float)
        if z.shape != (S + 2 * D,) or not np.all(np.isfinite(z)):
            raise ValueError(
                f"Encoded candidate must be a finite vector of shape ({S + 2 * D},)."
            )

        placement = np.clip(np.rint(z[:S]), 0, self.data.n_nodes - 1).astype(np.int64)
        routing = np.clip(np.rint(z[S : S + D]), 0, self.max_pcc - 1).astype(np.int64)
        quantities = np.clip(
            z[S + D :],
            self.data.min_flows,
            self.data.max_flows,
        )
        return placement.tolist(), routing.tolist(), quantities.tolist()

    def _violation_penalty(self, report: ValidationReport) -> float:
        penalty = 0.0
        for violation in report.violations:
            if (
                violation.constraint not in ("capacity", "resources", "latency")
                or not isinstance(violation.observed, float)
                or not isinstance(violation.limit, float)
            ):
                return self._flow_cap
            penalty += max(0.0, violation.observed - violation.limit)
        return penalty

    def violation(
        self,
        placement: list[int],
        routing: list[int],
        quantities: list[float],
    ) -> float:
        """Compatibility entry point using the reference validator's violations."""
        try:
            report = validate(
                self.data,
                CompactSolution(placement, routing, quantities),
                path_resolver=self.sol_ins.compute_nth_shortest_path,
            )
        except (TypeError, ValueError):
            return self._flow_cap
        return self._violation_penalty(report)

    def initial_value(self, method: str = "heuristic") -> FloatArray:
        if method == "basic":
            candidate = CompactSolution(*self.sol_ins.basic_placement_and_routing())
        elif method == "heuristic":
            from mecbench.solvers.heuristic import Solver as HeuristicSolver

            sol = HeuristicSolver(self.problem_instance, seed=self.seed)
            sol.run()
            if sol.solution is None:
                candidate = CompactSolution(*self.sol_ins.basic_placement_and_routing())
            elif isinstance(sol.solution, CompactSolution):
                candidate = sol.solution
            else:
                raise RuntimeError("The warm-start heuristic must publish a CompactSolution.")
        else:
            raise ValueError("method must be 'basic' or 'heuristic'.")

        value = np.concatenate((candidate.placement, candidate.routing, candidate.quantities))
        return np.concatenate(self.decode(value), dtype=np.float64)

    def _bounds(self) -> list[list[float]]:
        S = self.data.n_services
        D = self.data.n_demands
        n = self.data.n_nodes
        lower = [0] * S + [0] * D + self.data.min_flows.tolist()
        upper = [n - 1] * S + [self.max_pcc - 1] * D + self.data.max_flows.tolist()
        return [lower, upper]
