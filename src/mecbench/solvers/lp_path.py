import time

import numpy as np
import pyomo.environ as pyo

from mecbench.domain.problem import ProblemInstance
from mecbench.utils.ilp_solvers import path_model_arrays
from mecbench.utils.path_formulation import (
    add_path_constraints,
    add_path_params,
    enumerate_candidate_paths,
)
from mecbench.utils.solver_base import GenericSolver


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance, maximum_path_number: int = 1):
        super().__init__(problem_instance)
        self.limit = maximum_path_number
        self.solver_name = "LP_path"

    def _model_parameters(self):
        (
            self.shortest_paths,
            self.all_shortest_paths,
            self.D_path_number,
            self.path_start,
        ) = enumerate_candidate_paths(self.graph, self.data.origins, self.limit)
        self.C = len(self.all_shortest_paths)
        self.D_path = self.D_path_number
        self.m = max(self.data.max_flows)
        self.n = max(self.data.latency_limits)

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.data.n_demands))
        self.model.U = pyo.Set(initialize=range(self.data.n_nodes))
        self.model.S = pyo.Set(initialize=range(self.data.n_services))
        self.model.A = pyo.Set(initialize=range(self.data.n_arcs))
        self.model.C = pyo.Set(initialize=range(self.C))

    def _build_variables(self):
        self.model.l = pyo.Var(self.model.C, within=pyo.NonNegativeReals, bounds=(0, 1))
        self.model.y = pyo.Var(
            self.model.S, self.model.U, within=pyo.NonNegativeReals, bounds=(0, 1)
        )
        self.model.p = pyo.Var(self.model.D, within=pyo.NonNegativeReals)
        self.model.P = pyo.Var(self.model.C, within=pyo.NonNegativeReals)
        self.model.L = pyo.Var(self.model.C, self.model.A, within=pyo.NonNegativeReals)
        self.model.nul = pyo.Var(range(1), within=pyo.Binary)
        self.model.nul[0].fix(0)

    def _build_params(self):
        self.model.origins = pyo.Param(
            self.model.D,
            initialize={d: self.data.origins[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmin = pyo.Param(
            self.model.D,
            initialize={d: self.data.min_flows[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.Pdmax = pyo.Param(
            self.model.D,
            initialize={d: self.data.max_flows[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.latencies = pyo.Param(
            self.model.D,
            initialize={d: self.data.latency_limits[d] for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeReals,
        )
        self.model.resources = pyo.Param(
            self.model.S,
            initialize={s: self.data.service_resources[s] for s in range(self.data.n_services)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.node_resources = pyo.Param(
            self.model.U,
            initialize={u: self.data.node_resources[u] for u in range(self.data.n_nodes)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.alpha = pyo.Param(
            self.model.A,
            initialize={edge: self.data.latency_slopes[edge] for edge in self.model.A},
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.beta = pyo.Param(
            self.model.A,
            initialize={edge: self.data.latency_offsets[edge] for edge in self.model.A},
            mutable=False,
            within=pyo.NonNegativeReals,
        )

        self.model.arc_capacity = pyo.Param(
            self.model.A,
            initialize={a: self.data.arc_capacities[a] for a in range(self.data.n_arcs)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        self.model.serv_number = pyo.Param(
            self.model.D,
            initialize={d: int(self.data.demand_to_service[d]) for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )
        add_path_params(
            self.model,
            self.shortest_paths,
            self.all_shortest_paths,
            self.D_path_number,
            self.path_start,
            self.data.arc_index,
            self.data.arcs,
            self.data.max_flows,
            self.data.latency_limits,
        )

    def _build_constraints(self):
        add_path_constraints(self.model, self.m, self.n)

    def _build_objective(self):
        self.model.obj = pyo.Objective(sense=pyo.maximize, rule=lambda model: sum(model.p.values()))

    def run(
        self,
        options: dict | None = None,
    ):
        start_time = time.time()

        self._model_parameters()
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        opt = pyo.SolverFactory("gurobi")
        opt.options.update(options or {})
        solver_result = opt.solve(self.model)
        self.termination_condition = str(solver_result.solver.termination_condition)

        self._set_output_variables(
            *path_model_arrays(
                self.model.l,
                self.model.y,
                self.model.p,
                self.all_shortest_paths,
                self.D_path_number,
                self.data.arc_index,
                self.data.n_demands,
                self.data.n_arcs,
                self.data.n_services,
                self.data.n_nodes,
            )
        )
        self.runtime = time.time() - start_time
        self.score = float(np.sum(self.q_var))
        self.build_time = 0.0
