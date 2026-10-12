import time
from itertools import islice

import networkx as nx
import numpy as np
import pyomo.environ as pyo

from mecbench.domain.problem import ProblemInstance
from mecbench.utils.ilp_solvers import path_model_arrays
from mecbench.utils.solver_base import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        problem_instance: ProblemInstance,
        maximum_path_number: int = 1,
        eps: float = 0.0001,
    ):
        super().__init__(problem_instance)
        self.limit = maximum_path_number
        self.solver_name = "LP_path_strong_latency"
        self.eps = eps

    def _model_parameters(self):
        self.shortest_paths = [
            [
                elem
                for target in self.graph.nodes
                for elem in islice(
                    nx.algorithms.simple_paths.shortest_simple_paths(
                        self.graph, self.data.origins[d], target
                    ),
                    self.limit,
                )
            ]
            for d in range(self.data.n_demands)
        ]
        self.all_shortest_paths = [elem for path_list in self.shortest_paths for elem in path_list]
        self.C = sum(len(path) for path in self.shortest_paths)
        self.D_path_number = [
            i for i, path_list in enumerate(self.shortest_paths) for _ in path_list
        ]
        self.D_path = [i for i, path_list in enumerate(self.shortest_paths) for _ in path_list]
        self.m = max(self.data.max_flows)
        self.n = max(self.data.latency_limits)

        self.shortest_paths = [
            [
                list(
                    islice(
                        nx.algorithms.simple_paths.shortest_simple_paths(
                            self.graph, self.data.origins[d], target
                        ),
                        self.limit,
                    )
                )
                for target in self.graph.nodes
            ]
            for d in range(self.data.n_demands)
        ]
        self.paths = [path[0] for l_path in self.shortest_paths for path in l_path]
        self.P = len(self.paths)
        self.p_to_d = [
            i for i in range(len(self.shortest_paths)) for j in range(len(self.shortest_paths[i]))
        ]

        self.pd = [0] + [
            sum(len(elem) for elem in self.shortest_paths[:i])
            for i in range(1, len(self.shortest_paths) + 1)
        ]

        self.maximum_required_flow = max(self.data.max_flows)
        self.maximum_possible_latency = max(self.data.latency_slopes) * sum(
            self.data.max_flows
        ) + max(self.data.latency_offsets)

    def path_to_demand(self, p):
        return self.p_to_d[p]

    def delta(self, c, a):
        return int(
            self.data.arcs[a]
            in [(self.paths[c][i], self.paths[c][i + 1]) for i in range(len(self.paths[c]) - 1)]
        )

    def _model_creation(self):
        self.model = pyo.ConcreteModel()

    def _set_creation(self):
        self.model.A = pyo.Set(initialize=range(self.data.n_arcs))
        self.model.P = pyo.Set(initialize=range(self.P))
        self.model.E = pyo.Set(initialize=range(self.data.n_nodes))
        self.model.S = pyo.Set(initialize=range(self.data.n_services))
        self.model.D = pyo.Set(initialize=range(self.data.n_demands))

    def _variable_creation(self):
        self.model.L = pyo.Var(self.model.P, self.model.A, within=pyo.NonNegativeReals)
        self.model.Q = pyo.Var(self.model.P, within=pyo.NonNegativeReals)
        self.model.q = pyo.Var(self.model.D, within=pyo.NonNegativeReals)
        self.model.y = pyo.Var(
            self.model.S, self.model.E, within=pyo.NonNegativeReals, bounds=(0, 1)
        )
        self.model.l = pyo.Var(self.model.P, within=pyo.NonNegativeReals, bounds=(0, 1))
        self.model.z = pyo.Var(self.model.P, within=pyo.Binary)
        self.model.nul = pyo.Var(range(1))
        self.model.nul[0].fix(0)

    def _set_constraints(self):
        self.model.val_q = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: pyo.inequality(
                self.data.min_flows[d], model.q[d], self.data.max_flows[d]
            ),
        )
        self.model.res = pyo.Constraint(
            self.model.E,
            rule=lambda model, e: sum(
                model.y[s, e] * self.data.service_resources[s] for s in model.S
            )
            <= self.data.node_resources[e],
        )

        self.model.capa = pyo.Constraint(
            self.model.A,
            rule=lambda model, a: model.nul[0] + sum(self.delta(p, a) * model.Q[p] for p in model.P)
            <= self.data.arc_capacities[a],
        )

        self.model.lat = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: sum(
                model.L[p, a]
                for p in range(self.pd[d], self.pd[d + 1])
                for a in model.A
                if self.delta(p, a)
            ) <= self.data.latency_limits[d],
        )

        self.model.ind1 = pyo.Constraint(
            self.model.P, rule=lambda model, p: model.l[p] <= model.z[p]
        )

        self.model.ind2 = pyo.Constraint(
            self.model.P, rule=lambda model, p: model.z[p] <= 1 + model.l[p] - self.eps
        )

        self.model.res_reach = pyo.Constraint(
            self.model.S, rule=lambda model, s: sum(model.y[s, e] for e in model.E) == 1
        )
        self.model.path_selection = pyo.Constraint(
            self.model.D,
            rule=lambda model, d: sum(
                model.l[p] for p in range(self.pd[d], self.pd[d + 1])
            ) == 1,
        )

        self.model.reach_serv = pyo.Constraint(self.model.D, self.model.E, rule=self.reach_serv)

        self.model.qual1 = pyo.Constraint(
            self.model.P,
            rule=lambda model, p: model.Q[p] <= model.q[self.path_to_demand(p)],
        )

        self.model.qual2 = pyo.Constraint(
            self.model.P,
            rule=lambda model, p: model.Q[p]
            >= model.q[self.path_to_demand(p)] - self.maximum_required_flow * (1 - model.l[p]),
        )

        self.model.qual3 = pyo.Constraint(
            self.model.P,
            rule=lambda model, p: model.Q[p] <= self.maximum_required_flow * model.l[p],
        )

        self.model.lat1 = pyo.Constraint(
            self.model.P,
            self.model.A,
            rule=lambda model, p, a: model.L[p, a]
            <= self.data.latency_slopes[a] * sum(self.delta(pp, a) * model.Q[pp] for pp in model.P)
            + self.data.latency_offsets[a],
        )

        self.model.lat2 = pyo.Constraint(
            self.model.P,
            self.model.A,
            rule=lambda model, p, a: model.L[p, a]
            >= self.data.latency_slopes[a] * sum(self.delta(pp, a) * model.Q[pp] for pp in model.P)
            + self.data.latency_offsets[a]
            - self.maximum_possible_latency * (1 - self.delta(p, a) * model.z[p]),
        )

        self.model.lat3 = pyo.Constraint(
            self.model.P,
            self.model.A,
            rule=lambda model, p, a: model.L[p, a]
            <= self.maximum_possible_latency * self.delta(p, a) * model.z[p],
        )

    def gamma(self, p, e):
        return int(self.paths[p][-1] == e)

    def _set_objective(self):
        self.model.obj = pyo.Objective(sense=pyo.maximize, rule=lambda model: sum(model.q.values()))

    def _set_params(self):
        self.model.serv_number = pyo.Param(
            self.model.D,
            initialize={d: int(self.data.demand_to_service[d]) for d in range(self.data.n_demands)},
            mutable=False,
            within=pyo.NonNegativeIntegers,
        )

    def reach_serv(self, model, d, e):
        return model.y[model.serv_number[d], e] == sum(
            self.gamma(p, e) * model.l[p] for p in range(self.pd[d], self.pd[d + 1])
        )

    def run(
        self,
        options: dict | None = None,
    ):
        start_time = time.time()

        self._model_parameters()
        self._model_creation()
        self._set_creation()
        self._variable_creation()
        self._set_params()
        self._set_constraints()
        self._set_objective()

        opt = pyo.SolverFactory("gurobi")
        opt.options.update(options or {})
        solver_result = opt.solve(self.model)
        self.termination_condition = str(solver_result.solver.termination_condition)

        self._set_output_variables(
            *path_model_arrays(
                self.model.l,
                self.model.y,
                self.model.q,
                self.paths,
                self.p_to_d,
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
