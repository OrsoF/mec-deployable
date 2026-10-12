import time

import numpy as np
import pyomo.environ as pyo

from mecbench.benchmarks.metrics import get_result_informations
from mecbench.domain.problem import ProblemInstance
from mecbench.solvers.heuristic import Solver as Heursolv
from mecbench.utils.arc_formulation import add_arc_constraints
from mecbench.utils.arc_pyomo import add_arc_parameters
from mecbench.utils.ilp_solvers import (
    arc_model_arrays,
    configure_gurobi,
    objective_value,
)
from mecbench.utils.solver_base import GenericSolver


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        super().__init__(problem_instance)
        self.solver_name = "ILP_arc"

    def _build_model(self):
        self.model = pyo.ConcreteModel()
        self._has_initial_value = False

    def _build_sets(self):
        self.model.total_number_of_demands = pyo.Set(initialize=range(self.data.n_demands))
        self.model.number_of_services = pyo.Set(initialize=range(self.data.n_services))
        self.model.node_number = pyo.Set(initialize=range(self.data.n_nodes))
        self.model.arc_number = pyo.Set(initialize=range(self.data.n_arcs))

    def _build_variables(self):
        self.model.demand_arc_transit = pyo.Var(
            self.model.total_number_of_demands, self.model.arc_number, within=pyo.Binary
        )

        self.model.y = pyo.Var(
            self.model.number_of_services, self.model.node_number, within=pyo.Binary
        )

        self.model.p = pyo.Var(self.model.total_number_of_demands, within=pyo.NonNegativeIntegers)

        self.model.P = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

        self.model.L = pyo.Var(
            self.model.total_number_of_demands,
            self.model.arc_number,
            within=pyo.NonNegativeIntegers,
        )

    def _build_params(self):
        add_arc_parameters(
            self.model,
            self.data,
            self.graph,
            self.model.total_number_of_demands,
            self.model.number_of_services,
            self.model.node_number,
            self.model.arc_number,
        )

    def _build_constraints(self):
        add_arc_constraints(
            self.model,
            self.model.total_number_of_demands,
            self.model.arc_number,
            self.model.node_number,
            self.model.number_of_services,
            self.model.demand_arc_transit,
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(sense=pyo.maximize, rule=lambda model: sum(model.p.values()))

    def _set_initial_value(self, initial_value: tuple):
        x_var, y_var, q_var = initial_value
        for demand_index in range(self.data.n_demands):
            self.model.p[demand_index].value = q_var[demand_index]
            for arc_index in range(self.data.n_arcs):
                self.model.demand_arc_transit[demand_index, arc_index].value = x_var[
                    demand_index, arc_index
                ]
        for service_index in range(self.data.n_services):
            for node_index in range(self.data.n_nodes):
                self.model.y[service_index, node_index].value = y_var[service_index, node_index]
        self._has_initial_value = True

    def set_heuristic_placement(self):
        cloud_server = np.argmax(self.data.node_resources)

        print(cloud_server)

        for service in range(self.data.n_services):
            self.model.y[service, cloud_server].fix(1)

    def set_heuristic_solution(self, seed=0):
        heuristic_solver = Heursolv(self.problem_instance, seed=seed)
        heuristic_solver.run()
        if heuristic_solver.solution is not None:
            self._set_initial_value(
                (heuristic_solver.x_var, heuristic_solver.y_var, heuristic_solver.q_var)
            )

    def set_rl_solution(self, seed=0):
        from mecbench.solvers.ppo import Solver as RlPPO

        solver = RlPPO(self.problem_instance, seed=seed)
        solver.run()

        if solver.arc_solution is not None:
            self._set_initial_value(tuple(solver.arc_solution))

    def run(
        self,
        initial_value: tuple = None,
        options: dict | None = None,
        heuristic_placement: bool = False,
        heuristic_solution: bool = False,
        rl_solution: bool = False,
    ):
        start_time = time.time()

        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

        if initial_value is not None:
            self._set_initial_value(initial_value)

        if heuristic_placement:
            self.set_heuristic_placement()

        if heuristic_solution:
            self.set_heuristic_solution(seed=(options or {}).get("Seed", 0))

        if rl_solution:
            self.set_rl_solution(seed=(options or {}).get("Seed", 0))

        opt = configure_gurobi(
            pyo.SolverFactory("gurobi"),
            {"Heuristics": 0, "MemLimit": 8, **(options or {})},
        )

        solver_result = opt.solve(
            self.model, **({"warmstart": True} if self._has_initial_value else {})
        )
        self.termination_condition = str(solver_result.solver.termination_condition)
        solver_runtime = get_result_informations(solver_result)[0]

        self._set_output_variables(
            *arc_model_arrays(
                self.model.demand_arc_transit,
                self.model.y,
                self.model.p,
                self.data.n_demands,
                self.data.n_arcs,
                self.data.n_services,
                self.data.n_nodes,
            )
        )

        try:
            self.score = objective_value(self.q_var)
        except (TypeError, ValueError):
            self.score = 0

        self.runtime = time.time() - start_time
        self.build_time = self.runtime - float(solver_runtime)
