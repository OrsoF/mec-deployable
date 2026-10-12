import time

import numpy as np
import pyomo.environ as pyo

from mecbench.domain.problem import ProblemInstance
from mecbench.utils.arc_formulation import add_arc_constraints
from mecbench.utils.arc_pyomo import add_arc_parameters
from mecbench.utils.ilp_solvers import arc_model_arrays
from mecbench.utils.solver_base import GenericSolver


class Solver(GenericSolver):
    def __init__(self, problem_instance: ProblemInstance):
        super().__init__(problem_instance)
        self.solver_name = "LP_arc"

    def _build_model(self):
        self.model = pyo.ConcreteModel()

    def _build_sets(self):
        self.model.D = pyo.Set(initialize=range(self.data.n_demands))
        self.model.S = pyo.Set(initialize=range(self.data.n_services))
        self.model.U = pyo.Set(initialize=range(self.data.n_nodes))
        self.model.A = pyo.Set(initialize=range(self.data.n_arcs))

    def _build_variables(self):
        self.model.x = pyo.Var(
            self.model.D, self.model.A, within=pyo.NonNegativeReals, bounds=(0, 1)
        )

        self.model.y = pyo.Var(
            self.model.S, self.model.U, within=pyo.NonNegativeReals, bounds=(0, 1)
        )
        self.model.p = pyo.Var(self.model.D, within=pyo.NonNegativeReals)
        self.model.P = pyo.Var(self.model.D, self.model.A, within=pyo.NonNegativeReals)
        self.model.L = pyo.Var(self.model.D, self.model.A, within=pyo.NonNegativeReals)

    def _build_params(self):
        add_arc_parameters(
            self.model,
            self.data,
            self.graph,
            self.model.D,
            self.model.S,
            self.model.U,
            self.model.A,
        )

    def _build_constraints(self):
        add_arc_constraints(
            self.model,
            self.model.D,
            self.model.A,
            self.model.U,
            self.model.S,
            self.model.x,
        )

    def _build_objective(self):
        self.model.obj = pyo.Objective(sense=pyo.maximize, rule=lambda model: sum(model.p.values()))

    def _build_whole_problem(self):
        self._build_model()
        self._build_sets()
        self._build_variables()
        self._build_params()
        self._build_constraints()
        self._build_objective()

    def run(
        self,
        options: dict | None = None,
    ):
        start_time = time.time()

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
            *arc_model_arrays(
                self.model.x,
                self.model.y,
                self.model.p,
                self.data.n_demands,
                self.data.n_arcs,
                self.data.n_services,
                self.data.n_nodes,
            )
        )
        self.runtime = time.time() - start_time
        self.score = float(np.sum(self.q_var))
        self.build_time = 0.0
