import pyomo.environ as pyo


def add_arc_constraints(model, D, A, U, S, x) -> None:
    model.valeurs_p = pyo.Constraint(
        D,
        rule=lambda model, d: pyo.inequality(model.Pdmin[d], model.p[d], model.Pdmax[d]),
    )
    model.valeurs_P = pyo.Constraint(
        D,
        A,
        rule=lambda model, d, a: model.P[d, a] <= model.Pdmax[d],
    )
    model.conservation_constraint = pyo.Constraint(
        D,
        U,
        rule=lambda model, d, u: sum(x[d, arc] for arc in model.out_arcs[u])
        - sum(x[d, arc] for arc in model.inc_arcs[u])
        == model.ind_u_origin[u, d] - model.y[model.serv_number[d], u],
    )
    model.resources_constraint = pyo.Constraint(
        U,
        rule=lambda model, u: sum(model.y[s, u] * model.resources[s] for s in S)
        <= model.node_resources[u],
    )
    model.capacity_constraint = pyo.Constraint(
        A,
        rule=lambda model, a: sum(model.P[d, a] for d in D) <= model.arc_capacity[a],
    )
    model.latency_arc_1 = pyo.Constraint(
        D,
        rule=lambda model, d: sum(model.L[d, a] for a in A) <= model.latencies[d],
    )
    model.latency_arc_2 = pyo.Constraint(
        D,
        A,
        rule=lambda model, d, a: model.L[d, a]
        <= model.alpha[a] * sum(model.P[dp, a] for dp in D) + model.beta[a],
    )
    model.latency_arc_3 = pyo.Constraint(
        D,
        A,
        rule=lambda model, d, a: model.L[d, a]
        >= model.alpha[a] * sum(model.P[dp, a] for dp in D)
        + model.beta[a]
        - model.N * (1 - x[d, a]),
    )
    model.latency_arc_4 = pyo.Constraint(
        D,
        A,
        rule=lambda model, d, a: model.L[d, a] <= model.N * x[d, a],
    )
    model.power_1 = pyo.Constraint(
        D,
        A,
        rule=lambda model, d, a: model.P[d, a] <= model.p[d],
    )
    model.power_2 = pyo.Constraint(
        D,
        A,
        rule=lambda model, d, a: model.P[d, a] >= model.p[d] - model.M * (1 - x[d, a]),
    )
    model.power_3 = pyo.Constraint(
        D,
        A,
        rule=lambda model, d, a: model.P[d, a] <= model.M * x[d, a],
    )
