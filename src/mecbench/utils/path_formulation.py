from itertools import islice

import networkx as nx
import pyomo.environ as pyo


def enumerate_candidate_paths(G, origins, limit):
    paths_by_demand = [
        [
            path
            for target in G.nodes
            for path in islice(
                nx.algorithms.simple_paths.shortest_simple_paths(G, origins[d], target),
                limit,
            )
        ]
        for d in range(len(origins))
    ]
    paths = [path for demand_paths in paths_by_demand for path in demand_paths]
    path_to_demand = [d for d, demand_paths in enumerate(paths_by_demand) for _ in demand_paths]
    path_start = [0]
    for demand_paths in paths_by_demand:
        path_start.append(path_start[-1] + len(demand_paths))
    return paths_by_demand, paths, path_to_demand, path_start


def add_path_params(
    model,
    paths_by_demand,
    paths,
    path_to_demand,
    path_start,
    arc_dic,
    arc_list,
    max_flows,
    latencies,
) -> None:
    model.D_path = pyo.Param(
        model.C,
        initialize={p: path_to_demand[p] for p in range(len(paths))},
        mutable=False,
        within=pyo.NonNegativeIntegers,
    )
    model.Pdmax_path = pyo.Param(
        model.C,
        initialize={p: max_flows[path_to_demand[p]] for p in range(len(paths))},
        mutable=False,
        within=pyo.NonNegativeIntegers,
    )
    model.latency_path = pyo.Param(
        model.C,
        initialize={p: latencies[path_to_demand[p]] for p in range(len(paths))},
        mutable=False,
        within=pyo.NonNegativeReals,
    )
    model.delta = pyo.Param(
        model.C,
        model.A,
        initialize={
            (p, a): arc_list[a]
            in [(paths[p][i], paths[p][i + 1]) for i in range(len(paths[p]) - 1)]
            for a in model.A
            for p in model.C
        },
        mutable=False,
        within=pyo.Binary,
    )
    model.all_shortest_paths_arc = pyo.Param(
        model.C,
        initialize={
            p: [arc_dic[(paths[p][i], paths[p][i + 1])] for i in range(len(paths[p]) - 1)]
            for p in model.C
        },
        mutable=False,
        within=pyo.Any,
    )
    model.path_demand = pyo.Param(
        range(len(paths_by_demand) + 1),
        initialize={d: path_start[d] for d in range(len(paths_by_demand) + 1)},
        mutable=False,
        within=pyo.NonNegativeIntegers,
    )
    model.ind_destination = pyo.Param(
        model.C,
        model.U,
        initialize={(p, u): paths[p][-1] == u for p in model.C for u in model.U},
    )


def add_path_constraints(model, m, n) -> None:
    total_flow = sum(pyo.value(model.Pdmax[d]) for d in model.D)
    latency_bounds = {
        a: pyo.value(model.beta[a])
        + pyo.value(model.alpha[a]) * min(pyo.value(model.arc_capacity[a]), total_flow)
        for a in model.A
    }
    model.valeurs_p = pyo.Constraint(
        model.D,
        rule=lambda model, d: pyo.inequality(model.Pdmin[d], model.p[d], model.Pdmax[d]),
    )
    model.valeurs_P = pyo.Constraint(
        model.C,
        rule=lambda model, p: model.P[p] <= model.Pdmax_path[p],
    )
    model.valeurs_L = pyo.Constraint(
        model.C,
        model.A,
        rule=lambda model, p, a: model.L[p, a] <= model.latency_path[p],
    )
    model.resources_constraint = pyo.Constraint(
        model.U,
        rule=lambda model, u: sum(model.resources[s] * model.y[s, u] for s in model.S)
        <= model.node_resources[u],
    )
    model.capacity = pyo.Constraint(
        model.A,
        rule=lambda model, a: model.nul[0]
        + sum(float(model.delta[p, a]) * model.P[p] for p in model.C)
        <= model.arc_capacity[a],
    )
    model.latency = pyo.Constraint(
        model.D,
        rule=lambda model, d: sum(
            model.L[p, a]
            for p in range(model.path_demand[d], model.path_demand[d + 1])
            for a in model.all_shortest_paths_arc[p]
        )
        <= model.latencies[d],
    )
    model.realization = pyo.Constraint(
        model.S,
        rule=lambda model, s: sum(model.y[s, u] for u in model.U) == 1,
    )
    model.path_selection = pyo.Constraint(
        model.D,
        rule=lambda model, d: sum(
            model.l[p] for p in range(model.path_demand[d], model.path_demand[d + 1])
        ) == 1,
    )
    model.reach_server = pyo.Constraint(
        model.D,
        model.U,
        rule=lambda model, d, u: model.y[model.serv_number[d], u]
        == sum(
            float(model.ind_destination[p, u]) * model.l[p]
            for p in range(model.path_demand[d], model.path_demand[d + 1])
        ),
    )
    model.flow_rule_1 = pyo.Constraint(
        model.C,
        rule=lambda model, p: model.P[p] <= model.p[model.D_path[p]],
    )
    model.flow_rule_2 = pyo.Constraint(
        model.C,
        rule=lambda model, p: model.P[p] <= m * model.l[p],
    )
    model.flow_rule_3 = pyo.Constraint(
        model.C,
        rule=lambda model, p: model.P[p] >= model.p[model.D_path[p]] - m * (1 - model.l[p]),
    )
    model.latency_rule_1 = pyo.Constraint(
        model.C,
        model.A,
        rule=lambda model, p, a: model.L[p, a]
        <= model.alpha[a] * sum(float(model.delta[pp, a]) * model.P[pp] for pp in model.C)
        + model.beta[a],
    )
    model.latency_rule_2 = pyo.Constraint(
        model.C,
        model.A,
        rule=lambda model, p, a: model.L[p, a]
        <= latency_bounds[a] * model.delta[p, a] * model.l[p],
    )
    model.latency_rule_3 = pyo.Constraint(
        model.C,
        model.A,
        rule=lambda model, p, a: model.L[p, a]
        >= model.alpha[a] * sum(float(model.delta[pp, a]) * model.P[pp] for pp in model.C)
        + model.beta[a]
        - latency_bounds[a] * (1 - model.delta[p, a] * model.l[p]),
    )
