"""Shared Pyomo parameters for the arc formulations."""

from typing import Any

import networkx as nx
import pyomo.environ as pyo

from mecbench.domain.graph_utils import get_encoded_incoming_arc, get_encoded_outgoing_arc
from mecbench.domain.problem_arrays import ProblemArrays


def add_arc_parameters(
    model: Any, data: ProblemArrays, graph: nx.DiGraph, D: Any, S: Any, U: Any, A: Any
) -> None:
    """Keep each formulation's existing index sets, names and parameter domains."""
    for name, index, values, domain in (
        ("origins", D, data.origins, pyo.NonNegativeIntegers),
        ("Pdmin", D, data.min_flows, pyo.NonNegativeIntegers),
        ("Pdmax", D, data.max_flows, pyo.NonNegativeIntegers),
        ("latencies", D, data.latency_limits, pyo.NonNegativeReals),
        ("resources", S, data.service_resources, pyo.NonNegativeIntegers),
        ("alpha", A, data.latency_slopes, pyo.NonNegativeReals),
        ("beta", A, data.latency_offsets, pyo.NonNegativeReals),
    ):
        setattr(
            model,
            name,
            pyo.Param(
                index, initialize={i: values[i] for i in index}, mutable=False, within=domain
            ),
        )
    model.serv_number = pyo.Param(
        D,
        initialize={d: int(data.demand_to_service[d]) for d in D},
        mutable=False,
        within=pyo.NonNegativeIntegers,
    )
    model.out_arcs = pyo.Param(
        U,
        initialize={u: get_encoded_outgoing_arc(graph, data.arc_index, u) for u in U},
        mutable=False,
        within=pyo.Any,
    )
    model.inc_arcs = pyo.Param(
        U,
        initialize={u: get_encoded_incoming_arc(graph, data.arc_index, u) for u in U},
        mutable=False,
        within=pyo.Any,
    )
    model.ind_u_origin = pyo.Param(
        U,
        D,
        initialize={(u, d): int(u == model.origins[d]) for u in U for d in D},
        mutable=False,
        within=pyo.Binary,
    )
    model.node_resources = pyo.Param(
        U,
        initialize={u: data.node_resources[u] for u in U},
        mutable=False,
        within=pyo.NonNegativeIntegers,
    )
    model.arc_capacity = pyo.Param(
        A,
        initialize={a: data.arc_capacities[a] for a in A},
        mutable=False,
        within=pyo.NonNegativeIntegers,
    )
    model.N = pyo.Param(
        # Capacity bounds total traffic, including on arcs unused by a demand.
        initialize=max(
            data.latency_offsets
            + data.latency_slopes * data.arc_capacities.clip(max=data.max_flows.sum())
        ),
        mutable=False,
        within=pyo.NonNegativeReals,
    )
    model.M = pyo.Param(
        initialize=max(data.max_flows), mutable=False, within=pyo.NonNegativeIntegers
    )
