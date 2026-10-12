from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mecbench.domain.loads import residual_arrays, traffic_arrays
from mecbench.domain.problem import ProblemInstance
from mecbench.domain.problem_arrays import problem_arrays

ORIGIN_COLOR = "tab:red"
SERVER_COLOR = "tab:blue"
ORIGIN_SERVER_COLOR = "tab:purple"
DEFAULT_NODE_COLOR = "tab:gray"
USED_EDGE_COLOR = "tab:orange"
UNUSED_EDGE_COLOR = "lightgray"


def plot_instance_solution(
    problem_instance: ProblemInstance,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    q: np.ndarray | None = None,
    *,
    ax: plt.Axes | None = None,
    layout: str = "auto",
    show: bool = True,
    save_path: str | Path | None = None,
    with_labels: bool | None = None,
    show_node_usage: bool | None = None,
    show_edge_usage: bool | None = None,
    show_solution_summary: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot any problem instance from canonical `(x, y, q)` variables."""
    x, y, q = _resolve_solution(problem_instance, x=x, y=y, q=q)
    graph = problem_instance.network_graph.nx_graph
    data = problem_arrays(problem_instance)
    pos = _make_layout(graph, problem_instance, layout)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.figure

    node_colors = _node_colors(problem_instance, y)
    node_sizes = _node_sizes(graph, y)
    loads = residual_arrays(problem_instance, x, y, q)
    edge_indices = [data.arc_index[edge] for edge in graph.edges]
    capacity_usage = loads["edge_load"][edge_indices]
    edge_widths = _edge_widths(capacity_usage)
    edge_colors = [USED_EDGE_COLOR if width > 0.5 else UNUSED_EDGE_COLOR for width in edge_widths]
    resource_usage = loads["node_load"]

    ax.axis("off")
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=graph.is_directed(),
        arrowsize=10,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.8,
        edgecolors="white",
    )

    if with_labels is None:
        with_labels = graph.number_of_nodes() <= 100
    if show_node_usage is None:
        show_node_usage = graph.number_of_nodes() <= 100
    if show_edge_usage is None:
        show_edge_usage = graph.number_of_edges() <= 100

    if with_labels:
        nx.draw_networkx_labels(
            graph,
            pos,
            ax=ax,
            labels=_node_labels(graph, resource_usage, show_node_usage),
            font_size=8,
        )
    if show_edge_usage:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            ax=ax,
            edge_labels=_edge_labels(graph, capacity_usage),
            font_size=7,
            label_pos=0.55,
        )

    ax.set_title(problem_instance.problem_name)
    if show_solution_summary:
        ax.text(
            1.02,
            1.0,
            _solution_summary(problem_instance, x, q),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
        )
    fig.tight_layout()
    if show_solution_summary:
        fig.subplots_adjust(right=0.72)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def _resolve_solution(
    problem_instance: ProblemInstance,
    *,
    x: np.ndarray | None,
    y: np.ndarray | None,
    q: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x is None:
        x = getattr(problem_instance, "x_variable", None)
    if y is None:
        y = getattr(problem_instance, "y_variable", None)
    if q is None:
        q = getattr(problem_instance, "q_variable", None)

    if x is None or y is None or q is None:
        raise ValueError("A solution must provide x, y, and q variables.")

    x = np.asarray(x)
    y = np.asarray(y)
    q = np.asarray(q)
    _validate_solution_shapes(problem_instance, x, y, q)
    return x, y, q


def _validate_solution_shapes(
    problem_instance: ProblemInstance,
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
) -> None:
    graph = problem_instance.network_graph.nx_graph
    expected_x = (problem_instance.total_number_of_demands, graph.number_of_edges())
    expected_y = (problem_instance.number_of_service, graph.number_of_nodes())
    expected_q = (problem_instance.total_number_of_demands,)

    if x.shape != expected_x:
        raise ValueError(f"x has shape {x.shape}, expected {expected_x}.")
    if y.shape != expected_y:
        raise ValueError(f"y has shape {y.shape}, expected {expected_y}.")
    if q.shape != expected_q:
        raise ValueError(f"q has shape {q.shape}, expected {expected_q}.")


def _node_colors(problem_instance: ProblemInstance, y: np.ndarray) -> list[str]:
    graph = problem_instance.network_graph.nx_graph
    origins = set(problem_instance.origins)
    used_servers = set(np.nonzero(np.sum(y, axis=0))[0])

    colors = []
    for node in graph.nodes:
        is_origin = node in origins
        is_server = node in used_servers
        if is_origin and is_server:
            colors.append(ORIGIN_SERVER_COLOR)
        elif is_origin:
            colors.append(ORIGIN_COLOR)
        elif is_server:
            colors.append(SERVER_COLOR)
        else:
            colors.append(DEFAULT_NODE_COLOR)
    return colors


def _node_sizes(graph: nx.Graph, y: np.ndarray) -> np.ndarray:
    services_per_node = np.sum(y, axis=0)
    return 120 + 180 * services_per_node


def _edge_widths(edge_loads: np.ndarray) -> np.ndarray:
    if edge_loads.size == 0 or edge_loads.max(initial=0) <= 0:
        return np.full(edge_loads.size, 0.5)
    return 0.5 + 4.0 * edge_loads / edge_loads.max()


def _node_labels(
    graph: nx.Graph,
    resource_usage: np.ndarray,
    show_node_usage: bool,
) -> dict:
    if not show_node_usage:
        return {node: str(node) for node in graph.nodes}

    labels = {}
    for node in graph.nodes:
        available = graph.nodes[node]["available_resources"]
        used = resource_usage[node]
        labels[node] = f"{node}\nR {used:g}/{available:g}"
    return labels


def _edge_labels(
    graph: nx.Graph,
    capacity_usage: np.ndarray,
) -> dict:
    labels = {}
    for edge_index, edge in enumerate(graph.edges):
        capacity = graph.edges[edge]["capacity"]
        used = capacity_usage[edge_index]
        labels[edge] = f"C {used:g}/{capacity:g}"
    return labels


def _solution_summary(
    problem_instance: ProblemInstance,
    x: np.ndarray,
    q: np.ndarray,
) -> str:
    lines = ["solution"]
    data = problem_arrays(problem_instance)
    latencies = traffic_arrays(data, x, q)["demand_latency"]
    for service_index in range(data.n_services):
        lines.append(f"S{service_index}")
        for demand_index in np.flatnonzero(data.demand_to_service == service_index):
            latency = latencies[demand_index]
            max_latency = problem_instance.latencies[demand_index]
            max_quantity = problem_instance.max_flows[demand_index]
            quantity = q[demand_index]
            ratio = quantity / max_quantity if max_quantity else 0
            lines.append(
                f" d{demand_index}: L {latency:g}/{max_latency:g} "
                f"q {quantity:g}/{max_quantity:g} ({ratio:.0%})"
            )

    return "\n".join(lines)


def _make_layout(
    graph: nx.Graph,
    problem_instance: ProblemInstance,
    layout: str,
) -> dict:
    if layout == "auto":
        layout = "shell" if _looks_like_layered_large_instance(graph) else "spring"

    if layout == "shell":
        shells = _large_instance_shells(graph)
        if shells is not None:
            return nx.shell_layout(graph, shells)
        return nx.shell_layout(graph)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "spring":
        return nx.spring_layout(graph, seed=0)

    raise ValueError(f"Unsupported layout: {layout}.")


def _looks_like_layered_large_instance(graph: nx.Graph) -> bool:
    return graph.number_of_nodes() > 1000 and graph.number_of_nodes() % 31 == 0


def _large_instance_shells(graph: nx.Graph) -> list[range] | None:
    if graph.number_of_nodes() % 31 != 0:
        return None

    n = graph.number_of_nodes() // 31
    return [range(n), range(n, 11 * n), range(11 * n, 31 * n)]


def plot_results(args):
    import pandas as pd

    df = pd.read_csv(args.output)
    mask = (df["instance"] == args.instance) & (df["neighbors"] == args.neighbors)
    if args.instance == "star":
        mask &= df["groups"] == args.groups
    df = df[mask].copy()
    if df.empty:
        return

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df = df.groupby(["solver", "users"], as_index=False)[["score", "runtime"]].mean()
    df = df.sort_values(["solver", "users"])

    for metric, ylabel, filename in (
        ("score", "objective", f"{args.instance}_objective_vs_graph_size.pdf"),
        ("runtime", "solving time (s)", f"{args.instance}_runtime_vs_graph_size.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for solver, g in df.dropna(subset=[metric]).groupby("solver"):
            ax.plot(g["users"], g[metric], marker="o", linewidth=1.6, label=solver)

        ax.set_xlabel("graph size (number of users)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.figures_dir / filename, bbox_inches="tight")
        plt.close(fig)
