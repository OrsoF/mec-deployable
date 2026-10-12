import importlib
import inspect
import io
from contextlib import redirect_stderr, redirect_stdout

from mecbench.instances.synthetic import create_tree_problem_instance

SOLVERS = [
    ("heuristic", "mecbench.solvers.heuristic"),
    ("heuristic_unit_step", "mecbench.solvers.heuristic_unit_step"),
    ("fast_heuristic", "mecbench.solvers.fast_heuristic"),
    ("ilp_arc", "mecbench.solvers.ilp_arc"),
    ("ilp_arc_gap_5pct", "mecbench.solvers.ilp_arc_gap_5pct"),
    ("ilp_arc_heuristic_placement", "mecbench.solvers.ilp_arc_heuristic_placement"),
    ("ilp_arc_heuristic_warmstart", "mecbench.solvers.ilp_arc_heuristic_warmstart"),
    ("ilp_path", "mecbench.solvers.ilp_path"),
    ("milp_arc", "mecbench.solvers.milp_arc"),
    ("milp_arc_full", "mecbench.solvers.milp_arc_full"),
    ("milp_path", "mecbench.solvers.milp_path"),
    ("lp_arc", "mecbench.solvers.lp_arc"),
    ("lp_path", "mecbench.solvers.lp_path"),
    ("lp_path_latency_tightened", "mecbench.solvers.lp_path_latency_tightened"),
]


def _run(solver):
    kwargs = {}
    sig = inspect.signature(solver.run)

    if "solving_time_limit" in sig.parameters:
        kwargs["solving_time_limit"] = 5
    if "thread_limit" in sig.parameters:
        kwargs["thread_limit"] = 1

    solver.run(**kwargs)


def _float(x):
    return None if x is None else float(x)


def run_all_solvers():
    instance = create_tree_problem_instance(n_users=2, neighbors=0)
    rows = []

    for name, module_name in SOLVERS:
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                module = importlib.import_module(module_name)
                solver = module.Solver(instance)
                _run(solver)
            rows.append(
                {
                    "solver": name,
                    "status": "ok",
                    "score": _float(solver.score),
                    "runtime": _float(solver.runtime),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "solver": name,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    return rows


if __name__ == "__main__":
    for row in run_all_solvers():
        print(row)
