"""Run registered solvers and persist benchmark results."""

import csv
import time
from pathlib import Path

from mecbench.api import make_star_instance, make_tree_instance, solve
from mecbench.benchmarks.metrics import RESULT_COLUMNS, failed_row, result_row
from mecbench.utils.solver_registry import SOLVER_FILE_ALIASES, list_solvers
from mecbench.utils.solver_registry import make_solver as make_solver

INSTANCES = ("tree", "star")
SOLVERS = (*list_solvers(), *SOLVER_FILE_ALIASES)


def ensure_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as f:
            header = next(csv.reader(f), [])
        if header != list(RESULT_COLUMNS):
            raise ValueError(
                f"CSV {path} has an older or incompatible schema. Choose a new --output file."
            )
        return
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writeheader()


def append_row(path: Path, row: dict):
    ensure_csv(path)
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writerow(row)


def should_skip(name: str, problem) -> bool:
    return name == "ilp_arc" and problem.number_of_service > 150


def make_problem(args, size: int):
    if args.instance == "tree":
        return make_tree_instance(size, args.neighbors, seed=args.seed), size, 1

    problem = make_star_instance(args.groups, size, args.neighbors, seed=args.seed)
    return problem, args.groups * size, args.groups


def run_solver(name: str, problem, ilp_time_limit: float, seed: int = 0):
    from mecbench.utils.solver_registry import REGISTRY, canonical_name

    time_limit = ilp_time_limit if REGISTRY[canonical_name(name)].mathematical else None
    return solve(problem, solver=name, seed=seed, time_limit=time_limit)


def run_benchmarks(args):
    ensure_csv(args.output)
    failures = 0

    for size in range(args.start_users, args.stop_users, args.step_users):
        problem, users, groups = make_problem(args, size)

        for name in args.solvers:
            if should_skip(name, problem):
                print(f"Skipping {name} for {args.instance} size {users}: instance too large.")
                continue

            start = time.time()
            try:
                solver = run_solver(name, problem, args.ilp_time_limit, args.seed)
                row = result_row(args, name, users, groups, solver)
                if not solver.validation.feasible:
                    failures += 1
                msg = f"status={solver.status.value} score={solver.objective} runtime={solver.runtime:.3f}s"
                if solver.validation.violations:
                    msg += f": {solver.validation.violations[0].message}"
            except Exception as exc:
                failures += 1
                runtime = time.time() - start
                row = failed_row(args, name, users, groups, runtime, str(exc))
                msg = f"failed after {runtime:.3f}s: {type(exc).__name__}: {exc}"

            append_row(args.output, row)
            print(f"{args.instance} users={users} solver={name} {msg}")

    if not args.no_plots:
        try:
            from mecbench.benchmarks.reporting import plot_results
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Plots require: python -m pip install 'mec-placement-benchmarks[benchmarks]'."
            ) from exc
        plot_results(args)
    return 1 if failures else 0
