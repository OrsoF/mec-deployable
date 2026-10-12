"""Command-line entry point for MEC placement benchmarks."""

import argparse
import math
import sys
from pathlib import Path

from mecbench.benchmarks.runner import INSTANCES, SOLVERS, run_benchmarks
from mecbench.utils.solver_registry import make_solver as make_solver


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run SPP benchmark experiments.")
    parser.add_argument("--instance", choices=INSTANCES, default="tree")
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=["fast_heuristic"])
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--groups", type=positive_int, default=4)
    parser.add_argument("--neighbors", type=nonnegative_int, default=1)
    parser.add_argument("--start-users", type=positive_int)
    parser.add_argument("--stop-users", type=positive_int)
    parser.add_argument("--step-users", type=positive_int)
    parser.add_argument("--ilp-time-limit", type=positive_float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--no-plots", action="store_true", default=True)
    plots.add_argument(
        "--plots", action="store_false", dest="no_plots", help="requires the benchmarks extra"
    )
    args = set_instance_defaults(parser.parse_args(argv))
    if args.stop_users <= args.start_users:
        parser.error("--stop-users must be greater than --start-users (exclusive upper bound)")
    return args


def set_instance_defaults(args):
    if args.start_users is None:
        args.start_users = 20 if args.instance == "tree" else 15
    if args.stop_users is None:
        args.stop_users = args.start_users + 1
    if args.step_users is None:
        args.step_users = 1
    if args.output is None:
        args.output = Path("artifacts") / "data" / f"exp_{args.instance}.csv"
    if args.figures_dir is None:
        args.figures_dir = Path("artifacts") / "figures" / args.instance
    return args


def main(argv=None):
    args = set_instance_defaults(parse_args(argv))
    try:
        return run_benchmarks(args)
    except (ValueError, OSError, ImportError) as error:
        print(f"mecbench: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
