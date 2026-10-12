"""Benchmark result schema and solver result extraction."""

RESULT_COLUMNS = (
    "instance",
    "solver",
    "users",
    "neighbors",
    "groups",
    "score",
    "runtime",
    "buildtime",
    "status",
    "feasible",
    "seed",
    "error",
)


def result_row(args, name: str, users: int, groups: int, solver) -> dict:
    return {
        "instance": args.instance,
        "solver": solver.solver,
        "users": users,
        "neighbors": args.neighbors,
        "groups": groups,
        "score": solver.objective if solver.objective is not None else "",
        "runtime": solver.runtime,
        "buildtime": getattr(solver, "build_time", 0.0) or 0.0,
        "status": solver.status.value,
        "feasible": solver.validation.feasible,
        "seed": solver.seed,
        "error": "; ".join(v.message for v in solver.validation.violations),
    }


def failed_row(args, name: str, users: int, groups: int, runtime: float, error: str = "") -> dict:
    return {
        "instance": args.instance,
        "solver": name,
        "users": users,
        "neighbors": args.neighbors,
        "groups": groups,
        "score": "",
        "runtime": runtime,
        "buildtime": "",
        "status": "error",
        "feasible": False,
        "seed": args.seed,
        "error": error,
    }


def get_result_informations(solver_result: str) -> list:
    if hasattr(solver_result, "solver"):

        def numeric(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        return [
            numeric(getattr(solver_result.solver, "time", None)),
            numeric(getattr(solver_result.problem, "lower_bound", None)),
            numeric(getattr(solver_result.problem, "upper_bound", None)),
            str(solver_result.solver.termination_condition),
        ]
    results = [0, 0, 0, 0]
    results[0] = get_info_from_results(solver_result, "Time: ")
    results[1] = get_info_from_results(solver_result, "Lower bound: ")
    results[2] = get_info_from_results(solver_result, "Upper bound: ")
    results[3] = get_info_from_results(solver_result, "Termination condition: ")
    return results


def get_info_from_results(result_type: str, info_string: str):
    """
    Get information about results from Pyomo result info string.
    """
    text = str(result_type)
    start = text.lower().find(info_string.lower())
    if start < 0:
        raise ValueError(f"Missing solver result field: {info_string}")
    return text[start + len(info_string) :].splitlines()[0].strip()
