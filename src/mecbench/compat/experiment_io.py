"""Historical experiment interfaces; new code uses mecbench.solve and the CLI."""

import importlib
import os
from pathlib import Path

import pandas as pd

from mecbench.benchmarks.metrics import get_info_from_results as get_info_from_results
from mecbench.benchmarks.metrics import get_result_informations as get_result_informations
from mecbench.compat.pickle_io import pickle_load as pickle_load
from mecbench.compat.pickle_io import pickle_save as pickle_save
from mecbench.domain.problem import ProblemInstance
from mecbench.utils.solver_base import GenericSolver
from mecbench.utils.solver_registry import SOLVER_FILE_ALIASES as SOLVER_FILE_ALIASES
from mecbench.utils.solver_registry import SOLVER_MODULES, canonical_name

ARTIFACTS_PATH = os.path.join(os.getcwd(), "artifacts")
INSTANCES_PATH = os.path.join(ARTIFACTS_PATH, "instances")
RESULTS_PATH = os.path.join(ARTIFACTS_PATH, "data")
SOLVER_PATH = str(Path(__file__).resolve().parents[1] / "solvers")
CURRENT_FOLDER = os.getcwd()


def folder_management():
    topology_path = os.path.join(CURRENT_FOLDER, "data", "topologies")
    assert os.path.isdir(topology_path), "There is no topology data folder."
    assert os.listdir(topology_path), "There are no topology files."
    os.makedirs(INSTANCES_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)


def reset(verbose: bool = True):
    if verbose:
        print("Deleting instances...")
    instance_path = INSTANCES_PATH
    os.makedirs(instance_path, exist_ok=True)
    instance_list = os.listdir(instance_path)
    for current_instance in instance_list:
        current_instance_path = os.path.join(instance_path, current_instance)
        os.remove(current_instance_path)


def get_instance_list() -> list[ProblemInstance]:
    all_instances = []

    instance_path = INSTANCES_PATH
    instances_list = os.listdir(instance_path)
    for instance in instances_list:
        current_instance_path = os.path.join(instance_path, instance)
        all_instances.append(pickle_load(current_instance_path))

    return all_instances


def get_solver_list():
    return [importlib.import_module(module_name).Solver for module_name in SOLVER_MODULES.values()]


def experience_index() -> int:
    """
    Returns the number of the current experience.
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    file_list = os.listdir(RESULTS_PATH)
    if len(file_list) == 0:
        return 0
    else:
        return max(int(file_name.split("_")[0]) for file_name in file_list) + 1


def create_dataframe_from_dict(input_dict: dict):
    """
    Create a pandas.DataFrame from a dict {(row, col) : value}
    """
    # Extract unique row and column names
    rows = set()
    columns = set()
    for key in input_dict:
        row_name, column_name = key
        rows.add(row_name)
        columns.add(column_name)

    # Create an empty DataFrame with the extracted row and column names
    df = pd.DataFrame(index=rows, columns=columns)

    # Fill the DataFrame using the dictionary values
    for key, value in input_dict.items():
        row_name, column_name = key
        df.loc[row_name, column_name] = value

    return df


def import_instance_from_file(instance_name: str) -> ProblemInstance:
    instance_name = instance_name + ".pkl" if not instance_name.endswith(".pkl") else instance_name
    instance_path = os.path.join(INSTANCES_PATH, instance_name)
    return pickle_load(instance_path)


def import_solver_from_file(solver_name: str, instance: ProblemInstance) -> GenericSolver:
    solver_name = canonical_name(solver_name.removesuffix(".py"))
    module = importlib.import_module(SOLVER_MODULES[solver_name])
    return module.Solver(instance)


def solve(
    instance_name: str,
    solver_name: str,
    repeat: int = 1,
    check_solution: bool = False,
    initial_variable=None,
    verbose: bool = False,
    instance: ProblemInstance = None,
    solver: GenericSolver = None,
) -> tuple[ProblemInstance, GenericSolver]:
    """
    Allows to solve an instance by its name and by the solver name.
    """
    if instance is None:
        instance = import_instance_from_file(instance_name)

    if solver is None:
        solver = import_solver_from_file(solver_name, instance)

    if verbose:
        print("Start solving...")

    if repeat == 1:
        if initial_variable is not None:
            solver.run(initial_variable)
        else:
            solver.run()
    else:
        runtime = 0.0
        for _ in range(repeat):
            if initial_variable is not None:
                solver.run(initial_variable)
            else:
                solver.run()
            runtime += solver.runtime
        solver.runtime = runtime / repeat

    if verbose:
        print(f"Score : {solver.score}")
        print(f"Runtime : {solver.runtime}")

    if check_solution:
        instance._set_arc_solution(solver.x_var, solver.y_var, solver.q_var)
        instance.check_current_solution()
    return instance, solver
