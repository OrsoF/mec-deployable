"""Pickle helpers used by historical experiment scripts."""

import pickle as pkl


def pickle_save(object_instance: object, file_name: str):
    with open(file_name, "wb") as handle:
        pkl.dump(object_instance, handle, protocol=pkl.HIGHEST_PROTOCOL)


def pickle_load(file_name):
    with open(file_name, "rb") as handle:
        python_variable = pkl.load(handle)
    return python_variable
