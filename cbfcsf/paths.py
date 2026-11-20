"""Locations in the repository."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # repository root
ZBYSEK_DATA = os.path.join(ROOT, "zbysek_matlab_code", "data", "inputs_OK.mat")   # local only (gitignored)


def out_dir(model):
    """outputs/<model>/, created if missing: every figure and table of a model or experiment goes there."""
    d = os.path.join(ROOT, "outputs", model)
    os.makedirs(d, exist_ok=True)
    return d


def out_path(model, name):
    return os.path.join(out_dir(model), name)
