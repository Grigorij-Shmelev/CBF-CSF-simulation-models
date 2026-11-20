"""Old command `python vanek_driven_by_ursino.py`: still works. The script is experiments/vanek_driven_by_ursino.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.vanek_driven_by_ursino", run_name="__main__")
