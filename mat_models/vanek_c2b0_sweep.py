"""Old command `python vanek_c2b0_sweep.py`: still works. The script is experiments/vanek_c2b0_sweep.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.vanek_c2b0_sweep", run_name="__main__")
