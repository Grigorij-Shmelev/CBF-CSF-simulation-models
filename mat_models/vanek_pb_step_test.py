"""Old command `python vanek_pb_step_test.py`: still works. The script is experiments/vanek_pb_step_test.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.vanek_pb_step_test", run_name="__main__")
