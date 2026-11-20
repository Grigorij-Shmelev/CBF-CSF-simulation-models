"""Old command `python vanek_icp_sensitivity_tests.py`: still works. The script is experiments/vanek_icp_sensitivity_tests.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.vanek_icp_sensitivity_tests", run_name="__main__")
