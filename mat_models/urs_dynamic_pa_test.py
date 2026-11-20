"""Old command `python urs_dynamic_pa_test.py`: still works. The script is experiments/urs_dynamic_pa_test.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.urs_dynamic_pa_test", run_name="__main__")
