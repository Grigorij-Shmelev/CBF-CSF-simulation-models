"""Old command `python compare_all_models.py`: still works. The script is experiments/compare_all_models.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.compare_all_models", run_name="__main__")
