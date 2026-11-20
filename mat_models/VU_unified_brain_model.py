"""Old command `python VU_unified_brain_model.py`: still works. The script is experiments/VU_unified_brain_model.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.VU_unified_brain_model", run_name="__main__")
