"""Old command `python urs_occlusion_stiffness_test.py`: still works. The script is experiments/urs_occlusion_stiffness_test.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.urs_occlusion_stiffness_test", run_name="__main__")
