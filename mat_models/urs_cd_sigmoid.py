"""Old command `python urs_cd_sigmoid.py`: still works. The script is experiments/urs_cd_sigmoid.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("experiments.urs_cd_sigmoid", run_name="__main__")
