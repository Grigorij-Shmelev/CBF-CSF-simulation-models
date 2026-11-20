"""Old command `python urs_steal_test.py`: still works. The code is in cbfcsf/checks/ursino.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("cbfcsf.checks.ursino", run_name="__main__")
