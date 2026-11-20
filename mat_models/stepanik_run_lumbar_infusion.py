"""Old command `python stepanik_run_lumbar_infusion.py`: still works. The code is in cbfcsf/checks/stepanik.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("cbfcsf.checks.stepanik", run_name="__main__")
