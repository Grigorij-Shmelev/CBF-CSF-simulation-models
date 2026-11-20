"""Old command `python run_vanek_lumbar_infusion.py`: still works. The code is in cbfcsf/checks/vanek.py."""
import runpy

import _repo  # noqa: F401  (repository root on sys.path)

runpy.run_module("cbfcsf.checks.vanek", run_name="__main__")
