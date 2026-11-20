"""Old command of the new model: `python new_model.py [flags]` from mat_models/ still works (same flags), and
`from new_model import ...` still finds every name. The code is in the cbfcsf package:
models/new_model.py (Inputs, NewModel), checks/new.py (checks, main), crosscheck.py, fit_lit.py, plots.py."""
import _repo  # noqa: F401  (repository root on sys.path)
from cbfcsf.models.new_model import *  # noqa: F401,F403
from cbfcsf.common.report import *  # noqa: F401,F403
from cbfcsf.common.signals import *  # noqa: F401,F403
from cbfcsf.crosscheck import *  # noqa: F401,F403
from cbfcsf.fit_lit import *  # noqa: F401,F403
from cbfcsf.plots import *  # noqa: F401,F403
from cbfcsf.checks.new import *  # noqa: F401,F403
from cbfcsf.checks.new import main

if __name__ == "__main__":
    import sys
    raise SystemExit(1 if main(sys.argv[1:]) else 0)
