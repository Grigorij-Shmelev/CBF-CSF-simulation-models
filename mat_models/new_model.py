"""Old command of the new model: `python new_model.py [flags]` from mat_models/ still works (same flags).
The code is in the cbfcsf package."""
import _repo  # noqa: F401  (repository root on sys.path)
from cbfcsf.models.new_model import *  # noqa: F401,F403
from cbfcsf.models.new_model import main

if __name__ == "__main__":
    import sys
    raise SystemExit(1 if main(sys.argv[1:]) else 0)
