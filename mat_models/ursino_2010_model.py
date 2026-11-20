"""Old path of the Ursino & Giannessi 2010 model. The code is in cbfcsf/models/ursino_2010.py;
`python ursino_2010_model.py` still works."""
import _repo  # noqa: F401  (repository root on sys.path)
from cbfcsf.models.ursino_2010 import *  # noqa: F401,F403
from cbfcsf.models.ursino_2010 import main

if __name__ == "__main__":
    main()
