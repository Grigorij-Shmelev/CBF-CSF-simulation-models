"""Single entry point of the repository (run from the repository root):

    python run.py --list                    models, flags and experiments
    python run.py new [flags]               the new 0D model (no flag = the default run)
    python run.py ursino|vanek|stepanik [flags]
    python run.py exp <name> [args]         an analysis of experiments/

python run.py <model> --help describes every flag. Outputs go to outputs/<model>/.
"""
from cbfcsf.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
