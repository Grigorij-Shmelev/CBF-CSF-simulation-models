"""Command line of the repository: python run.py --list | python run.py <model> [flags] | python run.py exp <name> [args].

Every old command still works from mat_models/ (thin wrappers); this is the same code behind one entry point.
All figures and tables go to outputs/<model>/.
"""
import argparse
import runpy
import sys

# name: (what it does, arguments, runtime)
EXPERIMENTS = {
    "urs_cd_sigmoid": ("Ursino's compliance sigmoid C_d(x)/C_dn and where C_d crosses zero", "", "3 s"),
    "urs_dynamic_pa_test": ("Ursino driven by a pulsatile Pa(t). Modes: (none) 10-s smoke test, full (400 s), "
                            "check (node pressures), table (1550-s runs -> CSV), profile (from the table), "
                            "export (CSV for Vanek)", "[full|check|table|profile|export]", "6 s to 6 min"),
    "urs_occlusion_stiffness_test": ("ICA occlusion with a normal and a stiff skull (k_E 0.077 vs 0.15)", "", "3 s"),
    "vanek_icp_sensitivity_tests": ("Vanek LIT with high outflow resistance and with stiff veins", "", "1.5 min"),
    "vanek_c2b0_sweep": ("Vanek: the two roles of C2b0 (LIT storage, path of the cardiac pulse)", "", "2.5 min"),
    "vanek_pb_step_test": ("Vanek: lasting step of Pb 14 -> 20 mmHg and the mean ICP", "", "1.5 min"),
    "compare_all_models": ("Ursino steal run, Stepanik and Vanek LIT side by side (README Part 4)", "", "1 min"),
    "vanek_driven_by_ursino": ("Vanek driven by Ursino's CSV (run urs_dynamic_pa_test export first)", "[D100|D80]",
                               "1.5 min"),
    "VU_unified_brain_model": ("Ursino -> Vanek coupling in memory: steal + LIT, CSV for MRI validation", "", "4 s"),
}

NEW_PARTS = (
    ("rest", "T1: 600 s at rest"),
    ("co2", "T5: CO2 challenge 40 -> 60 mmHg (with T5g against the Ursino code)"),
    ("co2-5pct", "C1-C3: 5 %% CO2 (PaCO2 47.5 mmHg)"),
    ("lit", "L1-L6: lumbar infusion test, 1.5 mL/min for 10 min"),
    ("stroke", "S1-S6: left MCA occluded 30 min, then reopened"),
    ("ica", "I1: left ICA occluded 30 min, then reopened"),
    ("pulsatile", "T6 (info): pulsatile arterial pressure"),
    ("cross", "X1-X5: comparison with the Ursino and Vanek codes (runs stroke, lit, co2-5pct, ica)"),
    ("outputs", "O1-O4: CBV in %% and MCA velocity (runs rest, co2-5pct, stroke)"),
)
NEW_STAGES = (
    ("calibrate", "+ Vanek's raw spinal compliances (K1) and the data-file pulse check (K2), about 3 min"),
    ("sensitivity", "+ phi_v sensitivity V1 (spec 7.5), about 7 min"),
    ("params", "+ parameter sensitivity SA (spec 7.8), about 2 min"),
    ("options", "+ venous squeeze and sigmoid width (spec 7.10), about 2 min"),
    ("fit", "+ LIT fitting tool on synthetic data (spec 7.10), about 7 min, 8 processes"),
    ("fit-mismatch", "only: fit when the data come from a modified model (spec 7.13), about 5 min"),
    ("all", "default + calibrate + sensitivity + options + fit + params, about 20 min"),
)
SMALL = {
    "ursino": (("baseline", "baseline and steal summary (Ursino & Giannessi 2010), 3 s"),
               ("steal", "ICA occlusion + hypercapnia: steal phenomenon figure, 3 s"),
               ("aux-check", "node pressures of the circle of Willis checked against derivatives(), 11 s")),
    "vanek": (("baseline", "10-s baseline with synthetic inputs, 3 s"),
              ("lit", "lumbar infusion test 1.5 mL/min for 5 min, R_out, 6-panel figure, 40 s")),
    "stepanik": (("baseline", "10-s baseline with synthetic inputs and 6-panel figure, 4 s"),
                 ("lit", "lumbar infusion test 1.5 mL/min for 5 min, R_out, 6-panel figure, 20 s")),
}


def parser():
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list models, flags and experiments")
    sub = ap.add_subparsers(dest="model")
    p = sub.add_parser("new", help="the new 0D model (spec 7)", description="New model: no flag = the default run "
                       "(all scenarios, cross-checks, figures; about 2 min), as `python new_model.py`. Scenario flags "
                       "run only those parts of the default run. Stage flags add a stage to the full default run.")
    g = p.add_argument_group("parts of the default run")
    for flag, text in NEW_PARTS:
        g.add_argument(f"--{flag}", action="store_true", help=text)
    g = p.add_argument_group("stages")
    for flag, text in NEW_STAGES:
        g.add_argument(f"--{flag}", action="store_true", help=text)
    g.add_argument("--fit-csv", metavar="FILE", help="+ fit a measured LIT (CSV with header: time_s, pressure_mmHg)")
    g.add_argument("--rate", metavar="ML_MIN", help="infusion rate of --fit-csv [mL/min] (default 1.5)")
    g.add_argument("--t-on", metavar="S", help="infusion start of --fit-csv [s] (default 300)")
    g.add_argument("--t-off", metavar="S", help="infusion end of --fit-csv [s] (default 900)")
    g = p.add_argument_group("opt-in scenario (spec 7.17): runs on its own, not with other flags; not part of the "
                             "default run or of --all")
    g.add_argument("--etco2", action="store_true", help="E0-E8: 5 %% CO2 inhaled 3 x 180 s with an end-tidal-shaped "
                   "input (PLACEHOLDER shape), against the current 10-s ramps; tau and air-interval sweeps; about 2 min")
    g.add_argument("--etco2-csv", metavar="FILE", help="the model driven by a measured end-tidal curve (CSV with "
                   "header: time_s, etco2); needs --etco2-unit and --cycles")
    g.add_argument("--etco2-unit", choices=("mmHg", "kPa", "percent"), help="unit of the etco2 column, required with "
                   "--etco2-csv (never guessed). kPa: x 7.50062. percent: F/100 x (P_B - 47) mmHg, standard physiology, "
                   "not from the project sources")
    g.add_argument("--cycles", metavar="T1,T2,...", help="inhalation onsets of --etco2-csv in the CSV's own time [s]")
    g.add_argument("--cycle-length", metavar="S", help="inhalation length of --etco2-csv [s] (default 180)")
    g.add_argument("--barometric", metavar="MMHG", help="barometric pressure P_B for --etco2-unit percent [mmHg] "
                   "(default 760)")
    for model, flags in SMALL.items():
        p = sub.add_parser(model, help=f"checks of the {model} model", description=f"Checks of the {model} model.")
        for flag, text in flags:
            p.add_argument(f"--{flag}", action="store_true", help=text)
        p.add_argument("--all", action="store_true", help="all of the above")
    p = sub.add_parser("exp", help="run an analysis script of experiments/", description="Run experiments/<name>.py "
                       "with the given arguments. --list shows them.")
    p.add_argument("name", choices=sorted(EXPERIMENTS))
    p.add_argument("args", nargs=argparse.REMAINDER)
    return ap


def show_list():
    print("python run.py new [flags]            the new 0D model; no flag = default run (about 2 min)")
    for flag, text in NEW_PARTS + NEW_STAGES:
        print(f"    --{flag:<14}{text.replace('%%', '%')}")
    print(f"    --{'fit-csv FILE':<14}+ fit a measured LIT (--rate, --t-on, --t-off)")
    print(f"    --{'etco2':<14}only: end-tidal-shaped 5 % CO2 input, PLACEHOLDER shape (spec 7.17; not in --all), about 2 min")
    print(f"    --{'etco2-csv FILE':<14} only: a measured end-tidal curve (--etco2-unit, --cycles, --cycle-length, "
          f"--barometric)")
    for model, flags in SMALL.items():
        print(f"python run.py {model} [flags]")
        for flag, text in flags:
            print(f"    --{flag:<14}{text}")
        print(f"    --{'all':<14}all of the above")
    print("python run.py exp <name> [args]      analyses in experiments/")
    for name, (text, args, rt) in EXPERIMENTS.items():
        print(f"    {name + (' ' + args if args else ''):<44} {text} ({rt})")
    print("Outputs: outputs/<model>/. Old commands (python <script>.py from mat_models/) still work.")


def run_new(a):
    from cbfcsf.checks.new import main
    argv = [f"--{flag}" for flag, _ in NEW_STAGES if getattr(a, flag.replace("-", "_"))]
    for opt in ("fit_csv", "rate", "t_on", "t_off"):
        if getattr(a, opt) is not None:
            argv += [f"--{opt.replace('_', '-')}", getattr(a, opt)]
    if a.etco2:
        argv.append("--etco2")
    for opt in ("etco2_csv", "etco2_unit", "cycles", "cycle_length", "barometric"):
        if getattr(a, opt) is not None:
            argv += [f"--{opt.replace('_', '-')}", getattr(a, opt)]
    parts = [flag.replace("-", "_") for flag, _ in NEW_PARTS if getattr(a, flag.replace("-", "_"))]
    return main(argv, parts=parts or None)


def run_small(model, a):
    if model == "ursino":
        from cbfcsf.checks.ursino import steal
        from cbfcsf.models.ursino_2010 import main as baseline
        todo = {"baseline": baseline, "steal": steal,
                "aux_check": lambda: run_experiment("urs_dynamic_pa_test", ["check"])}
    elif model == "vanek":
        from cbfcsf.checks.vanek import run_vanek_lumbar_infusion_test
        from cbfcsf.models.vanek_2015 import main as baseline
        todo = {"baseline": baseline, "lit": run_vanek_lumbar_infusion_test}
    else:
        from cbfcsf.checks.stepanik import run_stepanik_lumbar_infusion_test
        from cbfcsf.models.stepanik_2009 import main as baseline
        todo = {"baseline": baseline, "lit": run_stepanik_lumbar_infusion_test}
    chosen = [k for k in todo if a.all or getattr(a, k)]
    if not chosen:
        parser().parse_args([model, "--help"])
    for k in chosen:
        todo[k]()
    return 0


def run_experiment(name, args):
    sys.argv = [f"{name}.py"] + list(args)
    runpy.run_module(f"experiments.{name}", run_name="__main__")
    return 0


def main(argv=None):
    a = parser().parse_args(argv)
    if a.list or a.model is None:
        show_list()
        return 0
    if a.model == "new":
        return 1 if run_new(a) else 0
    if a.model == "exp":
        return run_experiment(a.name, a.args)
    return run_small(a.model, a)
