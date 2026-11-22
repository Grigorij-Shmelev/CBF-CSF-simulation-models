"""Every captured command: one golden output per id.

argv is a dict of variants. "old" = the command as it was before the refactor (from mat_models/), "new" = the
command of the new layout. Both must reproduce the same golden output. "@path" = path relative to the run root.
Commands with the same `chain` run one after the other in the same copy (the later ones read earlier outputs).
Masks replace wall-clock times, the only output that may differ between runs.
stderr_unordered: the scipy warnings of parallel worker processes reach stderr in a scheduling-dependent order; for
these commands stderr is compared as a multiset of warning blocks (every block and its count must match).
"""
from dataclasses import dataclass, field

T_FIT = (r"^fit stage took \d+ s$", "fit stage took <T> s")
T_DPA = (r"^(=== \S+: )[\d.]+( s wall time)", r"\1<T>\2")
T_NAME_S = (r"^(.+): \d+ s$", r"\1: <T> s")
T_EXPORT = (r"(ms step\), )\d+( s -> )", r"\1<T>\2")


@dataclass
class Cmd:
    id: str
    argv: dict
    cwd: str = "mat_models"
    masks: tuple = ()
    chain: str = ""
    exclusive: bool = False      # run alone, so that its runtime is honest
    stderr_unordered: bool = False   # stderr written by several worker processes: compared as a multiset of blocks
    timeout: int = 7200
    deps: tuple = field(default_factory=tuple)   # extra files the import scan cannot see
    cwd_new: str = "."           # the new form runs from the repository root


NEW = "@mat_models/new_model.py"
M = "@mat_models/"


def old(*argv):
    return {"old": list(argv)}


def new(new_argv, *old_argv):
    """Both forms of a command: new_argv runs from the repository root, old_argv from mat_models/."""
    return {"old": list(old_argv), "new": new_argv}


def new_only(*argv):
    """A command added after the refactor: it exists only in the new layout (runs from the repository root)."""
    return {"new": list(argv)}


COMMANDS = [
    # ---- new model ------------------------------------------------------------------------------------------------
    Cmd("new.default", new(["@run.py", "new"], NEW)),
    Cmd("new.calibrate", new(["@run.py", "new", "--calibrate"], NEW, "--calibrate")),
    Cmd("new.sensitivity", new(["@run.py", "new", "--sensitivity"], NEW, "--sensitivity")),
    Cmd("new.params", new(["@run.py", "new", "--params"], NEW, "--params")),
    Cmd("new.options", new(["@run.py", "new", "--options"], NEW, "--options")),
    Cmd("new.fit", new(["@run.py", "new", "--fit"], NEW, "--fit"), masks=(T_FIT,), exclusive=True, stderr_unordered=True),
    Cmd("new.fit_mismatch", new(["@run.py", "new", "--fit-mismatch"], NEW, "--fit-mismatch"), stderr_unordered=True),
    Cmd("new.fit_csv", new(["@run.py", "new", "--fit-csv", "@tests/fixtures/lit_synthetic.csv", "--rate", "1.5", "--t-on", "300", "--t-off", "900"], NEW, "--fit-csv", "@tests/fixtures/lit_synthetic.csv", "--rate", "1.5", "--t-on", "300",
                           "--t-off", "900")),
    Cmd("new.all", new(["@run.py", "new", "--all"], NEW, "--all"), masks=(T_FIT,), stderr_unordered=True),
    Cmd("new.etco2", new_only("@run.py", "new", "--etco2")),      # spec 7.17, opt-in (not in --all)
    # ---- Ursino ---------------------------------------------------------------------------------------------------
    Cmd("ursino.baseline", new(["@run.py", "ursino", "--baseline"], M + "ursino_2010_model.py")),
    Cmd("ursino.steal", new(["@run.py", "ursino", "--steal"], M + "urs_steal_test.py")),
    Cmd("ursino.stiffness", new(["@run.py", "exp", "urs_occlusion_stiffness_test"], M + "urs_occlusion_stiffness_test.py")),
    Cmd("ursino.sigmoid", new(["@run.py", "exp", "urs_cd_sigmoid"], M + "urs_cd_sigmoid.py")),
    Cmd("ursino.dpa_smoke", new(["@run.py", "exp", "urs_dynamic_pa_test"], M + "urs_dynamic_pa_test.py"), masks=(T_DPA,)),
    Cmd("ursino.dpa_full", new(["@run.py", "exp", "urs_dynamic_pa_test", "full"], M + "urs_dynamic_pa_test.py", "full"), masks=(T_DPA,)),
    Cmd("ursino.dpa_check", new(["@run.py", "ursino", "--aux-check"], M + "urs_dynamic_pa_test.py", "check")),
    Cmd("ursino.dpa_table", new(["@run.py", "exp", "urs_dynamic_pa_test", "table"], M + "urs_dynamic_pa_test.py", "table"), masks=(T_NAME_S,), chain="table"),
    Cmd("ursino.dpa_profile", new(["@run.py", "exp", "urs_dynamic_pa_test", "profile"], M + "urs_dynamic_pa_test.py", "profile"), chain="table"),
    # ---- Vanek ----------------------------------------------------------------------------------------------------
    Cmd("vanek.baseline", new(["@run.py", "vanek", "--baseline"], M + "vanek_2015_model.py")),
    Cmd("vanek.lit", new(["@run.py", "vanek", "--lit"], M + "run_vanek_lumbar_infusion.py")),
    Cmd("vanek.icp_sensitivity", new(["@run.py", "exp", "vanek_icp_sensitivity_tests"], M + "vanek_icp_sensitivity_tests.py")),
    Cmd("vanek.pb_step", new(["@run.py", "exp", "vanek_pb_step_test"], M + "vanek_pb_step_test.py")),
    Cmd("vanek.c2b0_sweep", new(["@run.py", "exp", "vanek_c2b0_sweep"], M + "vanek_c2b0_sweep.py")),
    # ---- Stepanik -------------------------------------------------------------------------------------------------
    Cmd("stepanik.baseline", new(["@run.py", "stepanik", "--baseline"], M + "stepanik_2009_csf_model.py")),
    Cmd("stepanik.lit", new(["@run.py", "stepanik", "--lit"], M + "stepanik_run_lumbar_infusion.py")),
    # ---- comparison and coupling ----------------------------------------------------------------------------------
    Cmd("compare.all_models", new(["@run.py", "exp", "compare_all_models"], M + "compare_all_models.py")),
    Cmd("coupling.export", new(["@run.py", "exp", "urs_dynamic_pa_test", "export"], M + "urs_dynamic_pa_test.py", "export"), masks=(T_EXPORT,), chain="coupling"),
    Cmd("coupling.driven_D100", new(["@run.py", "exp", "vanek_driven_by_ursino"], M + "vanek_driven_by_ursino.py"), masks=(T_NAME_S,), chain="coupling"),
    Cmd("coupling.driven_D80", new(["@run.py", "exp", "vanek_driven_by_ursino", "D80"], M + "vanek_driven_by_ursino.py", "D80"), masks=(T_NAME_S,), chain="coupling"),
    Cmd("coupling.VU_unified", new(["@run.py", "exp", "VU_unified_brain_model"], M + "VU_unified_brain_model.py")),
    # ---- direct calls of the model functions ----------------------------------------------------------------------
    Cmd("direct.new", old("@tests/direct.py", "new"), cwd="."),
    Cmd("direct.ursino", old("@tests/direct.py", "ursino"), cwd="."),
    Cmd("direct.vanek", old("@tests/direct.py", "vanek"), cwd="."),
    Cmd("direct.stepanik", old("@tests/direct.py", "stepanik"), cwd="."),
]

BY_ID = {c.id: c for c in COMMANDS}
