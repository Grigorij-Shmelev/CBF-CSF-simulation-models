"""Direct calls of the model functions for the golden-output tests.

    python tests/direct.py <group>        group: new, ursino, vanek, stepanik

Run by the test harness with CAPTURE_DIR set: the results go to CAPTURE_DIR/direct.npz and direct.json.
The calls and their arguments must never change; only the imports in the api_* functions follow the code layout.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "hook"))
from capture_util import flatten  # noqa: E402

ARRAYS, TEXTS = {}, {}


def put(key, obj):
    flatten(obj, key, ARRAYS, TEXTS)


def attrs(obj):
    """Numeric attributes of a model instance (parameters and the computed resting state)."""
    return {k: v for k, v in vars(obj).items() if isinstance(v, (int, float, np.ndarray, np.generic, dict))}


# ---- imports (the only part that follows the layout) ----------------------------------------------------------------
def api_new():
    sys.path.insert(0, str(ROOT / "mat_models"))
    import new_model
    return new_model


def api_ursino():
    sys.path.insert(0, str(ROOT / "mat_models"))
    import ursino_2010_model
    import urs_dynamic_pa_test
    return ursino_2010_model, urs_dynamic_pa_test


def api_vanek():
    sys.path.insert(0, str(ROOT / "mat_models"))
    import vanek_2015_model
    return vanek_2015_model


def api_stepanik():
    sys.path.insert(0, str(ROOT / "mat_models"))
    import stepanik_2009_csf_model
    return stepanik_2009_csf_model


# ---- groups ---------------------------------------------------------------------------------------------------------
def group_new():
    nm = api_new()
    m = nm.NewModel()
    put("attrs/default", attrs(m))
    put("attrs/vanek_values", attrs(nm.NewModel.vanek_values()))
    put("attrs/kappa_v_0.1", attrs(nm.NewModel(kappa_v=0.1)))
    put("attrs/phi_v_0.5", attrs(nm.NewModel(phi_v=0.5)))
    put("attrs/kC_scale_10", attrs(nm.NewModel(kC_scale=10.0)))
    rep = nm.Report()
    for name in ("rest", "co2", "co2_5pct", "lit", "stroke", "pulsatile", "ica"):
        put(f"run/{name}", getattr(nm, f"test_{name}")(rep, m))
    put("report", rep.rows)
    put("quick_outputs", nm.quick_outputs(m))
    put("onset_metrics", nm.onset_metrics(m))
    proto = nm.lit_protocol()
    put("lit_curve", nm.lit_curve(nm.LIT_TRUTH, np.arange(0.0, proto["t_end"], 2.0), proto))
    t, p = nm.load_lit_csv(str(ROOT / "tests" / "fixtures" / "lit_synthetic.csv"))
    put("load_lit_csv", (t, p))
    put("resample_mean", nm.resample_mean(t, p))
    put("resample_mean_gaps", nm.resample_mean(np.r_[t[:100], t[400:]], np.r_[p[:100], p[400:]], dt=0.7))
    put("vanek_lit", nm.vanek_lit(1.5 / 60.0))
    put("vanek_lit_pb12", nm.vanek_lit(1.5 / 60.0, Pb=12.0))
    put("ramp", [nm.ramp(tt, [(0, 1), (10, 3), (20, -2)]) for tt in (-1.0, 0.0, 3.3, 10.0, 15.5, 25.0)])


def ursino_states(u):
    y0 = u.get_initial_state()
    y1 = y0.copy()
    y1[11:17] = [0.01, -0.02, 0.0, 0.03, -0.01, 0.02]          # x_aut
    y1[17:23] = [0.05, 0.01, -0.03, 0.0, 0.02, -0.04]          # x_CO2: both branches of the sigmoid
    y2 = y1.copy()
    y2[0], y2[1] = 40.0, 45.0                                    # Pic high: P_c <= Pic branch, collapsed veins
    return {"y0": y0, "y1": y1, "y2": y2}


def group_ursino():
    um, dpa = api_ursino()
    u = um.Ursino2010Model()
    put("attrs", attrs(u))
    kws = {"plain": {}, "co2": dict(PaCO2=60.0), "ica": dict(ica_stenosis_l=500.0), "mca": dict(mca_stenosis_l=1000.0),
           "pa80": dict(Pa=80.0)}
    for sname, y in ursino_states(u).items():
        for kname, kw in kws.items():
            put(f"derivatives/{sname}/{kname}", u.derivatives(0.0, y, **kw))
        put(f"flows/{sname}", u.get_flow_dict(y))
        put(f"flows_sten/{sname}", u.get_flow_dict(y, Pa=90.0, ica_stenosis_l=500.0, mca_stenosis_l=1000.0))
        put(f"district_Rd/{sname}", dpa.district_Rd(u, y))
        put(f"aux_pressures/{sname}", dpa.aux_pressures(u, y))
        put(f"aux_pressures_mca/{sname}", dpa.aux_pressures(u, y, mca_stenosis_l=1000.0))


def group_vanek():
    vm = api_vanek()
    v = vm.Vanek2015Model()
    put("attrs", attrs(v))
    put("run_vanek_simulation", vm.run_vanek_simulation(duration_s=10.0))
    put("run_vanek_simulation_inf", vm.run_vanek_simulation(duration_s=5.0, infusion_rate_ml_min=1.5))
    put("compliances", [v.compute_compliances(p, p - 1, p + 1, p + 2, p - 2, 80.0, 10.0, 2.0, 2.5, 1.5)
                        for p in (2.0, 9.5, 10.0, 23.0)])
    put("resorption", [v.resorption(p) for p in (0.0, 2.0, 6.0, 10.0, 25.0)])
    const = {k: (lambda t, c=c: c) for k, c in dict(Pa=80.0, Pb=10.0, PC=2.0, PT=2.0, PL=2.0).items()}
    P0 = np.array([11.0, 10.0, 10.0, 10.0, 10.0])
    put("derivatives", v.derivatives(0.0, P0, const))
    put("derivatives_inf", v.derivatives(0.0, P0, const, infusion_rates=np.array([0, 0, 0, 0, 0.025])))


def group_stepanik():
    sm = api_stepanik()
    s = sm.Stepanik2009Model()
    put("attrs", attrs(s))
    put("run_stepanik_simulation", sm.run_stepanik_simulation(duration_s=10.0, save_plot_path=None))
    s2 = sm.Stepanik2009Model()
    s2.infusion_rate[4], s2.infusion_start[4], s2.infusion_duration[4] = 0.025, 2.0, 3.0
    put("run_stepanik_simulation_inf", sm.run_stepanik_simulation(model=s2, duration_s=6.0, save_plot_path=None))
    put("resorption", [s.resorption(p) for p in (0.0, 2.0, 6.0, 10.0, 25.0)])
    put("infusion", [s2.infusion(t, 4) for t in (0.0, 2.0, 3.5, 5.0, 5.5)])


if __name__ == "__main__":
    group = sys.argv[1]
    globals()[f"group_{group}"]()
    out = os.environ.get("CAPTURE_DIR", ".")
    np.savez(os.path.join(out, "direct.npz"), **ARRAYS)
    with open(os.path.join(out, "direct.json"), "w") as fh:
        json.dump(TEXTS, fh, indent=0, default=repr)
    print(f"direct.{group}: {len(ARRAYS)} arrays, {len(TEXTS)} texts")
