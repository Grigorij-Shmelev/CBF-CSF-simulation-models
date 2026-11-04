"""
urs_dynamic_pa_test.py  --  Task D2.1 (+ scaffolding for D2.2/D2.3)

Feeds a time-varying arterial pressure Pa(t) into Ursino & Giannessi (2010).
ursino_2010_model.py is NOT modified: Pa(t) is passed through the existing
`Pa` keyword of derivatives() and get_flow_dict() at every time point.

Place in mat_models/ and run from there (like the other urs_* scripts):
    python urs_dynamic_pa_test.py            -> 10 s smoke test (normocapnia)
    python urs_dynamic_pa_test.py full       -> 400 s run, hypercapnia at t = 50 s

Scenarios
    S     static Pa = 100 mmHg                    (current behaviour)
    D80   80 + 10 sin(2*pi*1.2*t)                 (identical to Vanek's synthetic Pa,
                                                   vanek_2015_model.run_vanek_simulation)
    D100  100 + 10 sin(2*pi*1.2*t)                (same pulse, Ursino's mean; separates
                                                   the effect of pulsatility from the mean shift)
"""
import sys
import time as clock
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ursino_2010_model import Ursino2010Model

# ----------------------------------------------------------------------------
# 1. Arterial pressure inputs  Pa(t)  [mmHg]
#    Any function of t works. For a measured waveform (array), wrap it:
#    pa = PchipInterpolator(t_data, pa_data); pa_func = lambda t: float(pa(t))
# ----------------------------------------------------------------------------
F_CARDIAC = 1.2  # Hz, as in vanek_2015_model.py

SCENARIOS = {
    "S":    {"mean": 100.0, "func": lambda t: 100.0},
    "D80":  {"mean": 80.0,  "func": lambda t: 80.0 + 10.0 * np.sin(2 * np.pi * F_CARDIAC * t)},
    "D100": {"mean": 100.0, "func": lambda t: 100.0 + 10.0 * np.sin(2 * np.pi * F_CARDIAC * t)},
}

# State indices (see Ursino2010Model.derivatives)
NODES = {"Pic": 0, "Pv": 1, "PICA_l": 2, "PICA_r": 3, "PBA": 4,
         "Pd_ml": 5, "Pd_al": 6, "Pd_pl": 7, "Pd_mr": 8, "Pd_ar": 9, "Pd_pr": 10}

DT_OUT = 0.005    # output sampling [s] -> ~167 points per 1.2 Hz cycle
MAX_STEP = 0.01   # solver step limit [s]; must resolve the pulse (>= ~80 steps/cycle)


# ----------------------------------------------------------------------------
# 2. Settling: get_initial_state() is NOT an equilibrium (dPICA/dt ~ 2660 mmHg/s
#    at t = 0), so every scenario first settles with a static Pa = its mean.
# ----------------------------------------------------------------------------
def settle(model, pa_mean, t_settle=300.0):
    sol = solve_ivp(lambda t, y: model.derivatives(t, y, Pa=pa_mean, PaCO2=40.0),
                    [0.0, t_settle], model.get_initial_state(),
                    method="Radau", rtol=1e-6, atol=1e-6)
    return sol.y[:, -1]


# ----------------------------------------------------------------------------
# 3. Main run: normocapnia, then (optional) step to hypercapnia.
#    Absolute time is kept across phases, so pa_func(t) stays continuous.
# ----------------------------------------------------------------------------
def run_ursino(model, pa_func, y0, t_end, t_hyper=None, pco2_hyper=60.0):
    switch = [0.0] + ([t_hyper] if t_hyper is not None else []) + [t_end]
    t_all, y_all = [], []
    y_start = y0
    for k in range(len(switch) - 1):
        ta, tb = switch[k], switch[k + 1]
        pco2 = 40.0 if k == 0 else pco2_hyper
        t_eval = np.arange(ta, tb, DT_OUT)
        sol = solve_ivp(lambda t, y: model.derivatives(t, y, Pa=pa_func(t), PaCO2=pco2),
                        [ta, tb], y_start, method="Radau", t_eval=np.append(t_eval, tb),
                        max_step=MAX_STEP, rtol=1e-6, atol=1e-6)
        if not sol.success:
            raise RuntimeError(f"solver failed in phase {k}: {sol.message}")
        t_all.append(sol.t[:-1]); y_all.append(sol.y[:, :-1])
        y_start = sol.y[:, -1]
    return np.concatenate(t_all), np.hstack(y_all)


def flows(model, t, y, pa_func):
    """Flows need Pa at the SAME time point (get_flow_dict has no t argument)."""
    out = {k: np.empty(len(t)) for k in ("q_ica_l", "q_ica_r", "q_mca_l", "q_mca_r")}
    for i in range(len(t)):
        fl = model.get_flow_dict(y[:, i], Pa=pa_func(t[i]))
        for k in out:
            out[k][i] = fl[k]
    return out


def window_stats(t, sig, t0, t1):
    """Mean and peak-to-peak of a signal over [t0, t1) (use whole cycles)."""
    m = (t >= t0) & (t < t1)
    return sig[m].mean(), np.ptp(sig[m])


def min_control_signal(model, y):
    """Diagnostic: min of x = xCO2 - xaut per district vs. the Cd = 0 threshold."""
    x = y[17:23] - y[11:17]
    for i, d in enumerate(model.districts):
        x_crit = model.Cdn[d] * model.sat1 / 2.0 * np.log(0.75)
        flag = "  <-- Cd <= 0 reached" if x[i].min() < x_crit else ""
        print(f"    {d}: min x = {x[i].min():+.4f}   (Cd = 0 at x = {x_crit:+.4f}){flag}")


# ----------------------------------------------------------------------------
# 4. Experiment
# ----------------------------------------------------------------------------
def main(full=False):
    t_end, t_hyper = (400.0, 50.0) if full else (10.0, None)  # ICP still drifts at 200 s
    n_cyc = 6                                    # stats over whole cycles
    win = n_cyc / F_CARDIAC
    windows = {"pre": (t_hyper - win, t_hyper) if full else (t_end - win, t_end)}
    if full:
        windows["hyper"] = (t_end - win, t_end)

    model = Ursino2010Model()
    results = {}
    for name, sc in SCENARIOS.items():
        tic = clock.time()
        y0 = settle(model, sc["mean"])
        t, y = run_ursino(model, sc["func"], y0, t_end, t_hyper)
        q = flows(model, t, y, sc["func"])
        results[name] = (t, y, q)
        print(f"\n=== {name}: {clock.time() - tic:.1f} s wall time, {len(t)} samples")
        for wname, (a, b) in windows.items():
            print(f"  window '{wname}' [{a:.1f}, {b:.1f}) s   mean / peak-to-peak")
            pa = np.array([sc['func'](ti) for ti in t])
            mu, pp = window_stats(t, pa, a, b)
            print(f"    {'Pa':8s} {mu:8.2f} / {pp:6.3f}")
            for node in ("PICA_l", "PBA", "Pd_ml", "Pd_al", "Pd_pl", "Pic", "Pv"):
                mu, pp = window_stats(t, y[NODES[node]], a, b)
                print(f"    {node:8s} {mu:8.2f} / {pp:6.3f}")
            for k in ("q_ica_l", "q_mca_l"):
                mu, pp = window_stats(t, q[k], a, b)
                print(f"    {k:8s} {mu:8.3f} / {pp:6.3f}  mL/s")
        min_control_signal(model, y)

    # --- Plot: full run (left) and 3 s zoom at the end (right) ---
    rows = [("PICA_l", "ICA pressure, left [mmHg]"),
            ("Pd_ml", "Distal pial pressure, MCA-l [mmHg]"),
            ("Pic", "Intracranial pressure [mmHg]"),
            ("q_mca_l", "MCA flow, left [mL/s]")]
    fig, ax = plt.subplots(len(rows), 2, figsize=(13, 11), sharex="col")
    for name, (t, y, q) in results.items():
        sig_of = lambda key: q[key] if key.startswith("q_") else y[NODES[key]]
        zoom = t >= t_end - 3.0
        for r, (key, label) in enumerate(rows):
            st = dict(lw=1.4, zorder=3) if name == "S" else dict(lw=0.6)  # keep S visible
            ax[r, 0].plot(t, sig_of(key), label=name, **st)
            ax[r, 1].plot(t[zoom], sig_of(key)[zoom], label=name, **st)
            ax[r, 0].set_ylabel(label, fontsize=8)
    for r in range(len(rows)):
        for c in range(2):
            ax[r, c].grid(alpha=0.3)
            if full and c == 0:
                ax[r, c].axvline(t_hyper, color="k", ls=":", lw=0.8)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("Full run" + (" (hypercapnia at dotted line)" if full else ""))
    ax[0, 1].set_title("Zoom: last 3 s")
    ax[-1, 0].set_xlabel("t [s]"); ax[-1, 1].set_xlabel("t [s]")
    fig.suptitle("Ursino (2010) driven by time-varying Pa(t): S vs D80 vs D100")
    fig.tight_layout()
    out = "urs_dynamic_pa_full.png" if full else "urs_dynamic_pa_smoke.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved as {out}")


if __name__ == "__main__":
    main(full=(len(sys.argv) > 1 and sys.argv[1] == "full"))
