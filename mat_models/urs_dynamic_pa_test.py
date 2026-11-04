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
from ursino_2010_model import Ursino2010Model
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import time as clock
import numpy as np
import matplotlib
matplotlib.use("Agg")

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
# solver step limit [s]; must resolve the pulse (>= ~80 steps/cycle)
MAX_STEP = 0.01


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
        t_all.append(sol.t[:-1])
        y_all.append(sol.y[:, :-1])
        y_start = sol.y[:, -1]
    return np.concatenate(t_all), np.hstack(y_all)


def flows(model, t, y, pa_func):
    """Flows need Pa at the SAME time point (get_flow_dict has no t argument)."""
    out = {k: np.empty(len(t))
           for k in ("q_ica_l", "q_ica_r", "q_mca_l", "q_mca_r")}
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
        print(
            f"    {d}: min x = {x[i].min():+.4f}   (Cd = 0 at x = {x_crit:+.4f}){flag}")


# ----------------------------------------------------------------------------
# 3b. D2.3 Step 2: auxiliary node pressures.
#     derivatives() computes these internally but does not return them, so the
#     formulas are COPIED here 1:1 from Ursino2010Model.derivatives().
#     If you ever change them in the model, change them here too -> check_aux().
# ----------------------------------------------------------------------------
def district_Rd(model, y):
    """Pial arterial resistance Rd per district (copy of the loop in derivatives)."""
    Pic, Rd = y[0], {}
    for i, d in enumerate(model.districts):
        x = y[17 + i] - y[11 + i]                       # xCO2 - xaut
        Cd0 = model.Cdn[d]
        sat = model.sat1 if x < 0 else model.sat2
        DCd = 2.0 * sat
        kCd = Cd0 * sat / 2.0
        u = np.exp(x / kCd)
        Cd = Cd0 * ((1.0 - DCd / 2.0) + (1.0 + DCd / 2.0) * u) / (1.0 + u)
        Vd = max(Cd * (y[5 + i] - Pic), 1e-4)
        Rd[d] = model.kR[d] * (model.Cdn[d] ** 2) / (Vd ** 2)
    return Rd


def aux_pressures(model, y, mca_stenosis_l=1.0):
    """Pressures at the algebraic (compliance-free) nodes, plus P_MCA_end [mmHg]."""
    Pic, Pv, PICAl, PICAr, PBA = y[0], y[1], y[2], y[3], y[4]
    Pd = dict(zip(model.districts, y[5:11]))
    Rd = district_Rd(model, y)
    R = model.R_net
    R_MCAl, R_MCAr = R['MCA'] * mca_stenosis_l, R['MCA']

    # Posterior communicating junctions (PCA1/PCA2/PCoA meet)
    G_l = 1 / (R['PCA2'] + Rd['pl'] / 2) + 1 / R['PCoA'] + 1 / R['PCA1']
    PPCA1l = (Pd['pl'] / (R['PCA2'] + Rd['pl'] / 2) +
              PICAl / R['PCoA'] + PBA / R['PCA1']) / G_l
    G_r = 1 / (R['PCA2'] + Rd['pr'] / 2) + 1 / R['PCoA'] + 1 / R['PCA1']
    PPCA1r = (Pd['pr'] / (R['PCA2'] + Rd['pr'] / 2) +
              PICAr / R['PCoA'] + PBA / R['PCA1']) / G_r

    # Anterior junctions, coupled through ACoA -> 2x2 linear system (Cramer's rule)
    g_al = 1 / (R['ACA2'] + Rd['al'] / 2) + 1 / R['ACA1'] + 1 / R['ACoA']
    g_ar = 1 / (R['ACA2'] + Rd['ar'] / 2) + 1 / R['ACA1'] + 1 / R['ACoA']
    b_l = Pd['al'] / (R['ACA2'] + Rd['al'] / 2) + PICAl / R['ACA1']
    b_r = Pd['ar'] / (R['ACA2'] + Rd['ar'] / 2) + PICAr / R['ACA1']
    det = g_al * g_ar - (1 / R['ACoA']) ** 2
    PACA1l = (g_ar * b_l + (1 / R['ACoA']) * b_r) / det
    PACA1r = (g_al * b_r + (1 / R['ACoA']) * b_l) / det

    # Capillary node (if Pc <= Pic, CSF formation stops: Rf branch removed)
    G_cap = sum(2 / Rd[d] for d in model.districts) + \
        1 / model.Rpv + 1 / model.Rf
    num = sum(2 * Pd[d] / Rd[d]
              for d in model.districts) + Pic / model.Rf + Pv / model.Rpv
    Pc = num / G_cap
    if Pc <= Pic:
        G_cap -= 1 / model.Rf
        num = sum(2 * Pd[d] / Rd[d] for d in model.districts) + Pv / model.Rpv
        Pc = num / G_cap

    # Derived (NOT an Ursino node): pressure at the distal end of the MCA trunk,
    # i.e. after R_MCA, before the pial resistance Rd/2
    q_mca_l = (PICAl - Pd['ml']) / (R_MCAl + Rd['ml'] / 2)
    q_mca_r = (PICAr - Pd['mr']) / (R_MCAr + Rd['mr'] / 2)

    return {"PPCA1_l": PPCA1l, "PPCA1_r": PPCA1r, "PACA1_l": PACA1l, "PACA1_r": PACA1r,
            "Pc": Pc, "PMCA_end_l": PICAl - q_mca_l * R_MCAl, "PMCA_end_r": PICAr - q_mca_r * R_MCAr}


def check_aux(model, pa_func, t, y, label):
    """D2.3 Step 3. Two independent tests per sample, residuals in mL/s.
    (A) Kirchhoff: flows into each algebraic node sum to 0.
    (B) Agreement with the model's OWN derivatives(): rebuild the flow balance of
        ICA_l, ICA_r, BA and the venous node from the helper's pressures and compare
        with (dy[k] - dPic_dt) * C_k returned by derivatives()."""
    R = model.R_net
    worst = {}

    def keep(name, val):
        worst[name] = max(worst.get(name, 0.0), abs(val))
    for i in range(len(t)):
        yi, Pa = y[:, i], pa_func(t[i])
        Pic, Pv, PICAl, PICAr, PBA = yi[:5]
        Pd = dict(zip(model.districts, yi[5:11]))
        Rd, a = district_Rd(model, yi), aux_pressures(model, yi)

        # (A) Kirchhoff at the 5 algebraic nodes
        keep("A  PPCA1_l", (Pd['pl'] - a['PPCA1_l']) / (R['PCA2'] + Rd['pl'] / 2)
             + (PICAl - a['PPCA1_l']) / R['PCoA'] + (PBA - a['PPCA1_l']) / R['PCA1'])
        keep("A  PPCA1_r", (Pd['pr'] - a['PPCA1_r']) / (R['PCA2'] + Rd['pr'] / 2)
             + (PICAr - a['PPCA1_r']) / R['PCoA'] + (PBA - a['PPCA1_r']) / R['PCA1'])
        keep("A  PACA1_l", (PICAl - a['PACA1_l']) / R['ACA1'] + (a['PACA1_r'] - a['PACA1_l']) / R['ACoA']
             + (Pd['al'] - a['PACA1_l']) / (R['ACA2'] + Rd['al'] / 2))
        keep("A  PACA1_r", (PICAr - a['PACA1_r']) / R['ACA1'] + (a['PACA1_l'] - a['PACA1_r']) / R['ACoA']
             + (Pd['ar'] - a['PACA1_r']) / (R['ACA2'] + Rd['ar'] / 2))
        qf = (a['Pc'] - Pic) / model.Rf if a['Pc'] > Pic else 0.0
        keep("A  Pc", sum(2 * (Pd[d] - a['Pc']) / Rd[d] for d in model.districts)
             - (a['Pc'] - Pv) / model.Rpv - qf)

        # (B) agreement with derivatives()
        dy = model.derivatives(t[i], yi, Pa=Pa)
        net_l = ((Pa - PICAl) / R['ICA'] + (a['PPCA1_l'] - PICAl) / R['PCoA']
                 + (Pd['ml'] - PICAl) / (R['MCA'] + Rd['ml'] / 2) + (a['PACA1_l'] - PICAl) / R['ACA1'])
        keep("B  ICA_l node", net_l - (dy[2] - dy[0]) * model.C_ICA)
        net_r = ((Pa - PICAr) / R['ICA'] + (a['PPCA1_r'] - PICAr) / R['PCoA']
                 + (Pd['mr'] - PICAr) / (R['MCA'] + Rd['mr'] / 2) + (a['PACA1_r'] - PICAr) / R['ACA1'])
        keep("B  ICA_r node", net_r - (dy[3] - dy[0]) * model.C_ICA)
        net_ba = ((Pa - PBA) / R['BA'] + (a['PPCA1_l'] - PBA) /
                  R['PCA1'] + (a['PPCA1_r'] - PBA) / R['PCA1'])
        keep("B  BA node", net_ba - (dy[4] - dy[0]) * model.C_BA)
        Rvs = max((Pv - model.Pvs) / (Pv - Pic) *
                  model.Rvs1 if model.Pvs < Pic else model.Rvs1, 1e-4)
        Cvi = 1.0 / (model.kven * max(Pv - Pic - model.Pv1, 0.1))
        net_v = (a['Pc'] - Pv) / model.Rpv - (Pv - model.Pvs) / Rvs
        keep("B  venous node", net_v - (dy[1] - dy[0]) * Cvi)

    print(
        f"\n--- check_aux: {label} ({len(t)} samples), max |residual| in mL/s ---")
    ok = True
    for name, val in worst.items():
        flag = "OK" if val < 1e-8 else "FAIL"
        ok &= flag == "OK"
        print(f"    {name:16s} {val:.2e}  {flag}")
    return ok


# ----------------------------------------------------------------------------
# 4. Experiment
# ----------------------------------------------------------------------------
def main(full=False):
    t_end, t_hyper = (400.0, 50.0) if full else (
        10.0, None)  # ICP still drifts at 200 s
    n_cyc = 6                                    # stats over whole cycles
    win = n_cyc / F_CARDIAC
    windows = {"pre": (t_hyper - win, t_hyper)
               if full else (t_end - win, t_end)}
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
        print(
            f"\n=== {name}: {clock.time() - tic:.1f} s wall time, {len(t)} samples")
        for wname, (a, b) in windows.items():
            print(
                f"  window '{wname}' [{a:.1f}, {b:.1f}) s   mean / peak-to-peak")
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
        def sig_of(key): return q[key] if key.startswith(
            "q_") else y[NODES[key]]
        zoom = t >= t_end - 3.0
        for r, (key, label) in enumerate(rows):
            st = dict(lw=1.4, zorder=3) if name == "S" else dict(
                lw=0.6)  # keep S visible
            ax[r, 0].plot(t, sig_of(key), label=name, **st)
            ax[r, 1].plot(t[zoom], sig_of(key)[zoom], label=name, **st)
            ax[r, 0].set_ylabel(label, fontsize=8)
    for r in range(len(rows)):
        for c in range(2):
            ax[r, c].grid(alpha=0.3)
            if full and c == 0:
                ax[r, c].axvline(t_hyper, color="k", ls=":", lw=0.8)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(
        "Full run" + (" (hypercapnia at dotted line)" if full else ""))
    ax[0, 1].set_title("Zoom: last 3 s")
    ax[-1, 0].set_xlabel("t [s]")
    ax[-1, 1].set_xlabel("t [s]")
    fig.suptitle(
        "Ursino (2010) driven by time-varying Pa(t): S vs D80 vs D100")
    fig.tight_layout()
    out = "urs_dynamic_pa_full.png" if full else "urs_dynamic_pa_smoke.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved as {out}")


def run_checks():
    """D2.3 Step 3: verify aux_pressures() on a static steady state and along a
    pulsatile (D100) trajectory incl. hypercapnia. No stenosis (check (B) assumes it)."""
    model = Ursino2010Model()
    ok = True
    for name in ("S", "D100"):
        sc = SCENARIOS[name]
        y0 = settle(model, sc["mean"])
        t, y = run_ursino(model, sc["func"], y0, t_end=20.0, t_hyper=10.0)
        ok &= check_aux(model, sc["func"], t, y,
                        f"{name}, 20 s, hypercapnia at 10 s")
        a = aux_pressures(model, y[:, -1])
        print("    last sample: " +
              ", ".join(f"{k}={v:.2f}" for k, v in a.items()))
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "check":
        run_checks()
    else:
        main(full=(mode == "full"))
