"""Checks of the new model (spec 7.1-7.16) and its command line.

Usage (from the repository root; the old form `python new_model.py [flags]` from mat_models/ takes the same flags):
    python run.py new                   # tests of spec 7.1, 7.3, cross-checks of 7.8 and 7.15, figures to outputs/new/ (2 min)
    python run.py new --calibrate       # + comparison with Vanek's raw compliances (K1), data-file check (K2) (3 min)
    python run.py new --sensitivity     # + phi_v sensitivity V1 of spec 7.5                             (about 7 min)
    python run.py new --params          # + parameter sensitivity SA of spec 7.8                         (about 2 min)
    python run.py new --options         # + venous squeeze and sigmoid width of spec 7.10               (about 2 min)
    python run.py new --fit             # + LIT fitting tool on synthetic data of spec 7.10             (a few min, 8 processes)
    python run.py new --fit-mismatch    # only: fit when the data come from a modified model, spec 7.13 (a few min)
    python run.py new --fit-csv FILE --rate 1.5 --t-on 300 --t-off 900
                                        # fit a measured LIT (CSV header, columns: time_s, pressure_mmHg)
    python run.py new --rest --co2 --lit  # only these parts of the default run (python run.py new --help)
    python run.py new --all
    python run.py new --etco2           # only: end-tidal-shaped CO2 input, PLACEHOLDER shape, spec 7.17 (not in --all)
    python run.py new --etco2-csv FILE --etco2-unit mmHg --cycles 120,390,660 --cycle-length 180
                                        # only: a measured end-tidal curve (CSV header, columns: time_s, etco2)
"""
import os
from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

from cbfcsf.common.report import Report
from cbfcsf.common.signals import (EtCO2Protocol, etco2_curve, etco2_step, pulse, ramp, read_etco2_csv,
                                    write_etco2_csv)
from cbfcsf.crosscheck import test_cross_lit, test_cross_more, test_cross_stroke
from cbfcsf.fit_lit import LIT_DEFAULT, LIT_TRUTH, fit_file, test_fit, test_fit_mismatch
from cbfcsf.models.new_model import NAMES, Inputs, NewModel
from cbfcsf.paths import ROOT, ZBYSEK_DATA, out_dir
from cbfcsf.plots import plot_etco2, plot_etco2_onset, plot_fit, plot_recording, plot_run


# ----------------------------------------------------------------------------------------------------------------------
# Tests of spec section 7.1
# ----------------------------------------------------------------------------------------------------------------------
def marks(t, t_on, t_off):
    """Sample indices: 1 s before the intervention, 1 s before its end, last sample."""
    return np.searchsorted(t, t_on - 1), np.searchsorted(t, t_off - 1), len(t) - 1


# Scenario names (headings of the summaries, keys of the runs)
SC_CO2_60 = "CO2 challenge 40 -> 60 mmHg"
SC_CO2_5 = "5 % CO2 (PaCO2 47.5 mmHg)"
SC_LIT = "Lumbar infusion test, 1.5 mL/min for 10 min"
SC_STROKE = "Stroke: left MCA occluded 30 min, then reopened"
SC_ICA = "Left ICA occluded 30 min, then reopened"


def common_checks(rep, tag, r, symmetric=True):
    """T2 rigid skull, T3 algebraic residual, T7 CSF bookkeeping (= L5, applied to every run), T4 symmetry."""
    rep.check(f"{tag} T2 rigid skull: max |dV_cranial|", np.max(np.abs(r["Vcranial"] - r["Vcranial"][0])), 0, 1e-3, "mL")
    rep.check(f"{tag} T3 algebraic residual", np.max(r["res"]), 0, 1e-8, "mL/s")
    rep.check(f"{tag} T7 CSF bookkeeping error", np.max(np.abs(r["dVcsf"] - r["csf_net_in"])), 0, 0.05, "mL")
    if symmetric:
        asym = np.max(np.abs(r["Va"][:, 0:2] - r["Va"][:, 2:4]) / r["Va"][:, 0:2])
        rep.check(f"{tag} T4 left-right asymmetry", asym, 0, 1e-6, "rel")


def after_reopening(r, q0, i0, i2):
    """Largest relative deviation of flows, arterial and venous volumes at the end of the run from rest."""
    return max(np.max(np.abs(r["q_k"][i2] / q0 - 1)), np.max(np.abs(r["Va"][i2] / r["Va"][i0] - 1)),
               abs(r["Vv"][i2] / r["Vv"][i0] - 1))


def dcbv(r):
    """Stressed CBV change from rest to the end of the intervention."""
    return r["Vblood"][r["marks"][1]] - r["Vblood"][r["marks"][0]]


def test_rest(rep, m):
    r = m.simulate(Inputs(), 600.0, t_eval=np.linspace(0, 600, 601))
    rep.check("T1 rest: max |ICP - 9.47|", np.max(np.abs(r["ICP"] - m.P2_n)), 0, 0.05, "mmHg")
    rep.check("T1 rest: max |q_k/q_n - 1|", np.max(np.abs(r["q_k"] / m.q_n - 1)), 0, 0.005, "rel")
    vols = np.column_stack([r["Va"], r["Vv"], r["V1"], r["V2"]])
    rep.check("T1 rest: max volume drift", np.max(np.abs(vols / vols[0] - 1)), 0, 0.005, "rel")
    rep.check("T1 rest: max |x|", max(np.max(np.abs(r["x_aut"])), np.max(np.abs(r["x_CO2"]))), 0, 0.005, "")
    common_checks(rep, "rest", r)
    return r


def test_co2(rep, m):
    t_on, t_off, t_end = 120.0, 720.0, 3600.0
    pco2 = lambda t: pulse(t, t_on, t_off, t_end, 40, 60, 10)
    r = m.simulate(Inputs(PaCO2=pco2), t_end, t_eval=np.linspace(0, t_end, 3601), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = marks(t, t_on, t_off)

    rep.check("T5a CBF rise at end of hypercapnia", r["CBF"][i1] / r["CBF"][i0] - 1, 0.30, 1.00, "rel")
    rep.check("T5b arterial stressed volume rise", r["Va_total"][i1] / r["Va_total"][i0] - 1, 0.10, 0.45, "rel")
    rep.check("T5c stressed blood volume rise", r["Vblood"][i1] - r["Vblood"][i0], 1e-9, 10.0, "mL")
    rep.check("T5d ICP peak", np.max(r["ICP"]), r["ICP"][i0] + 0.5, 30.0, "mmHg")
    vmin = min(np.min(r["Va"]), np.min(r["Vv"]), np.min(r["V1"]), np.min(r["V2"]))
    rep.check("T5e smallest volume", vmin, 1e-9, np.inf, "mL")
    rep.check("T5e smallest C_k", np.min(r["C"]), 1e-9, np.inf, "mL/mmHg")
    rep.check("T5f recovery: CBF", r["CBF"][i2] / r["CBF"][i0] - 1, -0.02, 0.02, "rel")
    rep.check("T5f recovery: blood volume", r["Vblood"][i2] / r["Vblood"][i0] - 1, -0.02, 0.02, "rel")
    rep.check("T5f recovery: ICP", r["ICP"][i2] - r["ICP"][i0], -0.5, 0.5, "mmHg")

    # T5g cross-check against the existing Ursino code (imported, not modified)
    from cbfcsf.models.ursino_2010 import Ursino2010Model
    u = Ursino2010Model()
    s0 = solve_ivp(lambda tt, y: u.derivatives(tt, y), (0, 300), u.get_initial_state(), method="Radau")
    s1 = solve_ivp(lambda tt, y: u.derivatives(tt, y, PaCO2=60.0), (0, 600), s0.y[:, -1], method="Radau")
    rise_u = u.get_flow_dict(s1.y[:, -1])["q_mca_l"] / u.get_flow_dict(s0.y[:, -1])["q_mca_l"] - 1
    rise_new = r["q_k"][i1, 0] / r["q_k"][i0, 0] - 1
    rep.check("T5g MCA flow rise, new / Ursino code", rise_new / rise_u, 0.5, 1.5, "ratio")
    r["rise_ursino"], r["rise_new"] = rise_u, rise_new
    r["marks"], r["span"] = (i0, i1, i2), (t_on, t_off)
    common_checks(rep, "CO2", r)
    return r


def test_pulsatile(rep, m):
    pa = lambda t: 100.0 + 10.0 * np.sin(2 * np.pi * 1.2 * t)
    r = m.simulate(Inputs(Pa=pa), 30.0, t_eval=np.arange(0, 30, 0.005), max_step=0.02, rtol=1e-6, atol=1e-8)
    sel = r["t"] >= 20.0
    rep.info("T6 pulsatile: ICP pulse peak-to-peak", np.ptp(r["ICP"][sel]), 0.3, 5.0, "mmHg")
    rep.info("T6 pulsatile: stressed CBV pulse p-p", np.ptp(r["Vblood"][sel]), 0.0, np.inf, "mL")
    common_checks(rep, "puls", r)
    return r


def test_lit(rep, m):
    """Lumbar infusion test: 1.5 mL/min into compartment 5 for 10 min (Vanek thesis p. 62)."""
    rate, t_on, t_off, t_end = 1.5 / 60.0, 120.0, 720.0, 3120.0
    plateau = m.Pvs_n + (m.S1 + rate) / m.alpha

    long = m.simulate(Inputs(infusion=lambda t: np.array([0, 0, 0, 0, rate])), 5000.0, t_eval=[5000.0], max_step=20.0)
    rep.check("L1 LIT plateau: |ICP - 23.11|", abs(long["ICP"][-1] - plateau), 0, 0.1, "mmHg")
    plateau_sim = float(long["ICP"][-1])

    inf = lambda t: np.array([0, 0, 0, 0, pulse(t, t_on, t_off, t_end, 0, rate, 5)])
    r = m.simulate(Inputs(infusion=inf), t_end, t_eval=np.linspace(0, t_end, 3121), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = marks(t, t_on, t_off)
    rep.check("L2 ICP at end of infusion", r["ICP"][i1], r["ICP"][i0] + 5.0, 23.2, "mmHg")
    rep.check("L3 max |CBF/CBF_rest - 1|", np.max(np.abs(r["CBF"] / r["CBF"][i0] - 1)), 0, 0.10, "rel")
    rep.check("L4 |stressed CBV change| at end of infusion", abs(r["Vblood"][i1] - r["Vblood"][i0]), 0, 3.0, "mL")
    rep.check("L6 recovery: ICP", r["ICP"][i2] - r["ICP"][i0], -0.5, 0.5, "mmHg")
    rep.check("L6 recovery: CBV", r["Vblood"][i2] / r["Vblood"][i0] - 1, -0.02, 0.02, "rel")
    common_checks(rep, "LIT", r)
    r["marks"], r["span"], r["plateau"], r["plateau_sim"] = (i0, i1, i2), (t_on, t_off), plateau, plateau_sim
    return r


def test_stroke(rep, m):
    """Occlusion of the left MCA: R_MCA,L x 1000 for 30 min, then reopened."""
    t_on, t_off, t_end = 120.0, 1920.0, 3120.0
    occ = lambda t: np.array([pulse(t, t_on, t_off, t_end, 1, 1000, 5), 1.0, 1.0, 1.0])
    r = m.simulate(Inputs(occlusion=occ), t_end, t_eval=np.linspace(0, t_end, 3121), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = marks(t, t_on, t_off)
    q0, q1 = r["q_k"][i0], r["q_k"][i1]
    rep.check("S1 residual flow MCA_L", q1[0] / q0[0], 0.02, 0.60, "rel")
    rep.check("S2 stressed volume MCA_L", r["Va"][i1, 0] / r["Va"][i0, 0] - 1, -0.70, -0.05, "rel")
    rep.check("S3 right-side flows: max |change|", np.max(np.abs(q1[2:] / q0[2:] - 1)), 0, 0.05, "rel")
    rep.check("S3 rest_L flow: |change|", abs(q1[1] / q0[1] - 1), 0, 0.10, "rel")
    rep.check("S4 total CBF", r["CBF"][i1] / r["CBF"][i0] - 1, -0.30, -0.10, "rel")
    rep.check("S5 ICP at end of occlusion", r["ICP"][i1] - r["ICP"][i0], -0.5, 0.5, "mmHg")
    back = after_reopening(r, q0, i0, i2)
    rep.check("S6 after reopening: max |change|", back, 0, 0.02, "rel")
    common_checks(rep, "stroke", r, symmetric=False)
    r["marks"], r["span"] = (i0, i1, i2), (t_on, t_off)
    return r


def test_co2_5pct(rep, m):
    """5 % CO2 inhalation, taken as PaCO2 = 47.5 mmHg (student's figure, not from the project sources)."""
    t_on, t_off, t_end = 120.0, 720.0, 1920.0
    pco2 = lambda t: pulse(t, t_on, t_off, t_end, 40, 47.5, 10)
    r = m.simulate(Inputs(PaCO2=pco2), t_end, t_eval=np.linspace(0, t_end, 1921), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = marks(t, t_on, t_off)
    rep.check("C1 CBF rise, 47.5 mmHg", r["CBF"][i1] / r["CBF"][i0] - 1, 0.10, 0.55, "rel")
    last = r["CBF"][(t >= t_off - 100) & (t <= t_off - 1)]
    rep.check("C2 chatter: CBF spread over last 100 s", np.ptp(last) / np.mean(last), 0, 0.01, "rel")
    rep.check("C3 recovery: CBF", r["CBF"][i2] / r["CBF"][i0] - 1, -0.02, 0.02, "rel")
    rep.check("C3 recovery: CBV", r["Vblood"][i2] / r["Vblood"][i0] - 1, -0.02, 0.02, "rel")
    common_checks(rep, "CO2 5%", r)
    r["marks"], r["span"] = (i0, i1, i2), (t_on, t_off)
    return r


def test_ica(rep, m):
    """I1: occlusion of the left ICA. The left side is then fed only through R_com."""
    t_on, t_off, t_end = 120.0, 1920.0, 3120.0
    occ = lambda t: np.array([1.0, 1.0, pulse(t, t_on, t_off, t_end, 1, 1000, 5), 1.0])
    r = m.simulate(Inputs(occlusion=occ), t_end, t_eval=np.linspace(0, t_end, 3121), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = marks(t, t_on, t_off)
    q0, q1 = r["q_k"][i0], r["q_k"][i1]
    rep.check("I1 left-side flows, smallest fraction of rest", np.min(q1[:2] / q0[:2]), 0.20, 1.00, "rel")
    rep.check("I1 left-side flows, largest fraction of rest", np.max(q1[:2] / q0[:2]), 0.20, 1.00, "rel")
    rep.check("I1 right-side flows: max |change|", np.max(np.abs(q1[2:] / q0[2:] - 1)), 0, 0.10, "rel")
    rep.check("I1 stressed CBV change", r["Vblood"][i1] - r["Vblood"][i0], -5.0, 0.0, "mL")
    back = after_reopening(r, q0, i0, i2)
    rep.check("I1 after reopening: max |change|", back, 0, 0.02, "rel")
    common_checks(rep, "ICA", r, symmetric=False)
    r["marks"], r["span"] = (i0, i1, i2), (t_on, t_off)
    return r


SA_PARAMS = ("G_CO2", "G_aut", "sat2", "sat1", "C_n_total", "k_ven", "R_vs1", "R_pv", "alpha", "S1", "R_c", "R_com",
             "V_vn", "k_E")


def quick_outputs(m):
    """Stressed CBV change in the three scenarios, shortened protocols (no recovery phase)."""
    out = {}
    pco2 = lambda t: ramp(t, [(0, 40), (120, 40), (130, 47.5), (720, 47.5)])
    r = m.simulate(Inputs(PaCO2=pco2), 720.0, t_eval=[119.0, 719.0], max_step=5.0)
    out["CO2"] = r["Vblood"][1] - r["Vblood"][0]
    inf = lambda t: np.array([0, 0, 0, 0, ramp(t, [(0, 0), (120, 0), (125, 1.5 / 60.0), (720, 1.5 / 60.0)])])
    r = m.simulate(Inputs(infusion=inf), 720.0, t_eval=[119.0, 719.0], max_step=5.0)
    out["LIT"] = r["Vblood"][1] - r["Vblood"][0]
    occ = lambda t: np.array([ramp(t, [(0, 1), (120, 1), (125, 1000), (1020, 1000)]), 1.0, 1.0, 1.0])
    r = m.simulate(Inputs(occlusion=occ), 1020.0, t_eval=[119.0, 1019.0], max_step=5.0)
    out["stroke"] = r["Vblood"][1] - r["Vblood"][0]
    return out


def sensitivity_params(step=0.2):
    """SA (spec 7.8): one parameter at a time, x(1 - step) and x(1 + step); the model re-calibrates its own rest."""
    base_m = NewModel()
    base = quick_outputs(base_m)
    rows = []
    for name in SA_PARAMS:
        p0 = getattr(base_m, name)
        lo = quick_outputs(NewModel(**{name: p0 * (1 - step)}))
        hi = quick_outputs(NewModel(**{name: p0 * (1 + step)}))
        rows.append((name, p0, {k: (hi[k] - lo[k]) / (2 * step * base[k]) for k in base}, lo, hi))
        print(f"  {name:<10} done", flush=True)
    return base, rows


def beat_pp(t, x, period):
    """Median peak-to-peak of x over consecutive windows of one cardiac period."""
    n = int((t[-1] - t[0]) / period)
    pp = [np.ptp(x[(t >= t[0] + k * period) & (t < t[0] + (k + 1) * period)]) for k in range(n)]
    return float(np.median(pp))


def run_recording(m, path):
    """Drive the model with Stepanik's recorded arterial pressure; compare the cardiac ICP pulse (spec 7.5, K2)."""
    from scipy.io import loadmat
    d = loadmat(path, squeeze_me=True, struct_as_record=False)["inputs"]
    tr, pa, icp = np.asarray(d.time, float), np.asarray(d.pA, float), np.asarray(d.icp, float)
    f = np.fft.rfftfreq(len(tr), tr[1] - tr[0])
    amp = np.abs(np.fft.rfft(pa - pa.mean()))
    band = (f > 0.7) & (f < 3.0)
    period = 1.0 / f[band][np.argmax(amp[band])]

    steady = m.simulate(Inputs(Pa=float(pa.mean())), 900.0, t_eval=[900.0], max_step=20.0)
    T = float(tr[-1])
    r = m.simulate(Inputs(Pa=lambda t: float(np.interp(t, tr, pa))), T, y0=steady["y"][:, -1],
                   t_eval=np.arange(0.0, T, 0.004), max_step=0.01, rtol=1e-6, atol=1e-8)
    sel_m, sel_r = r["t"] >= 10.0, tr >= 10.0
    out = dict(period=period, Pa_mean=float(pa.mean()), Pa_pp=beat_pp(tr[sel_r], pa[sel_r], period),
               icp_rec_pp=beat_pp(tr[sel_r], icp[sel_r], period), icp_rec_mean=float(icp[sel_r].mean()),
               icp_mod_pp=beat_pp(r["t"][sel_m], r["ICP"][sel_m], period), icp_mod_mean=float(r["ICP"][sel_m].mean()),
               cbv_pp=beat_pp(r["t"][sel_m], r["Vblood"][sel_m], period), t=r["t"], icp_mod=r["ICP"], tr=tr, icp_rec=icp)
    out["ratio"] = out["icp_mod_pp"] / out["icp_rec_pp"]
    return out


def sensitivity_phi_v(rep, base, values=(1.0, 0.5)):
    """V1 (spec 7.5): do the results depend on which CSF space the veins push on?
    base = runs already made with phi_v = 1. The phi_v = 0.5 runs are stiff and slow (minutes).
    Result of 10 Nov (spec 7.6): phi_v = 0.5 makes P1 unstable in the stroke run; phi_a = phi_v = 1 is fixed."""
    res = {values[0]: base}
    for pv in values[1:]:
        m = NewModel(phi_v=pv)
        dummy = Report()
        res[pv] = {"5% CO2": test_co2_5pct(dummy, m), "LIT": test_lit(dummy, m), "stroke": test_stroke(dummy, m)}
        res[pv]["fails"] = sum(1 for row in dummy.rows if row[5] == "FAIL")
    rows = []
    for name in ("5% CO2", "LIT", "stroke"):
        d = []
        for pv in values:
            r = res[pv][name]; i0, i1, _ = r["marks"]
            d.append((r["Vblood"][i1] - r["Vblood"][i0], r["ICP"].max(), r["ICP"].min()))
        rep.check(f"V1 {name}: CBV change, phi_v {values[1]} vs {values[0]}", abs(d[1][0] / d[0][0] - 1), 0, 0.05, "rel")
        rows.append((name, d))
    return rows, res


def summary(name, r, m):
    i0, i1, i2 = r["marks"]
    print(f"\n=== {name} ===   (rest -> end of intervention)")
    print(f"CBF total      {r['CBF'][i0]:6.2f} -> {r['CBF'][i1]:6.2f} mL/s  ({100 * (r['CBF'][i1] / r['CBF'][i0] - 1):+.1f} %)")
    for j, n in enumerate(NAMES):
        print(f"  {n}: flow {r['q_k'][i0, j]:5.2f} -> {r['q_k'][i1, j]:5.2f} mL/s ({100 * (r['q_k'][i1, j] / r['q_k'][i0, j] - 1):+6.1f} %)"
              f"   volume {r['Va'][i0, j]:5.2f} -> {r['Va'][i1, j]:5.2f} mL ({100 * (r['Va'][i1, j] / r['Va'][i0, j] - 1):+6.1f} %)")
    print(f"venous V       {r['Vv'][i0]:6.2f} -> {r['Vv'][i1]:6.2f} mL")
    print(f"stressed CBV   {r['Vblood'][i0]:6.2f} -> {r['Vblood'][i1]:6.2f} mL  ({r['Vblood'][i1] - r['Vblood'][i0]:+.2f} mL)")
    print(f"ICP            rest {r['ICP'][i0]:.2f}, max {r['ICP'].max():.2f}, end of intervention {r['ICP'][i1]:.2f}, "
          f"min {r['ICP'].min():.2f}, end of run {r['ICP'][i2]:.2f} mmHg")
    print(f"CSF volumes    cranial {r['V1'][i1] + r['V2'][i1] - r['V1'][i0] - r['V2'][i0]:+.2f} mL, "
          f"spinal {r['dVspinal'][i1] - r['dVspinal'][i0]:+.2f} mL")


# ----------------------------------------------------------------------------------------------------------------------
# Fifth step (spec 7.10): new outputs, switchable options, LIT fitting tool
# ----------------------------------------------------------------------------------------------------------------------
def test_outputs(rep, r_rest, r_co2, r_stroke):
    """O1-O4: CBV in percent and MCA velocity."""
    rep.check("O1 rest: MCA velocity", float(np.mean(r_rest["v_MCA"][:, 0])), 60.4, 61.4, "cm/s")
    rep.check("O1 rest: max |CBV_pct|", float(np.max(np.abs(r_rest["CBV_pct"]))), 0, 1e-6, "%")
    i0, i1, _ = r_co2["marks"]
    expected = 100.0 * (r_co2["Vblood"][i1] - r_co2["Vblood"][i0]) / 56.0
    rep.check("O2 5 % CO2: CBV_pct vs 100*dV/56", abs(r_co2["CBV_pct"][i1] - expected), 0, 1e-6, "%")
    rep.check("O3 5 % CO2: MCA velocity rise", r_co2["v_MCA"][i1, 0] / r_co2["v_MCA"][i0, 0] - 1, 0.30, 0.55, "rel")
    j0, j1, _ = r_stroke["marks"]
    rep.check("O4 occlusion: MCA_L velocity / rest", r_stroke["v_MCA"][j1, 0] / r_stroke["v_MCA"][j0, 0], 0, 0.15, "rel")


def conservation_fails(rep_local):
    return [row[0] for row in rep_local.rows if any(t in row[0] for t in ("T2", "T3", "T7")) and row[5] == "FAIL"]


def test_venous_squeeze(rep, base_runs):
    """V0-V4 (spec 7.10): venous squeeze option kappa_v."""
    kappas = (0.0, 0.05, 0.1, 0.2)
    res, fails, scen_fails = {}, [], {}
    for k in kappas:
        d = Report()
        res[k] = test_lit(d, NewModel(kappa_v=k))
        fails += conservation_fails(d)
        scen_fails[k] = [row[0] for row in d.rows if row[5] == "FAIL" and row[0].startswith("L")]
    rep.check("V0 kappa_v = 0: LIT CBV change vs default", abs(dcbv(res[0.0]) - dcbv(base_runs[SC_LIT])), 0, 1e-9, "mL")
    rep.check("V1 kappa_v = 0.1: LIT stressed CBV change", dcbv(res[0.1]), -1.0, 0.0, "mL")
    vals = [dcbv(res[k]) for k in kappas]
    rep.check("V2 LIT CBV change strictly decreasing with kappa_v", float(all(np.diff(vals) < 0)), 1, 1, "")
    d = Report()
    co2 = test_co2_5pct(d, NewModel(kappa_v=0.1)); fails += conservation_fails(d)
    d = Report()
    stroke = test_stroke(d, NewModel(kappa_v=0.1)); fails += conservation_fails(d)
    rep.check("V3 kappa_v = 0.1: 5 % CO2 CBV change vs default", dcbv(co2) - dcbv(base_runs[SC_CO2_5]), -0.5, 0.5, "mL")
    rep.check("V3 kappa_v = 0.1: stroke CBV change vs default", dcbv(stroke) - dcbv(base_runs[SC_STROKE]), -0.5, 0.5, "mL")
    rep.check("V4 conservation checks failing in all runs", float(len(fails)), 0, 0, "")
    return res, co2, stroke, scen_fails


def onset_metrics(m):
    """Onset fraction at 10 s and steady CBF rise for a CO2 step; steady CBF fall for a pressure step."""
    r = m.simulate(Inputs(PaCO2=lambda t: ramp(t, [(0, 40), (1, 60)])), 600.0, t_eval=[0.0, 10.0, 600.0], max_step=2.0)
    rise = r["CBF"][2] / r["CBF"][0] - 1
    onset = (r["CBF"][1] - r["CBF"][0]) / (r["CBF"][2] - r["CBF"][0])
    r = m.simulate(Inputs(Pa=lambda t: ramp(t, [(0, 100), (1, 80)])), 600.0, t_eval=[0.0, 600.0], max_step=2.0)
    return onset, rise, 1.0 - r["CBF"][1] / r["CBF"][0]


def test_sigmoid_width(rep, scales=(1.0, 3.0, 10.0, 30.0)):
    """S0-S2 (spec 7.10): width of the C(x) sigmoid."""
    rows = [(sc,) + onset_metrics(NewModel(kC_scale=sc)) for sc in scales]
    dflt = onset_metrics(NewModel())
    rep.check("S0 kC_scale = 1: onset fraction vs default", abs(rows[0][1] - dflt[0]), 0, 1e-9, "")
    rep.check("S1 onset fraction strictly decreasing with kC_scale", float(all(np.diff([r[1] for r in rows]) < 0)), 1, 1, "")
    rep.check("S2 steady CBF fall at Pa = 80 non-decreasing with kC_scale", float(all(np.diff([r[3] for r in rows]) >= -1e-9)), 1, 1, "")
    return rows


def _argval(argv, name, default):
    return float(argv[argv.index(name) + 1]) if name in argv else default


PARTS = ("rest", "co2", "co2_5pct", "lit", "stroke", "pulsatile", "ica", "cross", "outputs")


def main(argv=(), parts=None):
    """argv: the old flags (--calibrate, --fit, ...). parts: subset of PARTS to run of the default run (run.py new
    --rest --lit ...); None = all of it. Stages need the whole default run, so they ignore parts."""
    if any(a in ETCO2_FLAGS for a in argv):   # opt-in scenario of spec 7.17: runs on its own, not with other flags
        return main_etco2(argv, parts)
    if "--fit-mismatch" in argv:        # stand-alone stage (spec 7.13), does not need the base runs
        repm = Report()
        out = test_fit_mismatch(repm)
        print("=== M: fit with the default model, data from a model with venous squeeze (spec 7.13) ===")
        print(f"truth: R_out {LIT_TRUTH[0]}, spinal scale {LIT_TRUTH[1]}, P0 {LIT_TRUTH[2]}")
        for f in out:
            print(f"kappa_v {f['kappa']}: R_out {f['theta'][0]:.2f} (SE {100 * f['se_rel'][0]:.1f} %), scale {f['theta'][1]:.2f} "
                  f"(SE {100 * f['se_rel'][1]:.1f} %), P0 {f['theta'][2]:.2f} (SE {f['se_P0']:.2f}), rms {f['rms']:.2f}, nfev {f['nfev']}")
        return repm.show()
    do_cal = "--calibrate" in argv or "--all" in argv
    do_sens = "--sensitivity" in argv or "--all" in argv
    stage = any(f in argv for f in ("--calibrate", "--sensitivity", "--params", "--options", "--fit", "--fit-csv",
                                    "--all"))
    sel = set(parts) if parts and not stage else set(PARTS)
    if "cross" in sel:
        sel |= {"stroke", "lit", "co2_5pct", "ica"}
    if "outputs" in sel:
        sel |= {"rest", "co2_5pct", "stroke"}
    m = NewModel()
    print("=== New model prototype: resting state (spec 4.2) ===")
    print(f"ICP (P2) {m.P2_n:.3f}  P1 {m.P1_n:.3f}  Pv {m.Pv_n:.3f}  Pc {m.Pc_n:.3f}  P_N {m.PN_n:.2f}  P_k {m.Pk_n:.2f} mmHg")
    print(f"R_n (MCA, rest) {m.R_n[0]:.2f}, {m.R_n[1]:.2f}   R_rest {m.R_rest:.3f}   R_c {m.R_c:.1f}   R_com {m.R_com:.1f}")
    print(f"V_n arterial {m.V_n.round(3)} (total {m.V_n.sum():.2f} mL)   V_vn {m.V_vn} mL")

    rep = Report()
    r_rest = test_rest(rep, m) if "rest" in sel else None
    runs = {}
    for part, name, test, fname in (("co2", SC_CO2_60, test_co2, "new_model_co2.png"),
                                    ("co2_5pct", SC_CO2_5, test_co2_5pct, "new_model_co2_5pct.png"),
                                    ("lit", SC_LIT, test_lit, "new_model_lit.png"),
                                    ("stroke", SC_STROKE, test_stroke, "new_model_stroke.png")):
        if part in sel:
            runs[name] = (test(rep, m), fname)
    if "pulsatile" in sel:
        test_pulsatile(rep, m)
    if "cross" in sel:
        # fourth step (spec 7.8): cross-checks against the source models and ICA occlusion
        urs, new = test_cross_stroke(rep, runs[SC_STROKE][0])
        van = test_cross_lit(rep, runs[SC_LIT][0])
    if "ica" in sel:
        runs[SC_ICA] = (test_ica(rep, m), "new_model_ica.png")
    if "cross" in sel:
        # sixth step (spec 7.15): further cross-checks against the source codes
        more = test_cross_more(rep, m, runs[SC_CO2_5][0], runs[SC_LIT][0],
                               runs[SC_ICA][0], van)
    if "outputs" in sel:
        # fifth step (spec 7.10): new outputs
        test_outputs(rep, r_rest, runs[SC_CO2_5][0], runs[SC_STROKE][0])
    n_fail = rep.show()
    if "cross" in sel:
        print("\n=== X1: left MCA occlusion, new model vs existing Ursino code ===")
        print(f"residual flow in the MCA_L territory: new {100 * new['res']:.1f} %, Ursino code {100 * urs['res']:.1f} % of rest")
        print(f"stressed volume of that territory:    new {100 * new['dV']:+.1f} %, Ursino code {100 * urs['dV']:+.1f} %")
        print(f"total CBF:                            new {100 * new['dCBF']:+.1f} %, Ursino code {100 * urs['dCBF']:+.1f} %")
        print(f"arterial stressed volume, all:        new {new['dVa']:+.2f} mL, Ursino code {urs['dVa']:+.2f} mL")
        print(f"Ursino code ICP: {urs['icp'][0]:.2f} -> {urs['icp'][1]:.2f} mmHg")
        print("\n=== X2: lumbar infusion, new model vs existing Vanek code (constant inputs) ===")
        lit = runs[SC_LIT][0]
        print(f"plateau ICP:              new {lit['plateau_sim']:.2f}, Vanek code {van['plateau']:.2f} mmHg")
        print(f"ICP rise after 10 min:    new {van['rise_new']:.2f}, Vanek code {van['rise']:.2f} mmHg (Vanek rest {van['rest']:.2f})")
        print(f"ICP rise after 5 min:     new {more['lit'][0]:.2f}, Vanek code {more['lit'][1]:.2f} mmHg (Pb = 10), {more['lit'][2]:.2f} mmHg (Pb = 12)")
        print("\n=== X3, X4: graded CO2 range and autoregulation, new model vs existing Ursino code (spec 7.15) ===")
        for name, key in (("5 % CO2 (47.5 mmHg)", "co2"), ("Pa 100 -> 80 mmHg", "pa")):
            dq_n, dq_u, dV_n, dV_u = more[key]
            print(f"{name:<20} total CBF: new {100 * dq_n:+.1f} %, Ursino code {100 * dq_u:+.1f} %;   "
                  f"arterial stressed volume: new {dV_n:+.2f} mL, Ursino code {dV_u:+.2f} mL")
        ica = more["ica"]
        print("\n=== X5 (info): left ICA occlusion, new model vs existing Ursino code. NOT validated: no basilar inlet (spec 3.2.2) ===")
        print(f"MCA_L flow:                new {100 * ica['mca'][0]:+.1f} %, Ursino code {100 * ica['mca'][1]:+.1f} %")
        print(f"rest_L flow:               new {100 * ica['rest'][0]:+.1f} %, Ursino code {100 * ica['rest'][1]:+.1f} % "
              f"(his ACA_L {100 * ica['urs_districts'][1]:+.1f} %, PCA_L {100 * ica['urs_districts'][2]:+.1f} %)")
        print(f"total CBF:                 new {100 * ica['dq'][0]:+.1f} %, Ursino code {100 * ica['dq'][1]:+.1f} %")
        print(f"arterial stressed volume:  new {ica['dV'][0]:+.2f} mL, Ursino code {ica['dV'][1]:+.2f} mL")

    fig_dir = out_dir("new")
    os.makedirs(fig_dir, exist_ok=True)
    for name, (r, fname) in runs.items():
        summary(name, r, m)
        plot_run(r, m, os.path.join(fig_dir, fname), f"New model. {name} (shaded). Pa = 100 mmHg")
    base_runs = {k: v[0] for k, v in runs.items()}

    if "co2" in sel:
        co2 = runs[SC_CO2_60][0]
        print(f"\nMCA flow rise at 60 mmHg: new model {100 * co2['rise_new']:+.1f} %, existing Ursino code {100 * co2['rise_ursino']:+.1f} %")
    print(f"Figures saved in {fig_dir}")
    if do_cal:
        n_fail += step_calibration(m, base_runs, fig_dir)
    if do_sens:
        n_fail += step_sensitivity(base_runs)
    if "--options" in argv or "--all" in argv:
        n_fail += step_options(base_runs, fig_dir)
    if "--fit" in argv or "--all" in argv:
        n_fail += step_fit(fig_dir)
    if "--fit-csv" in argv:
        fit_file(argv[argv.index("--fit-csv") + 1], _argval(argv, "--rate", 1.5), _argval(argv, "--t-on", 300.0),
                 _argval(argv, "--t-off", 900.0))
    if "--params" in argv or "--all" in argv:
        print("\n=== SA: parameter sensitivity of the stressed CBV change (spec 7.8) ===")
        base, rows = sensitivity_params()
        print(f"\nbase: CO2 {base['CO2']:+.3f} mL, LIT {base['LIT']:+.3f} mL, stroke {base['stroke']:+.3f} mL")
        print(f"{'parameter':<12}{'value':>10}{'S CO2':>9}{'S LIT':>9}{'S stroke':>10}    (S = relative change of output / relative change of parameter)")
        for name, p0, S, lo, hi in rows:
            print(f"{name:<12}{p0:>10.4g}{S['CO2']:>9.2f}{S['LIT']:>9.2f}{S['stroke']:>10.2f}")
    return n_fail


def step_options(base_runs, fig_dir):
    """Fifth step (spec 7.10): venous squeeze and sigmoid width."""
    rep3 = Report()
    res, co2, stroke, scen_fails = test_venous_squeeze(rep3, base_runs)
    print("\n=== V: venous squeeze kappa_v (spec 7.10) ===")
    print(f"{'kappa_v':>8}{'LIT dCBV [mL]':>15}{'arterial':>10}{'venous':>9}{'CBF change':>12}{'ICP end [mmHg]':>16}   LIT scenario bounds failing")
    for k, r in res.items():
        i0, i1, _ = r["marks"]
        print(f"{k:>8}{dcbv(r):>15.2f}{r['Va_total'][i1] - r['Va_total'][i0]:>10.2f}{r['Vv'][i1] - r['Vv'][i0]:>9.2f}"
              f"{100 * (r['CBF'][i1] / r['CBF'][i0] - 1):>11.1f}%{r['ICP'][i1]:>16.2f}   {scen_fails[k] or '-'}")
    print(f"kappa_v = 0.1: 5 % CO2 dCBV {dcbv(co2):+.2f} mL (default {dcbv(base_runs[SC_CO2_5]):+.2f}), "
          f"stroke dCBV {dcbv(stroke):+.2f} mL (default {dcbv(base_runs[SC_STROKE]):+.2f})")
    plot_run(res[0.1], NewModel(kappa_v=0.1), os.path.join(fig_dir, "new_model_lit_squeeze.png"),
             "New model, venous squeeze kappa_v = 0.1. Lumbar infusion test (shaded)")
    rows = test_sigmoid_width(rep3)
    print("\n=== S: sigmoid width kC_scale (spec 7.10) ===")
    print(f"{'kC_scale':>9}{'onset at 10 s':>15}{'CBF rise, 60 mmHg':>19}{'CBF fall, Pa 80':>17}")
    for sc, onset, rise, fall in rows:
        print(f"{sc:>9.0f}{100 * onset:>14.0f}%{100 * rise:>18.1f}%{100 * fall:>16.1f}%")
    return rep3.show()


def step_fit(fig_dir):
    """Fifth step (spec 7.10): LIT fitting tool on synthetic data."""
    import time
    t0 = time.time()
    rep4 = Report()
    f1, noisy, flong = test_fit(rep4)
    print("\n=== F: LIT fit on synthetic data (spec 7.10) ===")
    print(f"truth: R_out {LIT_TRUTH[0]} mmHg min/mL, spinal scale {LIT_TRUTH[1]}, P0 {LIT_TRUTH[2]} mmHg; start of fit {np.round(LIT_DEFAULT, 2)}")
    print(f"{'run':<22}{'R_out':>8}{'(SE %)':>8}{'scale':>8}{'(SE %)':>8}{'P0':>8}{'(SE)':>7}{'rms':>7}{'nfev':>6}")
    for name, f in [("noiseless", f1)] + [(f"noisy seed {f['seed']}", f) for f in noisy] + [("30-min infusion", flong)]:
        print(f"{name:<22}{f['theta'][0]:>8.2f}{100 * f['se_rel'][0]:>8.1f}{f['theta'][1]:>8.2f}{100 * f['se_rel'][1]:>8.1f}"
              f"{f['theta'][2]:>8.2f}{f['se_P0']:>7.2f}{f['rms']:>7.2f}{f['nfev']:>6}")
    errs = np.array([[f["theta"][0] / LIT_TRUTH[0] - 1, f["theta"][1] / LIT_TRUTH[1] - 1, f["theta"][2] - LIT_TRUTH[2]] for f in noisy])
    print(f"over {len(noisy)} seeds: R_out error {100 * errs[:, 0].mean():+.1f} % (sd {100 * errs[:, 0].std():.1f} %), "
          f"scale error {100 * errs[:, 1].mean():+.1f} % (sd {100 * errs[:, 1].std():.1f} %), P0 error {errs[:, 2].mean():+.2f} mmHg (sd {errs[:, 2].std():.2f})")
    plot_fit(noisy[0], os.path.join(fig_dir, "new_model_fit.png"))
    print(f"fit stage took {time.time() - t0:.0f} s")
    return rep4.show()


def step_calibration(m, base_runs, fig_dir):
    """Third step (spec 7.5): default (storage calibrated to Ursino) vs Vanek's raw values (K1), data-file check (K2)."""
    n_fail = 0
    mv = NewModel.vanek_values()
    mc = m
    print(f"\n=== Storage calibration (spec 7.5) ===")
    print(f"storage at rest: Vanek values {mv.storage_at_rest():.3f} mL/mmHg, default (Ursino target) {mc.storage_at_rest():.3f} mL/mmHg, "
          f"scale s = {mc.c_sp_scale:.3f}")
    rep_v = Report()
    test_rest(rep_v, mv)
    van_runs = {
        SC_CO2_60: test_co2(rep_v, mv),
        SC_CO2_5: test_co2_5pct(rep_v, mv),
        SC_LIT: test_lit(rep_v, mv),
        SC_STROKE: test_stroke(rep_v, mv),
    }
    test_pulsatile(rep_v, mv)
    print("K1: all checks again with Vanek's raw spinal compliances")
    n_fail += rep_v.show()
    print(f"\n{'scenario':<50}{'dCBV [mL]':>22}{'ICP max [mmHg]':>22}{'ICP min [mmHg]':>22}")
    print(f"{'':<50}{'Vanek':>11}{'default':>11}{'Vanek':>11}{'default':>11}{'Vanek':>11}{'default':>11}")
    for k in van_runs:
        a_, c_ = van_runs[k], base_runs[k]
        i0, i1, _ = a_["marks"]
        print(f"{k:<50}{a_['Vblood'][i1] - a_['Vblood'][i0]:>11.2f}{c_['Vblood'][i1] - c_['Vblood'][i0]:>11.2f}"
              f"{a_['ICP'].max():>11.2f}{c_['ICP'].max():>11.2f}{a_['ICP'].min():>11.2f}{c_['ICP'].min():>11.2f}")
    m = mv   # below: "m" = Vanek values, "mc" = default

    rep2 = Report()
    rec_path = ZBYSEK_DATA
    if os.path.exists(rec_path):
        vc, vb = run_recording(mc, rec_path), run_recording(m, rec_path)
        print(f"\n=== K2: Stepanik's data file, one beat repeated ({rec_path}) ===")
        print(f"cardiac period {vc['period']:.3f} s, mean P_A {vc['Pa_mean']:.1f} mmHg, arterial pulse {vc['Pa_pp']:.1f} mmHg p-p")
        print(f"ICP pulse p-p: data file {vc['icp_rec_pp']:.2f} mmHg | default model {vc['icp_mod_pp']:.2f} | Vanek values {vb['icp_mod_pp']:.2f}")
        print(f"ICP mean:      data file {vc['icp_rec_mean']:.2f} mmHg | default model {vc['icp_mod_mean']:.2f} | Vanek values {vb['icp_mod_mean']:.2f}")
        print(f"stressed CBV pulse p-p: default {vc['cbv_pp']:.2f} mL | Vanek values {vb['cbv_pp']:.2f} mL")
        rep2.check("K2 ICP pulse, default model / data file", vc["ratio"], 0.5, 2.0, "ratio")
        rep2.info("K2 ICP pulse, Vanek values / data file", vb["ratio"], 0.5, 2.0, "ratio")
        plot_recording(vc, vb, os.path.join(fig_dir, "new_model_recording.png"))
    else:
        print(f"\nK2 skipped: recording not found at {rec_path}")

    return n_fail + rep2.show()


def step_sensitivity(base_runs):
    """Third step (spec 7.5): V1, phi_v = 1 vs 0.5."""
    rep2 = Report()
    base = {"5% CO2": base_runs[SC_CO2_5], "LIT": base_runs[SC_LIT],
            "stroke": base_runs[SC_STROKE]}
    rows, res = sensitivity_phi_v(rep2, base)
    print("\n=== V1: phi_v sensitivity ===")
    print(f"{'scenario':<10}{'dCBV phi_v=1':>14}{'phi_v=0.5':>11}{'ICP max 1':>11}{'0.5':>8}{'ICP min 1':>11}{'0.5':>8}")
    for name, d in rows:
        print(f"{name:<10}{d[0][0]:>14.3f}{d[1][0]:>11.3f}{d[0][1]:>11.2f}{d[1][1]:>8.2f}{d[0][2]:>11.2f}{d[1][2]:>8.2f}")
    print(f"pre-set scenario bounds failed with phi_v = 0.5: {res[0.5]['fails']}")
    return rep2.show()


# ----------------------------------------------------------------------------------------------------------------------
# Seventh step (spec 7.17): end-tidal-shaped CO2 input. Opt-in: not in the default run, not in --all.
# ----------------------------------------------------------------------------------------------------------------------
ETCO2_FIXTURE = os.path.join(ROOT, "tests", "fixtures", "etco2_placeholder.csv")
ETCO2_PAD = 120.0        # rest before the first sample of a measured curve [s] (model time 0 = first sample - pad)
ETCO2_DT = 0.5           # output sampling of the end-tidal runs [s]


def run_etco2(m, pco2, t_end, t_eval=None):
    """One run with a PaCO2(t) input, solver settings of test C (max_step 5 s, default tolerances)."""
    if t_eval is None:
        t_eval = np.linspace(0.0, t_end, int(round(t_end / ETCO2_DT)) + 1)
    return m.simulate(Inputs(PaCO2=pco2), t_end, t_eval=t_eval, max_step=5.0)


def write_etco2_fixture(path=ETCO2_FIXTURE, proto=EtCO2Protocol()):
    """Write the placeholder curve at 1-s spacing (the fixture of test E1). Regenerate it when the placeholder changes:
    python -c "from cbfcsf.checks.new import write_etco2_fixture; write_etco2_fixture()" """
    t = np.arange(0.0, proto.t_end + 0.5, 1.0)
    write_etco2_csv(path, t, etco2_curve(proto)(t))


def test_etco2_input(rep, proto=EtCO2Protocol(), taus=(5.0, 20.0, 40.0)):
    """E0, E0f (spec 7.17): the end-tidal input function against its analytic properties."""
    f = etco2_curve(proto)
    on, off, P0, dP = proto.onsets, proto.offsets, proto.P0, proto.dP
    rep.check("E0 input before 1st onset: max |P - P0|", float(np.max(np.abs(f(np.arange(0.0, on[0], 0.01)) - P0))),
              0, 0, "mmHg")
    plat = max(np.max(np.abs(f(np.arange(a + 5 * proto.tau_on, b, 0.01)) - P0 - dP)) for a, b in zip(on, off))
    rep.check("E0 plateau from t_k + 5 tau_on: max dev / dP", float(plat / dP), 0, 0.01, "rel")
    # carry-over at onsets 2, 3: closed form per cycle, written independently of the segment recursion of etco2_curve()
    a_on, a_air = np.exp(-proto.t_inhale / proto.tau_on), np.exp(-proto.t_air / proto.tau_off)
    x, carry = 0.0, 0.0                    # x = P - P0 at an onset
    for k in range(1, proto.n_cycles):
        x = (dP + (x - dP) * a_on) * a_air
        carry = max(carry, abs(f(on[k] + proto.delay) - (P0 + x + proto.gap)))
    rep.check("E0 onsets 2, 3: |P - closed-form carry-over|", carry, 0, 1e-9, "mmHg")
    back = np.max(np.abs(f(np.arange(off[-1] + 5 * proto.tau_off, proto.t_end, 0.01)) - P0)) / dP
    rep.check("E0 back to P0 by 5 tau_off: max dev / dP", float(back), 0, 0.01, "rel")
    rep.check("E0 largest step between samples 0.01 s apart", float(np.max(np.abs(np.diff(
        f(np.arange(0.0, proto.t_end, 0.01)))))), 0, 0.01, "mmHg")
    jumps = []
    for tau in taus:
        g = etco2_curve(replace(proto, tau_on=tau, tau_off=tau))
        jumps += [abs(g(ts + proto.delay + 1e-9) - g(ts + proto.delay - 1e-9)) for ts in on + off]
    rep.check("E0f continuity at switches, tau 5/20/40 s", max(jumps), 0, 1e-6, "mmHg")


def test_etco2_csv(rep, r_an, proto=EtCO2Protocol(), path=ETCO2_FIXTURE):
    """E1a, E1 (spec 7.17): the committed placeholder fixture read back, and simulated against the analytic-input run
    r_an (same protocol). Compared at equal times after the first sample (CSV run: model time - ETCO2_PAD)."""
    d = read_etco2_csv(path, "mmHg")
    f = d.function(pad=ETCO2_PAD, gap=proto.gap)
    err = np.max(np.abs(f(d.to_model_time(d.t, pad=ETCO2_PAD)) - etco2_curve(proto)(d.t)))
    rep.check("E1a fixture read back: max |P - placeholder|", float(err), 0, 1e-9, "mmHg")
    rep.check("E1a baseline (first 60 s): |baseline - P0|", abs(d.baseline - proto.P0), 0, 1e-9, "mmHg")
    m = NewModel(PaCO2_n=d.baseline + proto.gap)
    n_pad = int(round(ETCO2_PAD / ETCO2_DT))
    t_eval = np.concatenate([np.arange(n_pad) * ETCO2_DT, r_an["t"] + ETCO2_PAD])
    r = run_etco2(m, f, t_eval[-1], t_eval=t_eval)
    diff = np.max(np.abs(r["CBF"][n_pad:] / r_an["CBF"] - 1))
    rep.check("E1 CSV round trip: max relative CBF difference", float(diff), 0, 1e-3, "rel")
    common_checks(rep, "etCO2 CSV", r)
    r["E1_diff"] = r["CBF"][n_pad:] / r_an["CBF"] - 1
    return r, d


# ---- metric, scenario and command line of spec 7.17 -----------------------------------------------------------------
ETCO2_FLAGS = ("--etco2", "--etco2-csv", "--etco2-unit", "--cycles", "--cycle-length", "--barometric")
ETCO2_RISING = ("PaCO2", "CBF", "v_MCA,L", "CBV")      # the signals E5 applies to; ICP is info only
BOLD_NOTE = ("The reference time, a vascular response at its maximum within about 60 s, comes from the BOLD MRI signal. The "
             "model has no BOLD\noutput; CBF and v_MCA (~ TCD) are its nearest outputs, so the comparison is one of timing only.")


def etco2_signals(r, pco2):
    """The signals of the metric on the output grid: input PaCO2, total CBF, left MCA velocity, stressed CBV, ICP."""
    return {"PaCO2": np.asarray(pco2(r["t"]), float), "CBF": r["CBF"], "v_MCA,L": r["v_MCA"][:, 0],
            "CBV": r["Vblood"], "ICP": r["ICP"]}


def _first_crossing(t, z, thr):
    """First time at which z reaches thr, linear interpolation between samples; t[0] if z[0] already does."""
    above = z >= thr
    if not above.any():
        return np.nan
    i = int(np.argmax(above))
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (thr - z[i - 1]) / (z[i] - z[i - 1]) * (t[i] - t[i - 1]))


def etco2_metrics(t, y, onsets, length):
    """Metric of spec 7.17 for one signal, per cycle k (window [t_k, t_k + length]). Two references: 'rest' = the value
    1 s before the FIRST onset (primary, all cycles), 'own' = the value 1 s before this onset. peak = maximum in the
    window; t50, t90 = first time after t_k at which y - ref reaches 50 / 90 % of peak - ref (s after t_k); flag = the
    peak is the last sample of the window (still rising). Also the share of the rise reached at t_k + 60 s and the rise
    during the last 30 s of the window (both in % of the cycle's rise). residual = y(t_k - 1 s) - rest, as a fraction
    of the cycle-1 rise."""
    y_rest = float(np.interp(onsets[0] - 1.0, t, y))
    out = []
    for tk in onsets:
        w = (t >= tk - 1e-9) & (t <= tk + length + 1e-9)
        tw, yw = t[w], y[w]
        ipk = int(np.argmax(yw))
        c = dict(onset=tk, peak=float(yw[ipk]), t_peak=float(tw[ipk] - tk), flag=ipk == len(yw) - 1, y_rest=y_rest,
                 y_on=float(np.interp(tk - 1.0, t, y)), y_end=float(np.interp(tk + length - 1.0, t, y)))
        for key, ref in (("rest", y_rest), ("own", c["y_on"])):
            rise = c["peak"] - ref
            d = dict(rise=rise, t50=np.nan, t90=np.nan, share60=np.nan, last30=np.nan)
            if rise > 0:
                d["t50"] = _first_crossing(tw, yw - ref, 0.5 * rise) - tk
                d["t90"] = _first_crossing(tw, yw - ref, 0.9 * rise) - tk
                d["share60"] = (np.interp(tk + 60.0, t, y) - ref) / rise
                d["last30"] = (np.interp(tk + length, t, y) - np.interp(tk + length - 30.0, t, y)) / rise
            c[key] = d
        out.append(c)
    rise1 = out[0]["rest"]["rise"]
    for c in out:
        c["residual"] = (c["y_on"] - y_rest) / rise1 if rise1 > 0 else np.nan
    return out


def etco2_sanity(met, length):
    """E5: number of (signal, cycle, reference) with a rise where t50 < t90 <= length does not hold."""
    bad = 0
    for k in ETCO2_RISING:
        for c in met[k]:
            for ref in ("rest", "own"):
                d = c[ref]
                if d["rise"] > 0 and not (d["t50"] < d["t90"] <= length + 1e-9):
                    bad += 1
    return bad


def etco2_case(rep, m, label, proto, pco2, t_eval=None, onsets=None, t_end=None):
    """One run of the scenario with its E2 (conservation) and E5 (sanity) checks and the metric of every signal."""
    t_end = proto.t_end if t_end is None else t_end
    onsets = proto.onsets if onsets is None else onsets
    r = run_etco2(m, pco2, t_end, t_eval=t_eval)
    common_checks(rep, f"etCO2 {label}", r)
    sig = etco2_signals(r, pco2)
    met = {k: etco2_metrics(r["t"], v, onsets, proto.t_inhale) for k, v in sig.items()}
    rep.check(f"E5 {label}: cases not t50 < t90 <= {proto.t_inhale:g} s", float(etco2_sanity(met, proto.t_inhale)),
              0, 0, "")
    return dict(r=r, sig=sig, met=met, proto=proto, onsets=onsets, label=label)


def test_co2_etco2(rep, m, proto=EtCO2Protocol(), taus=(5.0, 40.0), airs=(30.0, 60.0)):
    """Spec 7.17: 5 % CO2 inhaled 3 x 180 s with the end-tidal-shaped input (PLACEHOLDER shape), the same protocol with
    the current switch-like input (10-s ramps), and information-only sweeps of tau and of the air interval.
    Checks E0, E0f, E1a, E1, E2 (every run), E3, E4, E6 (main run), E5 (every run); E7, E8 informational."""
    test_etco2_input(rep, proto)                                              # E0, E0f
    main_ = etco2_case(rep, m, "shaped", proto, etco2_curve(proto))
    step = etco2_case(rep, m, "step", proto, etco2_step(proto))
    tau_runs = {proto.tau_on: main_}
    for tau in taus:
        pr = replace(proto, tau_on=tau, tau_off=tau)
        tau_runs[tau] = etco2_case(rep, m, f"tau {tau:g} s", pr, etco2_curve(pr))
    air_runs = {proto.t_air: main_}
    for air in airs:
        pr = replace(proto, t_air=air)
        air_runs[air] = etco2_case(rep, m, f"air {air:g} s", pr, etco2_curve(pr))
    r_csv, d = test_etco2_csv(rep, main_["r"], proto)                         # E1a, E1 (+ E2 on that run)
    csv_on = list(d.to_model_time(proto.onsets, pad=ETCO2_PAD))
    csv_met = {k: etco2_metrics(r_csv["t"], v, csv_on, proto.t_inhale)
               for k, v in etco2_signals(r_csv, d.function(pad=ETCO2_PAD, gap=proto.gap)).items()}
    rep.check(f"E5 CSV round trip: cases not t50 < t90 <= {proto.t_inhale:g} s",
              float(etco2_sanity(csv_met, proto.t_inhale)), 0, 0, "")

    r, t, cbf = main_["r"], main_["r"]["t"], main_["met"]["CBF"]
    rest = cbf[0]["y_rest"]
    for k, t_off in enumerate(proto.offsets):
        rep.check(f"E3 CBF rise at end of inhalation {k + 1}", float(np.interp(t_off - 1.0, t, r["CBF"]) / rest - 1),
                  0.10, 0.55, "rel")
    for k in range(1, proto.n_cycles):
        rep.check(f"E4 peak CBF rise, cycle {k + 1} / cycle 1 - 1",
                  cbf[k]["rest"]["rise"] / cbf[0]["rest"]["rise"] - 1, -0.05, 0.05, "rel")
    cbv_rest = main_["met"]["CBV"][0]["y_rest"]
    rep.check("E6 recovery: CBF", float(r["CBF"][-1] / rest - 1), -0.02, 0.02, "rel")
    rep.check("E6 recovery: CBV", float(r["Vblood"][-1] / cbv_rest - 1), -0.02, 0.02, "rel")
    for key, name in (("CBF", "CBF"), ("v_MCA,L", "v_MCA"), ("CBV", "CBV")):
        rep.info(f"E7 shaped: {name} t90 in cycle 1 (vs ~60 s)", main_["met"][key][0]["rest"]["t90"], 0, 60, "s")
    rep.info("E8 step: CBF t90 in cycle 1 (vs ~60 s)", step["met"]["CBF"][0]["rest"]["t90"], 0, 60, "s")
    return dict(main=main_, step=step, taus=tau_runs, airs=air_runs, csv=r_csv)


def _f(v, w=7, d=1):
    return f"{v:{w}.{d}f}" if np.isfinite(v) else f"{'-':>{w}}"


def print_etco2_table(title, case, m, clock_shift=0.0):
    """One row per cycle x signal (spec 7.17). clock_shift = model time - printed time of the onsets."""
    met = case["met"]
    print(f"\n=== {title} ===")
    print("Times in s after each onset. Reference 'rest' (primary) = 1 s before the FIRST onset; 'own' = 1 s before this "
          "onset.\nresid = value 1 s before the onset minus rest, in % of the cycle-1 rise. lag = t90 - t90 of the input "
          "(same reference).\nt peak with * = peak at the end of the window (still rising, no maximum reached). "
          "at60 = share of the rise\nreached 60 s after the onset; last30 = rise during the last 30 s of the window "
          "(both in % of the rise, rest reference).")
    print(f"{'cycle':<11}{'signal':<9}{'resid%':>7}{'t50':>7}{'t90':>7}{'lag':>7}{'t50own':>8}{'t90own':>8}{'lag own':>8}"
          f"{'t peak':>8} {'at60%':>7}{'last30%':>8}  rise (rest reference)")
    for k, tk in enumerate(case["onsets"]):
        inp = met["PaCO2"][k]
        for name in ("PaCO2", "CBF", "v_MCA,L", "CBV", "ICP"):
            c = met[name][k]
            a, o = c["rest"], c["own"]
            lag = a["t90"] - inp["rest"]["t90"] if name != "PaCO2" else np.nan
            lag_o = o["t90"] - inp["own"]["t90"] if name != "PaCO2" else np.nan
            if name in ("CBF", "v_MCA,L"):
                rise = f"{100 * a['rise'] / c['y_rest']:+.1f} %"
            elif name == "CBV":
                rise = f"{a['rise']:+.2f} mL"
            else:
                rise = f"{a['rise']:+.2f} mmHg"
            print(f"{k + 1} @{tk - clock_shift:<7.1f}  {name:<9}{_f(100 * c['residual'])}{_f(a['t50'])}{_f(a['t90'])}"
                  f"{_f(lag)}{_f(o['t50'], 8)}{_f(o['t90'], 8)}{_f(lag_o, 8)}{_f(c['t_peak'], 8)}{'*' if c['flag'] else ' '}"
                  f"{_f(100 * a['share60'])}{_f(100 * a['last30'], 8)}  {rise}")
    dv = met["CBV"][0]["rest"]["rise"]
    print(f"CBV rise in cycle 1: {dv:+.2f} mL = {100 * dv / m.CBV_total_n:+.1f} % of the PLACEHOLDER total CBV of "
          f"{m.CBV_total_n:g} mL (CBV_pct, spec 7.10)")


def print_etco2_sweep(title, cases, col):
    """E8 (info): CBF of cycle 1, residuals at onsets 2 and 3, x_CO2 residual, peak ratio; one row per case."""
    print(f"\n=== {title} ===")
    print("CBF, rest reference, times in s after onset 1. resid k = CBF 1 s before onset k minus rest, in % of the "
          "cycle-1 rise.\nx_CO2 res 2 = x_CO2 (MCA_L) 1 s before onset 2, in % of its value at the end of inhalation 1. "
          "peak 2/1 = peak CBF rise of\ncycle 2 / cycle 1.")
    print(f"{col:<12}{'t50':>7}{'t90':>7}{'at60%':>7}{'in t90':>8}{'lag':>7}{'resid 2%':>10}{'resid 3%':>10}"
          f"{'xCO2 res 2%':>13}{'peak 2/1':>10}")
    for key, case in cases.items():
        cb, inp, pr, t = case["met"]["CBF"], case["met"]["PaCO2"], case["proto"], case["r"]["t"]
        x = case["r"]["x_CO2"][:, 0]
        xres = np.interp(pr.onsets[1] - 1.0, t, x) / np.interp(pr.offsets[0], t, x)
        a = cb[0]["rest"]
        print(f"{key:<12}{_f(a['t50'])}{_f(a['t90'])}{_f(100 * a['share60'])}{_f(inp[0]['rest']['t90'], 8)}"
              f"{_f(a['t90'] - inp[0]['rest']['t90'])}{_f(100 * cb[1]['residual'], 10)}{_f(100 * cb[2]['residual'], 10)}"
              f"{_f(100 * xres, 13)}{_f(cb[1]['rest']['rise'] / a['rise'], 10, 4)}")


def _etco2_panels(case):
    sig, met = case["sig"], case["met"]

    def rel(k):
        return 100.0 * (sig[k] / met[k][0]["y_rest"] - 1.0)
    return (sig["PaCO2"], rel("CBF"), rel("v_MCA,L"), sig["CBV"] - met["CBV"][0]["y_rest"], sig["ICP"])


def _etco2_marks(case, shift=0.0):
    """t50, t90 of cycle 1 (rest reference) on the plotted clock, for the panels input, CBF, v_MCA, CBV."""
    tk = case["onsets"][0] - shift
    return {i: (tk + case["met"][k][0]["rest"]["t50"], tk + case["met"][k][0]["rest"]["t90"])
            for i, k in enumerate(ETCO2_RISING)}


def step_etco2():
    """python run.py new --etco2 (spec 7.17): placeholder shape, step comparison, sweeps; checks E0-E8."""
    proto = EtCO2Protocol()
    m = NewModel()
    print("=== End-tidal-shaped 5 % CO2 input (spec 7.17). Opt-in: not part of the default run, not part of --all ===")
    print(f"PLACEHOLDER input shape (from no source): first-order lag of the inhaled-gas schedule, tau_on "
          f"{proto.tau_on:g} s, tau_off {proto.tau_off:g} s,\nplateau {proto.P0 + proto.dP:g} mmHg (P0 {proto.P0:g} + "
          f"dP {proto.dP:g}; the student's assumption for 5 % CO2). End-tidal -> arterial gap {proto.gap:g} mmHg "
          f"(assumption).")
    print(f"Protocol: rest {proto.t_rest:g} s; {proto.n_cycles} x {proto.t_inhale:g} s of 5 % CO2 (MRI protocol); "
          f"{proto.t_air:g} s of air between them (the student's\nupper bound, exact value unknown); "
          f"{proto.t_after:g} s of air after the last. Onsets {', '.join(f'{t:g}' for t in proto.onsets)} s; end "
          f"{proto.t_end:g} s.\nComparison ('step'): the current switch-like input, 10-s linear ramps on the same schedule.")
    rep = Report()
    res = test_co2_etco2(rep, m, proto)
    n_fail = rep.show()
    print("\n" + BOLD_NOTE)
    print_etco2_table(f"Shaped input (PLACEHOLDER shape, tau {proto.tau_on:g} s), {proto.t_air:g} s of air", res["main"], m)
    print_etco2_table("Step input (current: 10-s linear ramps), same protocol", res["step"], m)
    taus = {"step 10 s": res["step"], **{f"tau {k:g} s": v for k, v in sorted(res["taus"].items())}}
    print_etco2_sweep("E8 (info): input shape, air 90 s (PLACEHOLDER shapes)", taus, "input")
    airs = {f"air {k:g} s": v for k, v in sorted(res["airs"].items())}
    print_etco2_sweep(f"E8 (info): air interval, tau {proto.tau_on:g} s (PLACEHOLDER shape)", airs, "interval")
    e = res["csv"]["E1_diff"]
    i = int(np.argmax(np.abs(e)))
    print(f"\nE1: largest relative CBF difference CSV round trip vs analytic input {e[i]:+.2e}, at t = "
          f"{res['main']['r']['t'][i]:.1f} s (the PCHIP interpolation of the 1-s samples next to a switch, spec 7.17)")

    path = os.path.join(out_dir("new"), "new_model_etco2.png")
    series = [(f"end-tidal-shaped (PLACEHOLDER, tau {proto.tau_on:g} s)", res["main"]["r"]["t"],
               _etco2_panels(res["main"]), dict(color="tab:blue", lw=1.3)),
              ("current input (10-s ramps)", res["step"]["r"]["t"], _etco2_panels(res["step"]),
               dict(color="tab:red", lw=1.0, ls="--"))]
    plot_etco2(series, list(zip(proto.onsets, proto.offsets)), _etco2_marks(res["main"]), path,
               f"New model: 5 % CO2, {proto.n_cycles} x {proto.t_inhale:g} s with {proto.t_air:g} s of air. "
               f"PLACEHOLDER input shape (first-order lag, tau {proto.tau_on:g} s, plateau {proto.P0 + proto.dP:g} mmHg)"
               f"\nvs the current 10-s ramps. Inhalations shaded; t50, t90 of cycle 1 of the shaped run marked. "
               f"Pa = 100 mmHg")
    print(f"Figure saved: {path}")

    # close-up of cycle 1: the onset shapes side by side (the runs above; nothing new is simulated)
    path = os.path.join(out_dir("new"), "new_model_etco2_cycle1.png")
    cases = [("current input: 10-s ramps", res["step"], dict(color="tab:red", lw=1.2, ls="--"))]
    cases += [(f"end-tidal, tau {k:g} s" + (" (default)" if k == proto.tau_on else ""), res["taus"][k], dict(color=c, lw=1.3))
              for k, c in zip(sorted(res["taus"]), ("tab:orange", "tab:blue", "tab:green"))]
    series = [(f"{label}: CBF t90 {case['met']['CBF'][0]['rest']['t90']:.0f} s", case["r"]["t"], _etco2_panels(case),
               style, {i: case["met"][k][0]["rest"]["t90"] for i, k in enumerate(ETCO2_RISING)})
              for label, case, style in cases]
    plot_etco2_onset(series, proto.onsets[0], proto.offsets[0], proto.onsets[1], path,
                     f"New model: cycle 1 of 5 % CO2, close-up. PLACEHOLDER end-tidal shapes (first-order lag, tau 5 / 20 / "
                     f"40 s, plateau {proto.P0 + proto.dP:g} mmHg)\nvs the current 10-s ramps. Dots: t90 of each curve; "
                     f"dotted line: 60 s after the onset; inhalation shaded. Pa = 100 mmHg")
    print(f"Figure saved: {path}")
    return n_fail


def run_etco2_measured(d, path, unit, cycles, length, barometric):
    """python run.py new --etco2-csv FILE ... (spec 7.17): the model driven by a measured end-tidal curve; E2, E5."""
    proto = replace(EtCO2Protocol(), t_inhale=length)
    print("=== Measured end-tidal CO2 curve (spec 7.17). Opt-in: not part of the default run, not part of --all ===")
    pb = (f" (P_B {barometric:g} mmHg; F/100 x (P_B - 47) is standard physiology, not from the project sources)"
          if unit == "percent" else "")
    print(f"{path}: {len(d.t)} samples, {d.t[0]:g} to {d.t[-1]:g} s; {d.dropped} non-finite or unreadable samples "
          f"dropped; unit {unit}{pb}")
    print(f"baseline (mean of the first 60 s of data) {d.baseline:.3f} mmHg -> model built with PaCO2_n = "
          f"{d.baseline + proto.gap:.3f} mmHg (end-tidal -> arterial gap {proto.gap:g} mmHg, assumption)")
    shift = ETCO2_PAD - d.t[0]                     # model time - CSV time
    print(f"cycles at {', '.join(f'{c:g}' for c in cycles)} s (CSV clock), length {length:g} s. Model time = CSV time "
          f"{'+' if shift >= 0 else '-'} {abs(shift):g} s ({ETCO2_PAD:g} s of rest holding the first value).")
    m = NewModel(PaCO2_n=d.baseline + proto.gap)
    f = d.function(pad=ETCO2_PAD, gap=proto.gap)
    rep = Report()
    case = etco2_case(rep, m, "measured", proto, f, onsets=list(d.to_model_time(cycles, pad=ETCO2_PAD)),
                      t_end=float(d.to_model_time(d.t[-1], pad=ETCO2_PAD)))
    n_fail = rep.show()
    print("\n" + BOLD_NOTE)
    print_etco2_table(f"Measured input {os.path.basename(path)} (onsets in the CSV clock)", case, m, clock_shift=shift)
    out = os.path.join(out_dir("new"), "new_model_etco2_csv.png")
    plot_etco2([(f"measured: {os.path.basename(path)}", case["r"]["t"] - shift, _etco2_panels(case),
                 dict(color="tab:blue", lw=1.3))], [(c, c + length) for c in cycles], _etco2_marks(case, shift), out,
               f"New model driven by the measured end-tidal curve {os.path.basename(path)} (unit {unit}).\n"
               f"Cycles shaded; t50, t90 of cycle 1 marked. Pa = 100 mmHg", xlabel="time [s, CSV clock]")
    print(f"Figure saved: {out}")
    return n_fail


def main_etco2(argv, parts=None):
    """--etco2 / --etco2-csv: stand-alone (spec 7.17). Combined with any other flag of `new` it stops with an error."""
    argv = list(argv)
    flags = [a for a in argv if a.startswith("--")]
    other = [a for a in flags if a not in ETCO2_FLAGS] + [f"--{q.replace('_', '-')}" for q in (parts or [])]
    if other:
        raise SystemExit(f"run.py new: --etco2 and --etco2-csv run on their own (spec 7.17); run the other parts "
                         f"separately. Also given: {' '.join(other)}")
    if ("--etco2" in argv) == ("--etco2-csv" in argv):
        raise SystemExit("run.py new: give either --etco2 (placeholder shape) or --etco2-csv FILE (measured curve)")
    if "--etco2" in argv:
        extra = [a for a in flags if a != "--etco2"]
        if extra:
            raise SystemExit(f"run.py new: {' '.join(extra)} only go with --etco2-csv")
        return step_etco2()
    if "--etco2-unit" not in argv or "--cycles" not in argv:
        raise SystemExit("run.py new: --etco2-csv needs --etco2-unit (mmHg, kPa or percent; never guessed) and "
                         "--cycles T1,T2,... (inhalation onsets in the CSV's own time [s])")
    path = argv[argv.index("--etco2-csv") + 1]
    unit = argv[argv.index("--etco2-unit") + 1]
    try:
        cycles = [float(c) for c in argv[argv.index("--cycles") + 1].split(",") if c.strip()]
        length = _argval(argv, "--cycle-length", 180.0)
        barometric = _argval(argv, "--barometric", 760.0)
        d = read_etco2_csv(path, unit, barometric)
        d.check_cycles(cycles, length)
    except (OSError, ValueError, IndexError) as e:
        raise SystemExit(f"run.py new --etco2-csv: {e}")
    return run_etco2_measured(d, path, unit, cycles, length, barometric)


if __name__ == "__main__":
    import sys
    raise SystemExit(1 if main(sys.argv[1:]) else 0)
