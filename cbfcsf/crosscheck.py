"""Cross-checks of the new model against the existing source codes (spec 7.8, 7.15): Ursino 2010, Vanek 2015.
The source codes are imported, not modified."""
import numpy as np
from scipy.integrate import solve_ivp

from cbfcsf.common.signals import ramp
from cbfcsf.models.new_model import Inputs


# ----------------------------------------------------------------------------------------------------------------------
# Fourth step (spec 7.8): cross-checks against the source models, ICA occlusion, parameter sensitivity
# ----------------------------------------------------------------------------------------------------------------------
def ursino_districts(u, y):
    """District flows and volumes of the existing Ursino code, recomputed from its state vector with the formulas of
    Ursino2010Model.derivatives() (that file is not modified). Order: ml, al, pl, mr, ar, pr."""
    Pic, Pv = y[0], y[1]
    Pd, x = np.asarray(y[5:11]), np.asarray(y[17:23]) - np.asarray(y[11:17])
    Cd0 = np.array([u.Cdn[d] for d in u.districts])
    kR = np.array([u.kR[d] for d in u.districts])
    sat = np.where(x < 0, u.sat1, u.sat2)
    uu = np.exp(x / (Cd0 * sat / 2.0))
    Cd = Cd0 * ((1.0 - sat) + (1.0 + sat) * uu) / (1.0 + uu)
    Vd = np.maximum(Cd * (Pd - Pic), 1e-4)
    Rd = kR * Cd0 ** 2 / Vd ** 2
    G = np.sum(2.0 / Rd) + 1.0 / u.Rpv + 1.0 / u.Rf
    Pc = (np.sum(2.0 * Pd / Rd) + Pic / u.Rf + Pv / u.Rpv) / G
    if Pc <= Pic:
        Pc = (np.sum(2.0 * Pd / Rd) + Pv / u.Rpv) / (G - 1.0 / u.Rf)
    return (Pd - Pc) / (Rd / 2.0), Vd


def test_cross_stroke(rep, r):
    """X1: left MCA occlusion in the existing Ursino code vs the new model (r = result of test_stroke)."""
    from cbfcsf.models.ursino_2010 import Ursino2010Model
    u = Ursino2010Model()
    s0 = solve_ivp(lambda t, y: u.derivatives(t, y), (0, 600), u.get_initial_state(), method="Radau")
    s1 = solve_ivp(lambda t, y: u.derivatives(t, y, mca_stenosis_l=1000.0), (0, 1800), s0.y[:, -1], method="Radau")
    q0, V0 = ursino_districts(u, s0.y[:, -1])
    q1, V1 = ursino_districts(u, s1.y[:, -1])
    urs = dict(res=q1[0] / q0[0], dV=V1[0] / V0[0] - 1, dCBF=q1.sum() / q0.sum() - 1,
               dVa=V1.sum() - V0.sum(), icp=(s0.y[0, -1], s1.y[0, -1]))
    i0, i1, _ = r["marks"]
    new = dict(res=r["q_k"][i1, 0] / r["q_k"][i0, 0], dV=r["Va"][i1, 0] / r["Va"][i0, 0] - 1,
               dCBF=r["CBF"][i1] / r["CBF"][i0] - 1, dVa=r["Va_total"][i1] - r["Va_total"][i0])
    rep.check("X1a residual MCA_L flow, new / Ursino code", new["res"] / urs["res"], 0.5, 2.0, "ratio")
    rep.check("X1b MCA_L volume change, new - Ursino", new["dV"] - urs["dV"], -0.15, 0.15, "rel")
    rep.check("X1c total CBF change, new - Ursino", new["dCBF"] - urs["dCBF"], -0.10, 0.10, "rel")
    return urs, new


def vanek_lit(rate, t_on=300.0, dur=600.0, Pb=10.0):
    """Lumbar infusion in the existing Vanek code with constant boundary pressures (that file is not modified).
    rise = ICP rise at the end of the infusion, rise5 = after 300 s of infusion (needs dur >= 300)."""
    from cbfcsf.models.vanek_2015 import Vanek2015Model
    v = Vanek2015Model()
    const = {k: (lambda t, c=c: c) for k, c in dict(Pa=80.0, Pb=Pb, PC=2.0, PT=2.0, PL=2.0).items()}
    P0 = np.array([11.0, 10.0, 10.0, 10.0, 10.0])
    inf = lambda t: np.array([0.0, 0.0, 0.0, 0.0, rate if t >= t_on else 0.0])
    a = solve_ivp(lambda t, P: v.derivatives(t, P, const, infusion_rates=inf(t)), (0.0, t_on + dur), P0, method="Radau",
                  t_eval=[t_on - 1.0, t_on + 300.0, t_on + dur], max_step=5.0, rtol=1e-7, atol=1e-9)
    b = solve_ivp(lambda t, P: v.derivatives(t, P, const, infusion_rates=np.array([0, 0, 0, 0, rate])), (0.0, 8000.0), P0,
                  method="Radau", t_eval=[8000.0], max_step=50.0, rtol=1e-7, atol=1e-9)
    return dict(rest=a.y[1, 0], rise5=a.y[1, 1] - a.y[1, 0], rise=a.y[1, 2] - a.y[1, 0], plateau=b.y[1, -1])


def test_cross_lit(rep, r):
    """X2: lumbar infusion in the existing Vanek code vs the new model (r = result of test_lit)."""
    van = vanek_lit(1.5 / 60.0)
    i0, i1, _ = r["marks"]
    rise_new = r["ICP"][i1] - r["ICP"][i0]
    rep.check("X2a LIT plateau, Vanek code - new", van["plateau"] - r["plateau_sim"], 0.43, 0.63, "mmHg")
    rep.check("X2b ICP rise after 10 min, new / Vanek code", rise_new / van["rise"], 0.5, 1.5, "ratio")
    van["rise_new"] = rise_new
    return van


def test_cross_more(rep, m, r_co2, r_lit, r_ica, van):
    """X3, X4, X2c, X5 (spec 7.15): further comparisons with the source codes, added in the review of 16 Nov.
    r_co2 = 5 % CO2 run, r_lit = LIT run, r_ica = ICA run, van = result of test_cross_lit. X5 is informational:
    it records a known disagreement (the lumped circle has no basilar inlet, spec 3.2.2)."""
    from cbfcsf.models.ursino_2010 import Ursino2010Model
    u = Ursino2010Model()
    end = lambda y0, T, **kw: solve_ivp(lambda t, y: u.derivatives(t, y, **kw), (0.0, T), y0, method="Radau",
                                        rtol=1e-7, atol=1e-9).y[:, -1]
    y_rest = end(u.get_initial_state(), 600.0)
    q0, V0 = ursino_districts(u, y_rest)

    def urs(T, **kw):
        """Ursino code after T s with the given inputs: district flows, total CBF change, arterial volume change."""
        q, V = ursino_districts(u, end(y_rest, T, **kw))
        return q, q.sum() / q0.sum() - 1, V.sum() - V0.sum()

    out = {}
    # X3: graded CO2 range (T5g compares at 60 mmHg, where the response is saturated)
    _, dq_u, dV_u = urs(600.0, PaCO2=47.5)
    i0, i1, _ = r_co2["marks"]
    dq_n, dV_n = r_co2["CBF"][i1] / r_co2["CBF"][i0] - 1, r_co2["Va_total"][i1] - r_co2["Va_total"][i0]
    rep.check("X3a 5 % CO2: total CBF rise, new - Ursino", dq_n - dq_u, -0.03, 0.03, "rel")
    rep.check("X3b 5 % CO2: arterial volume change, new / Ursino", dV_n / dV_u, 0.85, 1.15, "ratio")
    out["co2"] = (dq_n, dq_u, dV_n, dV_u)

    # X4: autoregulation, Pa 100 -> 80 mmHg held 600 s
    _, dq_u, dV_u = urs(600.0, Pa=80.0)
    r = m.simulate(Inputs(Pa=lambda t: ramp(t, [(0, 100), (1, 80)])), 600.0, t_eval=[0.0, 600.0], max_step=2.0)
    dq_n, dV_n = r["CBF"][1] / r["CBF"][0] - 1, r["Va_total"][1] - r["Va_total"][0]
    rep.check("X4a Pa 100 -> 80: total CBF change, new - Ursino", dq_n - dq_u, -0.03, 0.03, "rel")
    rep.check("X4b Pa 100 -> 80: arterial volume change, new / Ursino", dV_n / dV_u, 0.85, 1.15, "ratio")
    out["pa"] = (dq_n, dq_u, dV_n, dV_u)

    # X2c: LIT half-way up the curve (X2b compares at 10 min, where both models are close to the same plateau)
    j0 = r_lit["marks"][0]
    j5 = np.searchsorted(r_lit["t"], r_lit["span"][0] + 300.0)
    rise5 = r_lit["ICP"][j5] - r_lit["ICP"][j0]
    van12 = vanek_lit(1.5 / 60.0, Pb=12.0)
    rep.check("X2c ICP rise after 5 min, new / Vanek code", rise5 / van["rise5"], 0.75, 1.25, "ratio")
    rep.info("X2c same, Vanek code with Pb = 12", rise5 / van12["rise5"], 0.75, 1.25, "ratio")
    out["lit"] = (rise5, van["rise5"], van12["rise5"])

    # X5 (info): left ICA occlusion. Ursino's order of districts: ml, al, pl, mr, ar, pr; his "rest_L" = al + pl
    q1, dq_u, dV_u = urs(1800.0, ica_stenosis_l=1000.0)
    k0, k1, _ = r_ica["marks"]
    dq_n, dV_n = r_ica["CBF"][k1] / r_ica["CBF"][k0] - 1, r_ica["Va_total"][k1] - r_ica["Va_total"][k0]
    rest_u = (q1[1] + q1[2]) / (q0[1] + q0[2]) - 1
    rest_n = r_ica["q_k"][k1, 1] / r_ica["q_k"][k0, 1] - 1
    rep.info("X5a ICA occlusion: total CBF change, new - Ursino", dq_n - dq_u, -0.10, 0.10, "rel")
    rep.info("X5b ICA occlusion: rest_L flow change, new - Ursino", rest_n - rest_u, -0.10, 0.10, "rel")
    out["ica"] = dict(dq=(dq_n, dq_u), dV=(dV_n, dV_u), rest=(rest_n, rest_u),
                      mca=(r_ica["q_k"][k1, 0] / r_ica["q_k"][k0, 0] - 1, q1[0] / q0[0] - 1),
                      urs_districts=q1[:3] / q0[:3] - 1)
    return out
