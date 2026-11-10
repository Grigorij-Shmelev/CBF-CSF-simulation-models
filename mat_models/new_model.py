"""
new_model.py -- first prototype of the new 0D model of cerebral blood volume and CSF dynamics.

Specification: docs/specification.md (equations: section 3.5, parameters: section 4, tests: section 7.1).
Sources of the physics: Ursino & Giannessi 2010 (blood walls, autoregulation, CO2),
Vanek 2015 / Stepanik (CSF chain, spinal walls). No existing model file is modified.

Structure
    4 arterial compartments: mL, rL, mR, rR  (m = MCA territory, r = rest of brain; L/R = side)
    1 shared venous compartment
    5 CSF compartments: 1 ventricles, 2 cranial SAS, 3 cervical, 4 thoracic, 5 lumbar
    brain volume constant, skull rigid

States (18)
    y[0:4]   V_a   arterial stressed volumes [mL]      (mL, rL, mR, rR)
    y[4]     V_v   venous stressed volume [mL]
    y[5:7]   V_1, V_2  cranial CSF volumes [mL]
    y[7:10]  P_3, P_4, P_5  spinal CSF pressures [mmHg]
    y[10:14] x_aut  autoregulation state
    y[14:18] x_CO2  CO2 reactivity state

Algebraic unknowns, solved at every call (spec 3.5.6)
    z = [P_1, P_2, P_c, P_N,L, P_N,R]

Units: mmHg, mL, s.

Usage (from mat_models/):
    python new_model.py                 # tests of spec sections 7.1 and 7.3, figures to docs/figures/  (about 70 s)
    python new_model.py --calibrate     # + comparison with Vanek's raw compliances (K1), data-file check (K2) (3 min)
    python new_model.py --sensitivity   # + phi_v sensitivity V1 of spec 7.5                             (about 7 min)
    python new_model.py --all
"""

import os
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.special import expit

NAMES = ("mL", "rL", "mR", "rR")
SIDE = np.array([0, 0, 1, 1])        # node index of each arterial compartment (0 = left, 1 = right)
IS_MCA = np.array([True, False, True, False])


# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------
class Inputs:
    """Time-dependent inputs. Every entry is a function of time t [s]."""

    def __init__(self, Pa=100.0, PaCO2=40.0, Pvs=6.0, PC=2.0, PT=2.0, PL=2.0, infusion=None, occlusion=None):
        self.Pa = self._fn(Pa)            # arterial input pressure (AIF) [mmHg]
        self.PaCO2 = self._fn(PaCO2)      # arterial CO2 pressure [mmHg]
        self.Pvs = self._fn(Pvs)          # dural sinus pressure [mmHg]
        self.PC = self._fn(PC)            # cervical epidural pressure [mmHg]
        self.PT = self._fn(PT)            # thoracic epidural pressure [mmHg]
        self.PL = self._fn(PL)            # lumbar epidural pressure [mmHg]
        # infusion(t) -> 5 rates [mL/s] into CSF compartments 1..5
        self.infusion = infusion if infusion is not None else (lambda t: np.zeros(5))
        # occlusion(t) -> multipliers of (R_MCA,L, R_MCA,R, R_ICA,L, R_ICA,R); 1 = open
        self.occlusion = occlusion if occlusion is not None else (lambda t: np.ones(4))

    @staticmethod
    def _fn(v):
        return v if callable(v) else (lambda t, _v=float(v): _v)

    @staticmethod
    def ddt(f, t, h=1e-4):
        return (f(t + h) - f(t - h)) / (2.0 * h)


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
class NewModel:
    """New 0D model, spec section 3.5. Parameters: spec section 4 (source given next to each value)."""

    def __init__(self, phi_a=1.0, phi_v=1.0, R_com=15.0, c_sp_scale=None, k_E=0.077):
        # ---- Known: Ursino & Giannessi 2010, Table 1 -------------------------------------------------------------
        self.tau_aut, self.G_aut = 20.0, 0.9
        self.tau_CO2, self.G_CO2 = 40.0, 4.0
        self.sat1, self.sat2 = 7.0, 0.4
        self.k_CO2, self.b_CO2, self.PaCO2_n = 15.0, -0.5, 40.0
        self.R_pv, self.R_vs1 = 0.880, 0.366
        self.k_ven, self.P_v1 = 0.155, -2.5
        self.Pvs_n = 6.0
        self.q_n_total = 12.5
        self.C_n_total = 0.200
        self.w = np.array([0.30, 0.20, 0.30, 0.20])          # territory weights (MCA; ACA2 + PCA2)
        eta = 0.04 / 1333.22                                    # blood viscosity, Poise -> mmHg s
        hp = lambda r, l: 8.0 * eta * l / (np.pi * r ** 4)      # Hagen-Poiseuille
        self.R_ICA = hp(0.205, 13.15)                           # 0.569
        self.R_MCA = hp(0.14, 7.25)                             # 1.44
        self.Pc_target = 25.0                                   # basal capillary pressure (Ursino p. 962), check only
        self.R_c = 1.0 / (1.0 / 105.0 + 1.0 / 120.0)            # MCA <-> rest anastomosis: R_c,am || R_c,pm = 56

        # ---- Known: Vanek 2015 Tab. 6.2.1 / Stepanik MATLAB CSF_defSettings.m ------------------------------------
        self.R12, self.R23, self.R34, self.R45 = 160.0, 0.30, 0.40, 0.30
        self.C3C0, self.C4T0, self.C5L0 = 1.125, 1.500, 2.250
        self.a_sp, self.r_sp = 0.6, 0.6                         # spinal wall exponents (MATLAB only, spec 4.4)
        self.S1 = 550.0 / 86400.0                               # CSF production [mL/s]
        self.alpha = 0.11 / 60.0                                # resorption conductance [mL/(s mmHg)]
        self.V10, self.V20 = 30.0, 90.0                         # initial cranial CSF volumes [mL]
        self.P_epi_n = 2.0                                      # mean epidural pressure [mmHg]

        # ---- Unknown, start values (spec 4.4) --------------------------------------------------------------------
        self.phi_a, self.phi_v = phi_a, phi_v                   # share of a blood wall that faces the SAS. Fixed
        #                                                         at 1 (decision 10 Nov, spec 7.6); kept as a test knob
        self.R_com = R_com                                      # left-right communicating resistance (ACoA: 14.9)
        self.V_vn = 4.5                                         # venous stressed volume at rest [mL] (Ursino: 4-5)
        self.C_min_frac = 0.05                                  # bound C_k >= C_min_frac * C_n,k (default P11)
        self.Pa_n = 100.0

        self._calibrate()

        # Spinal compliance scale (spec 7.5, 7.6). Default: scale Vanek's three values by one factor so that the
        # storage at rest equals Ursino's 1/(k_E * ICP) (Table 1, k_E = 0.077 /mL). Vanek's own model had extra
        # storage in its venous walls (C1b, C2b), which do not exist here. c_sp_scale=1 gives Vanek's raw values.
        if c_sp_scale is None:
            c_sp_scale = 1.0 / (k_E * self.P2_n) / self.storage_at_rest()
        self.c_sp_scale = c_sp_scale
        self.C3C0, self.C4T0, self.C5L0 = (c_sp_scale * c for c in (self.C3C0, self.C4T0, self.C5L0))

    # ------------------------------------------------------------------------------------------------------------------
    def _calibrate(self):
        """Resting state (spec 4.2): every compartment carries its basal flow, no anastomotic flow, CSF in balance."""
        self.q_n = self.w * self.q_n_total
        self.C_n = self.w * self.C_n_total
        q_side = self.q_n[0] + self.q_n[1]

        self.P2_n = self.Pvs_n + self.S1 / self.alpha                    # production = resorption
        self.P1_n = self.P2_n + self.R12 * self.S1                       # F12 = S1
        self.Pout_a_n = self.phi_a * self.P2_n + (1 - self.phi_a) * self.P1_n
        self.Pout_v_n = self.phi_v * self.P2_n + (1 - self.phi_v) * self.P1_n
        self.Pv_n = max(self.Pout_v_n, self.Pvs_n) + self.q_n_total * self.R_vs1
        self.Pc_n = self.Pv_n + self.q_n_total * self.R_pv
        self.PN_n = self.Pa_n - self.R_ICA * q_side

        R_n_m = (self.PN_n - self.Pc_n) / self.q_n[0] - self.R_MCA       # MCA branch: P_N -> P_c at q_n,m
        self.Pk_n = self.Pc_n + self.q_n[0] * R_n_m / 2.0                # arterial compartment pressure
        R_n_r = 2.0 * (self.Pk_n - self.Pc_n) / self.q_n[1]              # P_r = P_m at rest
        self.R_rest = (self.PN_n - self.Pk_n) / self.q_n[1] - R_n_r / 2.0
        self.R_n = np.array([R_n_m, R_n_r, R_n_m, R_n_r])

        self.V_n = self.C_n * (self.Pk_n - self.Pout_a_n)                # arterial stressed volumes at rest
        self.u_n = self.Pv_n - self.Pout_v_n - self.P_v1                 # venous law reference

    def storage_at_rest(self):
        """Total spinal wall compliance at rest [mL/mmHg]: the only volume storage outside the skull."""
        u = self.P2_n - self.P_epi_n
        return (self.C3C0 + self.C4T0 + self.C5L0) * np.exp(-self.r_sp * abs(u) ** self.a_sp)

    @classmethod
    def vanek_values(cls, **kw):
        """Variant with Vanek's raw spinal compliances (no scaling)."""
        return cls(c_sp_scale=1.0, **kw)

    def initial_state(self):
        y0 = np.zeros(18)
        y0[0:4] = self.V_n
        y0[4] = self.V_vn
        y0[5], y0[6] = self.V10, self.V20
        y0[7:10] = self.P2_n          # no CSF flow along the spine at rest
        return y0

    # ------------------------------------------------------------------------------------------------------------------
    def _wall(self, y):
        """State-dependent wall quantities: C_k, transmural pressures, resistances (spec 3.5.3)."""
        Va = np.maximum(y[0:4], 1e-6)
        x = y[14:18] - y[10:14]                                          # x_CO2 - x_aut
        sat = np.where(x < 0, self.sat1, self.sat2)
        kC = self.C_n * sat / 2.0
        C = self.C_n * np.maximum((1.0 - sat) + 2.0 * sat * expit(x / kC), self.C_min_frac)
        e = Va / C                                                       # P_k - P_out,a
        R = self.R_n * (self.V_n / Va) ** 2
        wv = self.P_v1 + self.u_n * np.exp(self.k_ven * (y[4] - self.V_vn))   # P_v - P_out,v
        return C, e, R, wv

    def _flows(self, z, y, e, R, wv, inp_t, collapse, resorb):
        """All flows for given algebraic unknowns z. Affine in z for fixed switches."""
        P1, P2, Pc, PNL, PNR = z
        Pa, Pvs, I, occ = inp_t["Pa"], inp_t["Pvs"], inp_t["I"], inp_t["occ"]
        Pout_a = self.phi_a * P2 + (1 - self.phi_a) * P1
        Pout_v = self.phi_v * P2 + (1 - self.phi_v) * P1
        Pk = Pout_a + e
        Pv = Pout_v + wv
        PN = np.array([PNL, PNL, PNR, PNR])
        R_feed = np.array([self.R_MCA * occ[0], self.R_rest, self.R_MCA * occ[1], self.R_rest])
        q_in = (PN - Pk) / (R_feed + R / 2.0)
        q_k = (Pk - Pc) / (R / 2.0)
        q_cL = (e[1] - e[0]) / self.R_c                                  # rest -> MCA, left
        q_cR = (e[3] - e[2]) / self.R_c
        q_c = np.array([q_cL, -q_cL, q_cR, -q_cR])
        q_pv = (Pc - Pv) / self.R_pv
        q_vs = (Pv - (Pout_v if collapse else Pvs)) / self.R_vs1          # Starling resistor (Ursino Eq. 10)
        F12 = (P1 - P2) / self.R12
        F23 = (P2 - y[7]) / self.R23
        F_out = self.alpha * (P2 - Pvs) if resorb else 0.0
        q_ICA = np.array([(Pa - PNL) / (self.R_ICA * occ[2]), (Pa - PNR) / (self.R_ICA * occ[3])])

        dVa = q_in - q_k + q_c
        dVv = q_pv - q_vs
        dV1 = -F12 + self.S1 + I[0]
        dV2 = F12 - F23 - F_out + I[1]
        res = np.array([
            q_ICA[0] + (PNR - PNL) / self.R_com - q_in[0] - q_in[1],     # node L
            q_ICA[1] + (PNL - PNR) / self.R_com - q_in[2] - q_in[3],     # node R
            q_k.sum() - q_pv,                                            # capillary node
            dV1 + (1 - self.phi_a) * dVa.sum() + (1 - self.phi_v) * dVv,  # rigid skull, ventricular walls
            dV2 + self.phi_a * dVa.sum() + self.phi_v * dVv,             # rigid skull, SAS walls
        ])
        out = dict(P1=P1, P2=P2, Pc=Pc, PNL=PNL, PNR=PNR, Pk=Pk, Pv=Pv, Pout_v=Pout_v, q_in=q_in, q_k=q_k, q_c=q_c,
                   q_pv=q_pv, q_vs=q_vs, q_ICA=q_ICA, F12=F12, F23=F23, F_out=F_out,
                   dVa=dVa, dVv=dVv, dV1=dV1, dV2=dV2)
        return res, out

    def solve_algebraic(self, t, y, inp):
        """Solve the 5 algebraic equations (spec 3.5.6). Linear for fixed switches; iterate on the two switches."""
        C, e, R, wv = self._wall(y)
        inp_t = dict(Pa=inp.Pa(t), Pvs=inp.Pvs(t), I=np.asarray(inp.infusion(t), float),
                     occ=np.asarray(inp.occlusion(t), float))
        collapse, resorb = True, True
        zero = np.zeros(5)
        for _ in range(4):
            r0, _o = self._flows(zero, y, e, R, wv, inp_t, collapse, resorb)
            J = np.empty((5, 5))
            for j in range(5):
                ej = zero.copy(); ej[j] = 1.0
                J[:, j] = self._flows(ej, y, e, R, wv, inp_t, collapse, resorb)[0] - r0
            z = np.linalg.solve(J, -r0)
            res, out = self._flows(z, y, e, R, wv, inp_t, collapse, resorb)
            c_new = out["Pout_v"] >= inp_t["Pvs"]
            r_new = out["P2"] > inp_t["Pvs"]
            if c_new == collapse and r_new == resorb:
                break
            collapse, resorb = c_new, r_new
        out["C"] = C
        out["R"] = R
        out["res"] = float(np.max(np.abs(res)))
        return out

    # ------------------------------------------------------------------------------------------------------------------
    def rhs(self, t, y, inp):
        o = self.solve_algebraic(t, y, inp)
        I = np.asarray(inp.infusion(t), float)
        dy = np.zeros(18)
        dy[0:4] = o["dVa"]
        dy[4] = o["dVv"]
        dy[5], dy[6] = o["dV1"], o["dV2"]

        # spinal CSF (Vanek Eq. 5.2.7), walls C = C0 exp(-r |dP|^a)
        P3, P4, P5 = y[7], y[8], y[9]
        F34 = (P3 - P4) / self.R34
        F45 = (P4 - P5) / self.R45
        for i, (P, C0, f, net) in enumerate((
                (P3, self.C3C0, inp.PC, o["F23"] - F34 + I[2]),
                (P4, self.C4T0, inp.PT, F34 - F45 + I[3]),
                (P5, self.C5L0, inp.PL, F45 + I[4]))):
            C = C0 * np.exp(-self.r_sp * abs(P - f(t)) ** self.a_sp)
            dy[7 + i] = Inputs.ddt(f, t) + net / C

        # control (Ursino Eq. 21-23)
        rel = (o["q_k"] - self.q_n) / self.q_n
        dy[10:14] = (-y[10:14] + self.G_aut * rel) / self.tau_aut
        A = expit(self.k_CO2 * (rel - self.b_CO2))
        dy[14:18] = (-y[14:18] + self.G_CO2 * A * np.log10(max(inp.PaCO2(t), 1e-3) / self.PaCO2_n)) / self.tau_CO2
        return dy

    def simulate(self, inp, t_end, y0=None, t_eval=None, max_step=np.inf, rtol=1e-7, atol=1e-9):
        y0 = self.initial_state() if y0 is None else y0
        sol = solve_ivp(lambda t, y: self.rhs(t, y, inp), (0.0, t_end), y0, method="Radau",
                        t_eval=t_eval, max_step=max_step, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return self.observe(sol.t, sol.y, inp)

    def observe(self, t, Y, inp):
        """Rebuild pressures, flows and volumes on the solution grid (spec 3.5.7)."""
        keys = ("P1", "P2", "Pc", "Pv", "q_vs", "F_out", "F23", "res")
        r = {k: np.empty(len(t)) for k in keys}
        r.update(Pk=np.empty((len(t), 4)), q_k=np.empty((len(t), 4)), C=np.empty((len(t), 4)),
                 q_ICA=np.empty((len(t), 2)), q_c=np.empty((len(t), 4)))
        for i, ti in enumerate(t):
            o = self.solve_algebraic(ti, Y[:, i], inp)
            for k in keys:
                r[k][i] = o[k]
            for k in ("Pk", "q_k", "C", "q_ICA", "q_c"):
                r[k][i] = o[k]
        r.update(t=t, y=Y, Va=Y[0:4].T, Vv=Y[4], V1=Y[5], V2=Y[6], P3=Y[7], P4=Y[8], P5=Y[9],
                 x_aut=Y[10:14].T, x_CO2=Y[14:18].T)
        r["ICP"] = r["P2"]
        r["CBF"] = r["q_k"].sum(axis=1)
        r["Va_total"] = r["Va"].sum(axis=1)
        r["Vblood"] = r["Va_total"] + r["Vv"]                # stressed blood volume (CBV minus unstressed part)
        r["Vcranial"] = r["Vblood"] + r["V1"] + r["V2"]      # must stay constant (rigid skull)
        # Spinal CSF volume: dV = C(u) du with u = P_i - P_epidural, so the volume is a function of u alone.
        # Integrate the wall law exactly (Gauss-Legendre) instead of integrating flows on the output grid,
        # which loses volume at abrupt events.
        xi, wi = np.polynomial.legendre.leggauss(32)
        r["dVspinal"] = np.zeros(len(t))
        for P, C0, f in ((r["P3"], self.C3C0, inp.PC), (r["P4"], self.C4T0, inp.PT), (r["P5"], self.C5L0, inp.PL)):
            u = P - np.array([f(ti) for ti in t])
            half = (u - u[0]) / 2.0
            nodes = u[0] + half[:, None] * (xi[None, :] + 1.0)
            r["dVspinal"] += half * (C0 * np.exp(-self.r_sp * np.abs(nodes) ** self.a_sp) @ wi)
        I = np.array([np.asarray(inp.infusion(ti), float) for ti in t])
        r["I_total"] = I.sum(axis=1)
        r["dVcsf"] = (r["V1"] + r["V2"]) - (r["V1"][0] + r["V2"][0]) + r["dVspinal"]
        r["csf_net_in"] = cumulative_trapezoid(self.S1 + r["I_total"] - r["F_out"], t, initial=0.0)
        return r


# ----------------------------------------------------------------------------------------------------------------------
# Tests of spec section 7.1
# ----------------------------------------------------------------------------------------------------------------------
def ramp(t, pts):
    """Piecewise-linear signal through pts = [(t0, v0), (t1, v1), ...]."""
    tt, vv = zip(*pts)
    return float(np.interp(t, tt, vv))


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, value, lo, hi, unit=""):
        ok = bool(lo <= value <= hi)
        self.rows.append((name, value, lo, hi, unit, "PASS" if ok else "FAIL"))
        return ok

    def info(self, name, value, lo, hi, unit=""):
        inside = lo <= value <= hi
        self.rows.append((name, value, lo, hi, unit, "info (inside)" if inside else "info (OUTSIDE)"))

    def show(self):
        print(f"\n{'test':<46}{'value':>12}  {'bound':<24}{'':<8}result")
        for name, v, lo, hi, unit, res in self.rows:
            print(f"{name:<46}{v:>12.5g}  [{lo:.4g}, {hi:.4g}] {unit:<10}{res}")
        n_fail = sum(1 for r in self.rows if r[5] == "FAIL")
        print(f"\n{len(self.rows)} checks, {n_fail} failed")
        return n_fail


def common_checks(rep, tag, r, symmetric=True):
    """T2 rigid skull, T3 algebraic residual, T7 CSF bookkeeping (= L5, applied to every run), T4 symmetry."""
    rep.check(f"{tag} T2 rigid skull: max |dV_cranial|", np.max(np.abs(r["Vcranial"] - r["Vcranial"][0])), 0, 1e-3, "mL")
    rep.check(f"{tag} T3 algebraic residual", np.max(r["res"]), 0, 1e-8, "mL/s")
    rep.check(f"{tag} T7 CSF bookkeeping error", np.max(np.abs(r["dVcsf"] - r["csf_net_in"])), 0, 0.05, "mL")
    if symmetric:
        asym = np.max(np.abs(r["Va"][:, 0:2] - r["Va"][:, 2:4]) / r["Va"][:, 0:2])
        rep.check(f"{tag} T4 left-right asymmetry", asym, 0, 1e-6, "rel")


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
    pco2 = lambda t: ramp(t, [(0, 40), (t_on, 40), (t_on + 10, 60), (t_off, 60), (t_off + 10, 40), (t_end, 40)])
    r = m.simulate(Inputs(PaCO2=pco2), t_end, t_eval=np.linspace(0, t_end, 3601), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = np.searchsorted(t, t_on - 1), np.searchsorted(t, t_off - 1), len(t) - 1

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
    from ursino_2010_model import Ursino2010Model
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

    inf = lambda t: np.array([0, 0, 0, 0, ramp(t, [(0, 0), (t_on, 0), (t_on + 5, rate), (t_off, rate),
                                                    (t_off + 5, 0), (t_end, 0)])])
    r = m.simulate(Inputs(infusion=inf), t_end, t_eval=np.linspace(0, t_end, 3121), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = np.searchsorted(t, t_on - 1), np.searchsorted(t, t_off - 1), len(t) - 1
    rep.check("L2 ICP at end of infusion", r["ICP"][i1], r["ICP"][i0] + 5.0, 23.2, "mmHg")
    rep.check("L3 max |CBF/CBF_rest - 1|", np.max(np.abs(r["CBF"] / r["CBF"][i0] - 1)), 0, 0.10, "rel")
    rep.check("L4 |stressed CBV change| at end of infusion", abs(r["Vblood"][i1] - r["Vblood"][i0]), 0, 3.0, "mL")
    rep.check("L6 recovery: ICP", r["ICP"][i2] - r["ICP"][i0], -0.5, 0.5, "mmHg")
    rep.check("L6 recovery: CBV", r["Vblood"][i2] / r["Vblood"][i0] - 1, -0.02, 0.02, "rel")
    common_checks(rep, "LIT", r)
    r["marks"], r["span"], r["plateau"] = (i0, i1, i2), (t_on, t_off), plateau
    return r


def test_stroke(rep, m):
    """Occlusion of the left MCA: R_MCA,L x 1000 for 30 min, then reopened."""
    t_on, t_off, t_end = 120.0, 1920.0, 3120.0
    occ = lambda t: np.array([ramp(t, [(0, 1), (t_on, 1), (t_on + 5, 1000), (t_off, 1000), (t_off + 5, 1), (t_end, 1)]),
                              1.0, 1.0, 1.0])
    r = m.simulate(Inputs(occlusion=occ), t_end, t_eval=np.linspace(0, t_end, 3121), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = np.searchsorted(t, t_on - 1), np.searchsorted(t, t_off - 1), len(t) - 1
    q0, q1 = r["q_k"][i0], r["q_k"][i1]
    rep.check("S1 residual flow MCA_L", q1[0] / q0[0], 0.02, 0.60, "rel")
    rep.check("S2 stressed volume MCA_L", r["Va"][i1, 0] / r["Va"][i0, 0] - 1, -0.70, -0.05, "rel")
    rep.check("S3 right-side flows: max |change|", np.max(np.abs(q1[2:] / q0[2:] - 1)), 0, 0.05, "rel")
    rep.check("S3 rest_L flow: |change|", abs(q1[1] / q0[1] - 1), 0, 0.10, "rel")
    rep.check("S4 total CBF", r["CBF"][i1] / r["CBF"][i0] - 1, -0.30, -0.10, "rel")
    rep.check("S5 ICP at end of occlusion", r["ICP"][i1] - r["ICP"][i0], -0.5, 0.5, "mmHg")
    back = max(np.max(np.abs(r["q_k"][i2] / q0 - 1)), np.max(np.abs(r["Va"][i2] / r["Va"][i0] - 1)),
               abs(r["Vv"][i2] / r["Vv"][i0] - 1))
    rep.check("S6 after reopening: max |change|", back, 0, 0.02, "rel")
    common_checks(rep, "stroke", r, symmetric=False)
    r["marks"], r["span"] = (i0, i1, i2), (t_on, t_off)
    return r


def test_co2_5pct(rep, m):
    """5 % CO2 inhalation, taken as PaCO2 = 47.5 mmHg (student's figure, not from the project sources)."""
    t_on, t_off, t_end = 120.0, 720.0, 1920.0
    pco2 = lambda t: ramp(t, [(0, 40), (t_on, 40), (t_on + 10, 47.5), (t_off, 47.5), (t_off + 10, 40), (t_end, 40)])
    r = m.simulate(Inputs(PaCO2=pco2), t_end, t_eval=np.linspace(0, t_end, 1921), max_step=5.0)
    t = r["t"]
    i0, i1, i2 = np.searchsorted(t, t_on - 1), np.searchsorted(t, t_off - 1), len(t) - 1
    rep.check("C1 CBF rise, 47.5 mmHg", r["CBF"][i1] / r["CBF"][i0] - 1, 0.10, 0.55, "rel")
    last = r["CBF"][(t >= t_off - 100) & (t <= t_off - 1)]
    rep.check("C2 chatter: CBF spread over last 100 s", np.ptp(last) / np.mean(last), 0, 0.01, "rel")
    rep.check("C3 recovery: CBF", r["CBF"][i2] / r["CBF"][i0] - 1, -0.02, 0.02, "rel")
    rep.check("C3 recovery: CBV", r["Vblood"][i2] / r["Vblood"][i0] - 1, -0.02, 0.02, "rel")
    common_checks(rep, "CO2 5%", r)
    r["marks"], r["span"] = (i0, i1, i2), (t_on, t_off)
    return r


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


def plot_run(r, m, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = r["t"] / 60.0
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    ax[0, 0].plot(t, r["CBF"], "k", label="total")
    for j, n in enumerate(NAMES):
        ax[0, 0].plot(t, r["q_k"][:, j], label=n, ls="-" if j < 2 else "--")
    ax[0, 0].set_title("Flow [mL/s]"); ax[0, 0].legend(fontsize=8, ncol=2)
    for j, n in enumerate(NAMES):
        ax[0, 1].plot(t, r["Va"][:, j], label=n, ls="-" if j < 2 else "--")
    ax[0, 1].plot(t, r["Vv"], color="tab:purple", label="venous")
    ax[0, 1].set_title("Stressed blood volume per compartment [mL]"); ax[0, 1].legend(fontsize=8, ncol=2)
    ax[0, 2].plot(t, r["ICP"], label="ICP = P2"); ax[0, 2].plot(t, r["P5"], "--", label="P5 lumbar")
    ax[0, 2].axhline(m.Pvs_n, color="gray", lw=0.8, ls=":"); ax[0, 2].set_title("CSF pressure [mmHg]")
    ax[0, 2].legend(fontsize=8)
    ax[1, 0].plot(t, r["Vblood"] - r["Vblood"][0], label="blood (stressed CBV)")
    ax[1, 0].plot(t, r["V1"] + r["V2"] - (r["V1"][0] + r["V2"][0]), label="cranial CSF")
    ax[1, 0].plot(t, r["dVspinal"], label="spinal CSF")
    ax[1, 0].set_title("Volume change [mL]"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, r["x_CO2"][:, 0] - r["x_aut"][:, 0], label="mL"); ax[1, 1].plot(t, r["x_CO2"][:, 1] - r["x_aut"][:, 1], label="rL")
    ax[1, 1].plot(t, r["x_CO2"][:, 2] - r["x_aut"][:, 2], "--", label="mR")
    ax[1, 1].set_title("Control x = x_CO2 - x_aut (> 0: dilation)"); ax[1, 1].legend(fontsize=8)
    ax[1, 2].plot(t, r["Pk"][:, 0], label="P arterial mL"); ax[1, 2].plot(t, r["Pk"][:, 1], label="P arterial rL")
    ax[1, 2].plot(t, r["Pc"], label="P capillary"); ax[1, 2].plot(t, r["Pv"], label="P venous")
    ax[1, 2].set_title("Blood pressures [mmHg]"); ax[1, 2].legend(fontsize=8)
    for a in ax.ravel():
        a.axvspan(r["span"][0] / 60.0, r["span"][1] / 60.0, color="tab:orange", alpha=0.10); a.grid(alpha=0.3)
    for a in ax[1]:
        a.set_xlabel("time [min]")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


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


def main(argv=()):
    do_cal = "--calibrate" in argv or "--all" in argv
    do_sens = "--sensitivity" in argv or "--all" in argv
    m = NewModel()
    print("=== New model prototype: resting state (spec 4.2) ===")
    print(f"ICP (P2) {m.P2_n:.3f}  P1 {m.P1_n:.3f}  Pv {m.Pv_n:.3f}  Pc {m.Pc_n:.3f}  P_N {m.PN_n:.2f}  P_k {m.Pk_n:.2f} mmHg")
    print(f"R_n (MCA, rest) {m.R_n[0]:.2f}, {m.R_n[1]:.2f}   R_rest {m.R_rest:.3f}   R_c {m.R_c:.1f}   R_com {m.R_com:.1f}")
    print(f"V_n arterial {m.V_n.round(3)} (total {m.V_n.sum():.2f} mL)   V_vn {m.V_vn} mL")

    rep = Report()
    test_rest(rep, m)
    runs = {
        "CO2 challenge 40 -> 60 mmHg": (test_co2(rep, m), "new_model_co2.png"),
        "5 % CO2 (PaCO2 47.5 mmHg)": (test_co2_5pct(rep, m), "new_model_co2_5pct.png"),
        "Lumbar infusion test, 1.5 mL/min for 10 min": (test_lit(rep, m), "new_model_lit.png"),
        "Stroke: left MCA occluded 30 min, then reopened": (test_stroke(rep, m), "new_model_stroke.png"),
    }
    test_pulsatile(rep, m)
    n_fail = rep.show()

    fig_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures"))
    os.makedirs(fig_dir, exist_ok=True)
    for name, (r, fname) in runs.items():
        summary(name, r, m)
        plot_run(r, m, os.path.join(fig_dir, fname), f"New model. {name} (shaded). Pa = 100 mmHg")
    co2 = runs["CO2 challenge 40 -> 60 mmHg"][0]
    base_runs = {k: v[0] for k, v in runs.items()}

    print(f"\nMCA flow rise at 60 mmHg: new model {100 * co2['rise_new']:+.1f} %, existing Ursino code {100 * co2['rise_ursino']:+.1f} %")
    print(f"Figures saved in {fig_dir}")
    if do_cal:
        n_fail += step_calibration(m, base_runs, fig_dir)
    if do_sens:
        n_fail += step_sensitivity(base_runs)
    return n_fail


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
        "CO2 challenge 40 -> 60 mmHg": test_co2(rep_v, mv),
        "5 % CO2 (PaCO2 47.5 mmHg)": test_co2_5pct(rep_v, mv),
        "Lumbar infusion test, 1.5 mL/min for 10 min": test_lit(rep_v, mv),
        "Stroke: left MCA occluded 30 min, then reopened": test_stroke(rep_v, mv),
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
    rec_path = os.path.normpath(os.path.join(fig_dir, "..", "..", "zbysek_matlab_code", "data", "inputs_OK.mat"))
    if os.path.exists(rec_path):
        vc, vb = run_recording(mc, rec_path), run_recording(m, rec_path)
        print(f"\n=== K2: Stepanik's data file, one beat repeated ({rec_path}) ===")
        print(f"cardiac period {vc['period']:.3f} s, mean P_A {vc['Pa_mean']:.1f} mmHg, arterial pulse {vc['Pa_pp']:.1f} mmHg p-p")
        print(f"ICP pulse p-p: data file {vc['icp_rec_pp']:.2f} mmHg | default model {vc['icp_mod_pp']:.2f} | Vanek values {vb['icp_mod_pp']:.2f}")
        print(f"ICP mean:      data file {vc['icp_rec_mean']:.2f} mmHg | default model {vc['icp_mod_mean']:.2f} | Vanek values {vb['icp_mod_mean']:.2f}")
        print(f"stressed CBV pulse p-p: default {vc['cbv_pp']:.2f} mL | Vanek values {vb['cbv_pp']:.2f} mL")
        rep2.check("K2 ICP pulse, default model / data file", vc["ratio"], 0.5, 2.0, "ratio")
        rep2.info("K2 ICP pulse, Vanek values / data file", vb["ratio"], 0.5, 2.0, "ratio")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        w = (vc["tr"] >= 20) & (vc["tr"] <= 30)
        ax.plot(vc["tr"][w], vc["icp_rec"][w] - vc["icp_rec_mean"], "k", label="Stepanik data file (one beat repeated)")
        for v, lab in ((vc, "model, default (storage calibrated)"), (vb, "model, Vanek values")):
            w2 = (v["t"] >= 20) & (v["t"] <= 30)
            ax.plot(v["t"][w2], v["icp_mod"][w2] - v["icp_mod_mean"], label=lab)
        ax.set_xlabel("time [s]"); ax.set_ylabel("ICP - mean [mmHg]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("ICP pulse: model driven by the arterial pressure of the data file vs its ICP")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "new_model_recording.png"), dpi=140); plt.close(fig)
    else:
        print(f"\nK2 skipped: recording not found at {rec_path}")

    return n_fail + rep2.show()


def step_sensitivity(base_runs):
    """Third step (spec 7.5): V1, phi_v = 1 vs 0.5."""
    rep2 = Report()
    base = {"5% CO2": base_runs["5 % CO2 (PaCO2 47.5 mmHg)"], "LIT": base_runs["Lumbar infusion test, 1.5 mL/min for 10 min"],
            "stroke": base_runs["Stroke: left MCA occluded 30 min, then reopened"]}
    rows, res = sensitivity_phi_v(rep2, base)
    print("\n=== V1: phi_v sensitivity ===")
    print(f"{'scenario':<10}{'dCBV phi_v=1':>14}{'phi_v=0.5':>11}{'ICP max 1':>11}{'0.5':>8}{'ICP min 1':>11}{'0.5':>8}")
    for name, d in rows:
        print(f"{name:<10}{d[0][0]:>14.3f}{d[1][0]:>11.3f}{d[0][1]:>11.2f}{d[1][1]:>8.2f}{d[0][2]:>11.2f}{d[1][2]:>8.2f}")
    print(f"pre-set scenario bounds failed with phi_v = 0.5: {res[0.5]['fails']}")
    return rep2.show()


if __name__ == "__main__":
    import sys
    raise SystemExit(1 if main(sys.argv[1:]) else 0)
