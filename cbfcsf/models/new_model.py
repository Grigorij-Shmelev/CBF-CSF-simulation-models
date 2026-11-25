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
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.special import expit

from cbfcsf.common.physics import hagen_poiseuille, wall_compliance


NAMES = ("mL", "rL", "mR", "rR")
SIDE = np.array([0, 0, 1, 1])        # node index of each arterial compartment (0 = left, 1 = right)
IS_MCA = np.array([True, False, True, False])


# ----------------------------------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------------------------------
class Inputs:
    """Time-dependent inputs. Every entry is a function of time t [s].
    Note: the model's resting state is computed for its own parameter Pvs_n (6 mmHg). A model built with another
    Pvs_n must be run with Inputs(Pvs=m.Pvs_n), otherwise it does not start in equilibrium (lit_curve() does this)."""

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

    def __init__(self, phi_a=1.0, phi_v=1.0, R_com=15.0, c_sp_scale=None, k_E=0.077, **overrides):
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
        self.R_ICA = hagen_poiseuille(0.205, 13.15, eta)        # 0.569
        self.R_MCA = hagen_poiseuille(0.14, 7.25, eta)          # 1.44
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
        self.k_E = k_E                                          # Ursino's elastance coefficient, storage target only

        # Switchable options (spec 7.10). Defaults reproduce the model without them exactly.
        self.kC_scale = 1.0        # width of the C(x) sigmoid relative to Ursino's. 1 = Ursino
        # Outputs (spec 7.10)
        self.CBV_total_n = 56.0    # total CBV at rest [mL]. PLACEHOLDER: 4 mL/100 g x 1400 g, not from the project sources
        self.r_MCA_n, self.k_MCA = 0.14, 12.0   # MCA radius at rest [cm] and rigidity (Ursino Table 1, Eq. 20)

        # Parameter overrides for sensitivity studies (spec 7.8), applied before the resting state is computed
        for name, value in overrides.items():
            if not hasattr(self, name):
                raise AttributeError(f"unknown parameter {name}")
            setattr(self, name, value)

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
        return wall_compliance(self.C3C0 + self.C4T0 + self.C5L0, self.r_sp, self.a_sp, u)

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
        kC = self.C_n * sat / 2.0 * self.kC_scale
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
            C = wall_compliance(C0, self.r_sp, self.a_sp, P - f(t))
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
                 q_ICA=np.empty((len(t), 2)), q_c=np.empty((len(t), 4)), q_in=np.empty((len(t), 4)),
                 PN=np.empty((len(t), 2)))
        for i, ti in enumerate(t):
            o = self.solve_algebraic(ti, Y[:, i], inp)
            for k in keys:
                r[k][i] = o[k]
            for k in ("Pk", "q_k", "C", "q_ICA", "q_c", "q_in"):
                r[k][i] = o[k]
            r["PN"][i] = (o["PNL"], o["PNR"])
        r.update(t=t, y=Y, Va=Y[0:4].T, Vv=Y[4], V1=Y[5], V2=Y[6], P3=Y[7], P4=Y[8], P5=Y[9],
                 x_aut=Y[10:14].T, x_CO2=Y[14:18].T)
        r["ICP"] = r["P2"]
        r["CBF"] = r["q_k"].sum(axis=1)
        r["Va_total"] = r["Va"].sum(axis=1)
        r["Vblood"] = r["Va_total"] + r["Vv"]                # stressed blood volume (CBV minus unstressed part)
        r["Vcranial"] = r["Vblood"] + r["V1"] + r["V2"]      # must stay constant (rigid skull)
        # Outputs of spec 7.10. CBV in percent of the (placeholder) total CBV at rest.
        r["Vblood0"] = self.V_n.sum() + self.V_vn
        r["CBV_pct"] = 100.0 * (r["Vblood"] - r["Vblood0"]) / self.CBV_total_n
        # MCA velocity, Ursino Eq. 19-20 with the transmural pressure (P_N - ICP) of the MCA inlet
        x_p = np.clip((r["PN"] - r["P2"][:, None]) / (self.PN_n - self.P2_n), 1e-6, None)
        r["r_MCA"] = self.r_MCA_n * (np.log(x_p) / self.k_MCA + 1.0)
        r["v_MCA"] = r["q_in"][:, [0, 2]] / (np.pi * r["r_MCA"] ** 2)      # [cm/s], columns: left, right
        # Spinal CSF volume: dV = C(u) du with u = P_i - P_epidural, so the volume is a function of u alone.
        # Integrate the wall law exactly (Gauss-Legendre) instead of integrating flows on the output grid,
        # which loses volume at abrupt events.
        xi, wi = np.polynomial.legendre.leggauss(32)
        r["dVspinal"] = np.zeros(len(t))
        for P, C0, f in ((r["P3"], self.C3C0, inp.PC), (r["P4"], self.C4T0, inp.PT), (r["P5"], self.C5L0, inp.PL)):
            u = P - np.array([f(ti) for ti in t])
            half = (u - u[0]) / 2.0
            nodes = u[0] + half[:, None] * (xi[None, :] + 1.0)
            r["dVspinal"] += half * (wall_compliance(C0, self.r_sp, self.a_sp, nodes) @ wi)
        I = np.array([np.asarray(inp.infusion(ti), float) for ti in t])
        r["I_total"] = I.sum(axis=1)
        r["dVcsf"] = (r["V1"] + r["V2"]) - (r["V1"][0] + r["V2"][0]) + r["dVspinal"]
        r["csf_net_in"] = cumulative_trapezoid(self.S1 + r["I_total"] - r["F_out"], t, initial=0.0)
        return r
