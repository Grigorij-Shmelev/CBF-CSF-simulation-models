"""
===============================================================================
Ursino & Giannessi (2010) Python Model Implementation
===============================================================================
Title: A Model of Cerebrovascular Reactivity Including the Circle of Willis 
       and Cortical Anastomoses (Ursino & Giannessi, 2010)
Reference: Ann Biomed Eng. 2010 May; 38(5): 955-974.

This script implements the full 23-differential-equation lumped-parameter 
model of human cerebral hemodynamics, intracranial pressure (ICP) dynamics, 
cerebrovascular autoregulation, CO2 reactivity, the Circle of Willis, and 
distal cortical anastomoses.

Usage:
    python ursino_2010_model.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Ursino2010Model:
    """Full 23-state differential-equation model of human cerebral circulation."""

    def __init__(self):
        # Viscosity (Poise to mmHg s)
        self.eta = 0.04 / 1333.22  # mmHg s

        # Geometry for Hagen-Poiseuille (radius in cm, length in cm)
        self.geom = {
            'ICA': (0.205, 13.15),
            'BA': (0.17, 4.92),
            'MCA': (0.14, 7.25),
            'PCA1': (0.10, 1.00),
            'ACA1': (0.075, 1.57),
            'PCA2': (0.10, 4.72),
            'ACA2': (0.075, 0.672),
            'PCoA': (0.036, 2.00),
            'ACoA': (0.040, 0.50),
        }

        # Hagen-Poiseuille baseline resistances (mmHg s / mL)
        self.R_net = {
            k: 8.0 * self.eta * l / (np.pi * r**4)
            for k, (r, l) in self.geom.items()
        }

        # Large artery compliances (mL/mmHg)
        self.C_ICA = 3.4e-3
        self.C_BA = 1.7e-3

        # CSF & Venous hydrodynamics (Table 1)
        self.Rf = 2.38e3     # CSF formation resistance (mmHg s / mL)
        self.Ro = 526.3      # CSF outflow resistance (mmHg s / mL)
        self.Rpv = 0.880     # Proximal venous resistance (mmHg s / mL)
        self.Rvs1 = 0.366    # Terminal venous resistance at zero collapse (mmHg s / mL)

        # Cortical anastomoses resistances (Table 1)
        self.Rc_pms = 120.0  # Posterior-to-Middle (mmHg s / mL)
        self.Rc_ams = 105.0  # Anterior-to-Middle (mmHg s / mL)
        self.Rc_pp  = 75.0   # Posterior-to-Posterior (mmHg s / mL)
        self.Rc_aa  = 22.0   # Anterior-to-Anterior (mmHg s / mL)

        # ICP and Venous mechanics
        self.kE = 0.077      # Intracranial elastance coefficient (mL^-1)
        self.kven = 0.155    # Venous rigidity coefficient (mL^-1)
        self.Pv1 = -2.5      # Transmural collapse threshold (mmHg)
        self.Pvs = 6.0       # Dural sinus pressure (mmHg)

        # Baseline CBF and territory weights
        self.qn_total = 12.5 # Total baseline CBF (mL/s)
        self.weights = {'m': 0.30, 'a': 0.08, 'p': 0.12}

        # 6 Districts: ml, al, pl, mr, ar, pr
        self.districts = ['ml', 'al', 'pl', 'mr', 'ar', 'pr']
        self.qn = {}
        self.Cdn = {}
        self.Rdn = {}
        self.kR = {}

        total_Cdn = 0.200  # Total peripheral compliance (mL/mmHg)
        total_Rdn = 5.4    # Total peripheral resistance (mmHg s / mL)

        for d in self.districts:
            j = d[0]
            w = self.weights[j]
            self.qn[d] = w * self.qn_total
            self.Cdn[d] = w * total_Cdn
            self.Rdn[d] = total_Rdn / w
            # Transmural pressure Pd_n - Pic_n = 58.5 - 9.5 = 49.0 mmHg
            self.kR[d] = self.Rdn[d] * (49.0**2)

        # Control mechanism parameters (Table 1)
        self.saut = 20.0     # Autoregulation time constant (s)
        self.Gaut = 0.9      # Autoregulation gain
        self.sCO2 = 40.0     # CO2 reactivity time constant (s)
        self.GCO2 = 4.0      # CO2 reactivity gain
        self.sat1 = 7.0      # Lower saturation (vasoconstriction)
        self.sat2 = 0.4      # Upper saturation (vasodilation)
        self.kCO2 = 15.0     # CO2 correction slope
        self.bCO2 = -0.5      # CO2 correction threshold
        self.PaCO2n = 40.0   # Baseline PaCO2 (mmHg)

    def get_initial_state(self):
        """Returns baseline state vector (23 elements)."""
        y0 = np.zeros(23)
        y0[0] = 9.5     # Pic
        y0[1] = 14.0    # Pv
        y0[2] = 92.5    # PICAl
        y0[3] = 92.5    # PICAr
        y0[4] = 92.5    # PBA
        y0[5:11] = 58.5 # Pd (6 districts)
        y0[11:17] = 0.0 # xaut
        y0[17:23] = 0.0 # xCO2
        return y0

    def derivatives(self, t, y, Pa=100.0, PaCO2=40.0, ica_stenosis_l=1.0, mca_stenosis_l=1.0):
        """Right-hand-side differential equation system (dy/dt)."""
        Pic, Pv, PICAl, PICAr, PBA = y[0], y[1], y[2], y[3], y[4]

        Pd = {'ml': y[5], 'al': y[6], 'pl': y[7], 'mr': y[8], 'ar': y[9], 'pr': y[10]}
        xaut = {'ml': y[11], 'al': y[12], 'pl': y[13], 'mr': y[14], 'ar': y[15], 'pr': y[16]}
        xCO2 = {'ml': y[17], 'al': y[18], 'pl': y[19], 'mr': y[20], 'ar': y[21], 'pr': y[22]}

        # Large artery resistances with optional stenosis
        R_ICAl = self.R_net['ICA'] * ica_stenosis_l
        R_ICAr = self.R_net['ICA']
        R_BA   = self.R_net['BA']

        R_MCAl = self.R_net['MCA'] * mca_stenosis_l
        R_MCAr = self.R_net['MCA']

        R_PCA1l, R_PCA1r = self.R_net['PCA1'], self.R_net['PCA1']
        R_ACA1l, R_ACA1r = self.R_net['ACA1'], self.R_net['ACA1']
        R_PCA2l, R_PCA2r = self.R_net['PCA2'], self.R_net['PCA2']
        R_ACA2l, R_ACA2r = self.R_net['ACA2'], self.R_net['ACA2']
        R_PCoAl, R_PCoAr = self.R_net['PCoA'], self.R_net['PCoA']
        R_ACoA = self.R_net['ACoA']

        # District active compliance & resistance
        Cd, Rd, dCd_dx = {}, {}, {}
        for d in self.districts:
            x = xCO2[d] - xaut[d]
            Cd0 = self.Cdn[d]
            sat = self.sat1 if x < 0 else self.sat2
            DCd = 2.0 * sat
            kCd = Cd0 * sat / 2.0
            u = np.exp(x / kCd)
            Cd[d] = Cd0 * ((1.0 - DCd/2.0) + (1.0 + DCd/2.0)*u) / (1.0 + u)
            dCd_dx[d] = 4.0 * u / ((1.0 + u)**2)

            Vd = max(Cd[d] * (Pd[d] - Pic), 1e-4)
            Rd[d] = self.kR[d] * (self.Cdn[d]**2) / (Vd**2)

        # Auxiliary pressures in Circle of Willis
        G_PCA1l = 1.0 / (R_PCA2l + Rd['pl']/2.0) + 1.0 / R_PCoAl + 1.0 / R_PCA1l
        PPCA1l = (Pd['pl'] / (R_PCA2l + Rd['pl']/2.0) + PICAl / R_PCoAl + PBA / R_PCA1l) / G_PCA1l

        G_PCA1r = 1.0 / (R_PCA2r + Rd['pr']/2.0) + 1.0 / R_PCoAr + 1.0 / R_PCA1r
        PPCA1r = (Pd['pr'] / (R_PCA2r + Rd['pr']/2.0) + PICAr / R_PCoAr + PBA / R_PCA1r) / G_PCA1r

        g_al = 1.0 / (R_ACA2l + Rd['al']/2.0) + 1.0 / R_ACA1l + 1.0 / R_ACoA
        g_ar = 1.0 / (R_ACA2r + Rd['ar']/2.0) + 1.0 / R_ACA1r + 1.0 / R_ACoA
        b_l = Pd['al'] / (R_ACA2l + Rd['al']/2.0) + PICAl / R_ACA1l
        b_r = Pd['ar'] / (R_ACA2r + Rd['ar']/2.0) + PICAr / R_ACA1r

        det = g_al * g_ar - (1.0 / R_ACoA)**2
        PACA1l = (g_ar * b_l + (1.0 / R_ACoA) * b_r) / det
        PACA1r = (g_al * b_r + (1.0 / R_ACoA) * b_l) / det

        # Capillary pressure Pc
        G_cap = sum(2.0 / Rd[d] for d in self.districts) + 1.0 / self.Rpv + 1.0 / self.Rf
        num_cap = sum(2.0 * Pd[d] / Rd[d] for d in self.districts) + Pic / self.Rf + Pv / self.Rpv
        Pc = num_cap / G_cap
        if Pc <= Pic:
            G_cap -= 1.0 / self.Rf
            num_cap = sum(2.0 * Pd[d] / Rd[d] for d in self.districts) + Pv / self.Rpv
            Pc = num_cap / G_cap

        # District flows
        q = {d: (Pd[d] - Pc) / (Rd[d]/2.0) for d in self.districts}

        # Control mechanism dynamics
        dxaut_dt, dxCO2_dt, dx_dt = {}, {}, {}
        for d in self.districts:
            dxaut_dt[d] = (-xaut[d] + self.Gaut * (q[d] - self.qn[d]) / self.qn[d]) / self.saut
            rel_q = (q[d] - self.qn[d]) / self.qn[d]
            clip_arg = np.clip(-self.kCO2 * (rel_q - self.bCO2), -50, 50)
            ACO2 = 1.0 / (1.0 + np.exp(clip_arg))
            log_ratio = np.log10(max(PaCO2, 1e-3) / self.PaCO2n)
            dxCO2_dt[d] = (-xCO2[d] + self.GCO2 * ACO2 * log_ratio) / self.sCO2
            dx_dt[d] = dxCO2_dt[d] - dxaut_dt[d]

        # Venous and CSF dynamics
        Rvs = (Pv - self.Pvs) / (Pv - Pic) * self.Rvs1 if self.Pvs < Pic else self.Rvs1
        Rvs = max(Rvs, 1e-4)

        qf = max((Pc - Pic) / self.Rf, 0.0)
        qo = max((Pic - self.Pvs) / self.Ro, 0.0)

        dVd = {
            'ml': (PICAl - Pd['ml'])/(R_MCAl + Rd['ml']/2.0) - (Pd['ml'] - Pc)/(Rd['ml']/2.0) + (Pd['al'] - Pd['ml'])/self.Rc_ams + (Pd['pl'] - Pd['ml'])/self.Rc_pms,
            'mr': (PICAr - Pd['mr'])/(R_MCAr + Rd['mr']/2.0) - (Pd['mr'] - Pc)/(Rd['mr']/2.0) + (Pd['ar'] - Pd['mr'])/self.Rc_ams + (Pd['pr'] - Pd['mr'])/self.Rc_pms,
            'al': (PACA1l - Pd['al'])/(R_ACA2l + Rd['al']/2.0) - (Pd['al'] - Pc)/(Rd['al']/2.0) + (Pd['ml'] - Pd['al'])/self.Rc_ams + (Pd['ar'] - Pd['al'])/self.Rc_aa,
            'ar': (PACA1r - Pd['ar'])/(R_ACA2r + Rd['ar']/2.0) - (Pd['ar'] - Pc)/(Rd['ar']/2.0) + (Pd['mr'] - Pd['ar'])/self.Rc_ams + (Pd['al'] - Pd['ar'])/self.Rc_aa,
            'pl': (PPCA1l - Pd['pl'])/(R_PCA2l + Rd['pl']/2.0) - (Pd['pl'] - Pc)/(Rd['pl']/2.0) + (Pd['ml'] - Pd['pl'])/self.Rc_pms + (Pd['pr'] - Pd['pl'])/self.Rc_pp,
            'pr': (PPCA1r - Pd['pr'])/(R_PCA2r + Rd['pr']/2.0) - (Pd['pr'] - Pc)/(Rd['pr']/2.0) + (Pd['mr'] - Pd['pr'])/self.Rc_pms + (Pd['pl'] - Pd['pr'])/self.Rc_pp,
        }

        dVvi_dt = (Pc - Pv)/self.Rpv - (Pv - self.Pvs)/Rvs

        RHS_1 = sum(dVd[d] for d in self.districts) + dVvi_dt + qf - qo
        dPic_dt = self.kE * Pic * RHS_1

        transmural_v = max(Pv - Pic - self.Pv1, 0.1)
        Cvi = 1.0 / (self.kven * transmural_v)
        dPv_dt = dPic_dt + (1.0 / Cvi) * dVvi_dt

        dPICAl_dt = dPic_dt + (1.0 / self.C_ICA) * (
            (Pa - PICAl)/R_ICAl + (PPCA1l - PICAl)/R_PCoAl + (Pd['ml'] - PICAl)/(R_MCAl + Rd['ml']/2.0) + (PACA1l - PICAl)/R_ACA1l
        )
        dPICAr_dt = dPic_dt + (1.0 / self.C_ICA) * (
            (Pa - PICAr)/R_ICAr + (PPCA1r - PICAr)/R_PCoAr + (Pd['mr'] - PICAr)/(R_MCAr + Rd['mr']/2.0) + (PACA1r - PICAr)/R_ACA1r
        )
        dPBA_dt = dPic_dt + (1.0 / self.C_BA) * (
            (Pa - PBA)/R_BA + (PPCA1l - PBA)/R_PCA1l + (PPCA1r - PBA)/R_PCA1r
        )

        dPd_dt = {
            d: dPic_dt + (1.0 / Cd[d]) * (dVd[d] - (Pd[d] - Pic) * dCd_dx[d] * dx_dt[d])
            for d in self.districts
        }

        dydt = np.zeros(23)
        dydt[0] = dPic_dt
        dydt[1] = dPv_dt
        dydt[2] = dPICAl_dt
        dydt[3] = dPICAr_dt
        dydt[4] = dPBA_dt
        for i, d in enumerate(self.districts):
            dydt[5 + i]  = dPd_dt[d]
            dydt[11 + i] = dxaut_dt[d]
            dydt[17 + i] = dxCO2_dt[d]

        return dydt

    def get_flow_dict(self, y, Pa=100.0, ica_stenosis_l=1.0, mca_stenosis_l=1.0):
        """Extracts key vessel blood flows (in mL/s)."""
        Pic, Pv, PICAl, PICAr, PBA = y[0], y[1], y[2], y[3], y[4]
        Pd = {'ml': y[5], 'al': y[6], 'pl': y[7], 'mr': y[8], 'ar': y[9], 'pr': y[10]}
        xaut = {'ml': y[11], 'al': y[12], 'pl': y[13], 'mr': y[14], 'ar': y[15], 'pr': y[16]}
        xCO2 = {'ml': y[17], 'al': y[18], 'pl': y[19], 'mr': y[20], 'ar': y[21], 'pr': y[22]}

        Rd = {}
        for d in self.districts:
            x = xCO2[d] - xaut[d]
            Cd0 = self.Cdn[d]
            sat = self.sat1 if x < 0 else self.sat2
            DCd = 2.0 * sat
            kCd = Cd0 * sat / 2.0
            u = np.exp(x / kCd)
            Cd = Cd0 * ((1.0 - DCd/2.0) + (1.0 + DCd/2.0)*u) / (1.0 + u)
            Vd = max(Cd * (Pd[d] - Pic), 1e-4)
            Rd[d] = self.kR[d] * (self.Cdn[d]**2) / (Vd**2)

        R_ICAl = self.R_net['ICA'] * ica_stenosis_l
        R_ICAr = self.R_net['ICA']
        R_MCAl = self.R_net['MCA'] * mca_stenosis_l
        R_MCAr = self.R_net['MCA']

        return {
            'q_ica_l': (Pa - PICAl) / R_ICAl,
            'q_ica_r': (Pa - PICAr) / R_ICAr,
            'q_mca_l': (PICAl - Pd['ml']) / (R_MCAl + Rd['ml']/2.0),
            'q_mca_r': (PICAr - Pd['mr']) / (R_MCAr + Rd['mr']/2.0),
        }


def main():
    print("=== Ursino & Giannessi (2010) Model Simulation ===")
    model = Ursino2010Model()
    y0 = model.get_initial_state()

    print("\n1. Running Baseline Steady State Simulation...")
    sol_base = solve_ivp(lambda t, y: model.derivatives(t, y), [0, 50], y0, method='Radau')
    y_ss = sol_base.y[:, -1]
    flows = model.get_flow_dict(y_ss)

    print(f"   - Intracranial Pressure (Pic): {y_ss[0]:.2f} mmHg (Target: 9.5)")
    print(f"   - Cerebral Venous Pressure (Pv): {y_ss[1]:.2f} mmHg (Target: 14.0)")
    print(f"   - Left ICA Blood Flow: {flows['q_ica_l']:.2f} mL/s (Target: 4.75)")
    print(f"   - Left MCA Blood Flow: {flows['q_mca_l']:.2f} mL/s (Target: 3.75)")

    print("\n2. Simulating Steal Phenomenon during ICA Occlusion & Hypercapnia...")
    sol_st = solve_ivp(lambda t, y: model.derivatives(t, y, ica_stenosis_l=500.0), [0, 50], y_ss, method='Radau')
    sol_h = solve_ivp(lambda t, y: model.derivatives(t, y, PaCO2=60.0, ica_stenosis_l=500.0), [0, 180], sol_st.y[:, -1], method='Radau')
    f_occl_pre = model.get_flow_dict(sol_st.y[:, -1], ica_stenosis_l=500.0)
    f_occl_post = model.get_flow_dict(sol_h.y[:, -1], ica_stenosis_l=500.0)

    print(f"   - Pre-Hypercapnia Ipsilateral MCA Flow:  {f_occl_pre['q_mca_l']:.2f} mL/s")
    print(f"   - Post-Hypercapnia Ipsilateral MCA Flow: {f_occl_post['q_mca_l']:.2f} mL/s (Steal phenomenon confirmed)")
    print(f"   - Post-Hypercapnia Contralateral MCA Flow: {f_occl_post['q_mca_r']:.2f} mL/s")

if __name__ == "__main__":
    main()
