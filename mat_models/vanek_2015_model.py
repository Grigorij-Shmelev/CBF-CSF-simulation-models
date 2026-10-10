"""
===============================================================================
Petr Vaněk (2015) 3rd Generation Craniospinal CSF Model Implementation
===============================================================================
Title: Modelování změn intrakraniálního tlaku během fyziologických a 
       patologických dějů (Disertační práce, Univerzita Karlova v Praze)
Author: Petr Vaněk (2015), FTVS UK, Prague.

This script implements Petr Vaněk's 3rd generation 5-compartment model 
of craniospinal cerebrospinal fluid (CSF) dynamics, which extends earlier 
models (Štěpáník, Otáhal) by incorporating an explicit intracranial venous 
sub-compartment (b - mozkové žíly) alongside arterial pulsations (a), 
non-linear exponential compliances, pressure-dependent variable resorption,
and multi-compartmental spinal pathways (Cervical, Thoracic, Lumbar).

Usage:
    python vanek_2015_model.py
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Vanek2015Model:
    """3rd Generation 5-compartment craniospinal CSF model (Vaněk 2015)."""

    def __init__(self):
        # Inter-compartmental hydraulic resistances [mmHg * s / mL]
        self.R12 = 160.0   # Ventricles -> Intracranial SAS
        self.R23 = 0.30    # Intracranial SAS -> Cervical SAS
        self.R34 = 0.40    # Cervical SAS -> Thoracic SAS
        self.R45 = 0.30    # Thoracic SAS -> Lumbar SAS

        self.Z12 = 1.0 / self.R12
        self.Z23 = 1.0 / self.R23
        self.Z34 = 1.0 / self.R34
        self.Z45 = 1.0 / self.R45

        # Maximum compliances C0 [mL / mmHg] at zero pressure differential
        # Arterial boundary (a)
        self.C1a0 = 1.5e-5  # Arteries - Ventricles
        self.C2a0 = 2.0e-2  # Arteries - Intracranial SAS
        
        # Venous sub-compartment (b) - Added in 3rd Generation (Vaněk 2015)
        self.C1b0 = 6.0e-1  # Veins - Ventricles
        self.C2b0 = 6.0e-1  # Veins - Intracranial SAS

        # Spinal epidural boundaries (C, T, L)
        self.C3C0 = 1.125   # Cervical
        self.C4T0 = 1.500   # Thoracic
        self.C5L0 = 2.250   # Lumbar

        # Non-linear compliance parameters: curvature exponent (a) and slope (r)
        self.a1a, self.r1a = 0.3, 1.0
        self.a2a, self.r2a = 0.3, 1.0
        self.a1b, self.r1b = 0.6, 0.6
        self.a2b, self.r2b = 0.6, 0.6
        self.a3C, self.r3C = 0.6, 0.6
        self.a4T, self.r4T = 0.6, 0.6
        self.a5L, self.r5L = 0.6, 0.6

        # Baseline resting volumes [mL]
        self.V10 = 30.0   # Ventricular system
        self.V20 = 90.0   # Intracranial SAS
        self.V30 = 10.0   # Cervical SAS
        self.V40 = 10.0   # Thoracic SAS
        self.V50 = 10.0   # Lumbar SAS

        # CSF Formation & Resorption
        self.S1 = 550.0 / 86400.0  # CSF production in ventricles (~0.006366 mL/s)
        self.peq = 10.0            # Equilibrium ICP [mmHg]
        self.alpha = 0.11 / 60.0   # Resorption outflow conductance [mL / (mmHg * s)]
        self.Pmin = (self.alpha * self.peq - self.S1) / self.alpha  # Resorption threshold (~2.0 mmHg)

    def resorption(self, p2):
        """Pressure-dependent linear resorption of CSF into dural veins."""
        if p2 <= self.Pmin:
            return 0.0
        return self.alpha * (p2 - self.Pmin)

    def compute_compliances(self, P1, P2, P3, P4, P5, Pa, Pb, PC, PT, PL):
        """Calculates non-linear exponential compliances for all interfaces."""
        C1a = self.C1a0 * np.exp(-self.r1a * (abs(P1 - Pa)**self.a1a))
        C2a = self.C2a0 * np.exp(-self.r2a * (abs(P2 - Pa)**self.a2a))
        
        C1b = self.C1b0 * np.exp(-self.r1b * (abs(P1 - Pb)**self.a1b))
        C2b = self.C2b0 * np.exp(-self.r2b * (abs(P2 - Pb)**self.a2b))

        C3C = self.C3C0 * np.exp(-self.r3C * (abs(P3 - PC)**self.a3C))
        C4T = self.C4T0 * np.exp(-self.r4T * (abs(P4 - PT)**self.a4T))
        C5L = self.C5L0 * np.exp(-self.r5L * (abs(P5 - PL)**self.a5L))

        return C1a, C2a, C1b, C2b, C3C, C4T, C5L

    def derivatives(self, t, P, interpolators, infusion_rates=None):
        """
        Derivatives dP/dt for 5 compartmental pressures [P1, P2, P3, P4, P5].
        P1: Ventricles
        P2: Intracranial SAS
        P3: Cervical SAS
        P4: Thoracic SAS
        P5: Lumbar SAS
        """
        P1, P2, P3, P4, P5 = P[0], P[1], P[2], P[3], P[4]

        # Interpolate driving boundary pressures at time t
        Pa = float(interpolators['Pa'](t))  # Arterial pressure
        Pb = float(interpolators['Pb'](t))  # Intracranial venous pressure
        PC = float(interpolators['PC'](t))  # Cervical epidural
        PT = float(interpolators['PT'](t))  # Thoracic epidural
        PL = float(interpolators['PL'](t))  # Lumbar epidural

        # Numerical time derivatives of boundary driving pressures (dPa/dt, dPb/dt, etc.)
        dt_step = 1e-4
        dPa_dt = (float(interpolators['Pa'](t + dt_step)) - Pa) / dt_step
        dPb_dt = (float(interpolators['Pb'](t + dt_step)) - Pb) / dt_step
        dPC_dt = (float(interpolators['PC'](t + dt_step)) - PC) / dt_step
        dPT_dt = (float(interpolators['PT'](t + dt_step)) - PT) / dt_step
        dPL_dt = (float(interpolators['PL'](t + dt_step)) - PL) / dt_step

        # Calculate active non-linear compliances
        C1a, C2a, C1b, C2b, C3C, C4T, C5L = self.compute_compliances(
            P1, P2, P3, P4, P5, Pa, Pb, PC, PT, PL
        )

        # Inter-compartmental CSF flows
        F12 = (P1 - P2) * self.Z12
        F23 = (P2 - P3) * self.Z23
        F34 = (P3 - P4) * self.Z34
        F45 = (P4 - P5) * self.Z45

        # Resorption and Infusion rates
        F_res = self.resorption(P2)
        inf = infusion_rates if infusion_rates is not None else np.zeros(5)

        # Net volume rates of change (dV/dt) from mass conservation
        dV1_dt = -F12 + self.S1 + inf[0]
        dV2_dt =  F12 - F23 - F_res + inf[1]
        dV3_dt =  F23 - F34 + inf[2]
        dV4_dt =  F34 - F45 + inf[3]
        dV5_dt =  F45 + inf[4]

        # Pressure derivatives (dP/dt) incorporating arterial and venous boundary driving terms
        dPdt = np.zeros(5)
        
        # Ventricles (1): Total compliance C1_tot = C1a + C1b
        dPdt[0] = (dV1_dt + C1a * dPa_dt + C1b * dPb_dt) / (C1a + C1b)
        
        # Intracranial SAS (2): Total compliance C2_tot = C2a + C2b
        dPdt[1] = (dV2_dt + C2a * dPa_dt + C2b * dPb_dt) / (C2a + C2b)

        # Spinal canal (3, 4, 5)
        dPdt[2] = (dV3_dt + C3C * dPC_dt) / C3C
        dPdt[3] = (dV4_dt + C4T * dPT_dt) / C4T
        dPdt[4] = (dV5_dt + C5L * dPL_dt) / C5L

        return dPdt


def run_vanek_simulation(duration_s=10.0, infusion_rate_ml_min=0.0):
    """
    Executes a complete simulation of Vaněk's 3rd generation craniospinal model.
    """
    model = Vanek2015Model()

    # Time grid (500 Hz resolution)
    time = np.linspace(0, duration_s, int(duration_s * 500))

    # Synthetic driving pressure signals (Cardiac 1.2 Hz + Respiratory 0.25 Hz)
    Pa = 80.0 + 10.0 * np.sin(2.0 * np.pi * 1.2 * time)  # Arterial pressure (mean 80 mmHg)
    Pb = 10.0 + 1.5 * np.sin(2.0 * np.pi * 1.2 * time)   # Intracranial Venous pressure (mean 10 mmHg)
    PC = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time)   # Cervical epidural (mean 2 mmHg)
    PT = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time)   # Thoracic epidural (Respiration)
    PL = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time)   # Lumbar epidural

    interpolators = {
        'Pa': PchipInterpolator(time, Pa),
        'Pb': PchipInterpolator(time, Pb),
        'PC': PchipInterpolator(time, PC),
        'PT': PchipInterpolator(time, PT),
        'PL': PchipInterpolator(time, PL),
    }

    # Initial state: Resting compartmental pressures [P1, P2, P3, P4, P5]
    P0 = np.array([11.0, 10.0, 10.0, 10.0, 10.0])

    # Convert infusion rate from mL/min to mL/s (applied to Lumbar SAS - compartment 5)
    infusion_rates = np.zeros(5)
    infusion_rates[4] = infusion_rate_ml_min / 60.0

    # Solve system of ODEs
    sol = solve_ivp(
        fun=lambda t, P: model.derivatives(t, P, interpolators, infusion_rates),
        t_span=(0.0, duration_s),
        y0=P0,
        method='Radau',
        rtol=1e-5,
        atol=1e-5
    )

    t = sol.t
    P = sol.y  # Shape (5, N)

    # Reconstruct flows, volumes, and resorption over time
    F12 = (P[0] - P[1]) * model.Z12
    F23 = (P[1] - P[2]) * model.Z23
    F34 = (P[2] - P[3]) * model.Z34
    F45 = (P[3] - P[4]) * model.Z45

    resorption = np.array([model.resorption(p2) for p2 in P[1]])

    return {
        't': t,
        'P1': P[0], 'P2': P[1], 'P3': P[2], 'P4': P[3], 'P5': P[4],
        'F12': F12, 'F23': F23, 'F34': F34, 'F45': F45,
        'resorption': resorption
    }


def main():
    print("=== Petr Vaněk (2015) 3rd Generation Craniospinal CSF Model ===")
    results = run_vanek_simulation(duration_s=10.0)
    
    print("\nBaseline Simulation Results (Mean Values over 10s):")
    print(f"  - Ventricular Pressure (P1):      {np.mean(results['P1']):.2f} mmHg")
    print(f"  - Intracranial SAS Pressure (P2): {np.mean(results['P2']):.2f} mmHg (ICP Target: ~10.0)")
    print(f"  - Cervical SAS Pressure (P3):     {np.mean(results['P3']):.2f} mmHg")
    print(f"  - Thoracic SAS Pressure (P4):     {np.mean(results['P4']):.2f} mmHg")
    print(f"  - Lumbar SAS Pressure (P5):       {np.mean(results['P5']):.2f} mmHg")
    print(f"  - Peak Cervical CSF Flow (F23):   {np.max(abs(results['F23'])):.2f} mL/s")


if __name__ == "__main__":
    main()
