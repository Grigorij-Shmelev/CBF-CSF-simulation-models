"""
===============================================================================
Zbyšek Štěpáník (2009) 2nd Generation Craniospinal CSF Model Implementation
===============================================================================
Title: Compartmental Model of Cerebrospinal Fluid Transport Dynamics
       in the Craniospinal System (Autoreferat / Dissertation)
Author: Zbyšek Štěpáník (2009), Charles University, Prague.

This script implements Zbyšek Štěpáník's 2nd generation 5-compartment model 
of craniospinal cerebrospinal fluid (CSF) dynamics. The system models fluid 
transport across five anatomical compartments:
  1. Ventricles
  2. Intracranial Subarachnoid Space (SAS)
  3. Cervical SAS
  4. Thoracic SAS
  5. Lumbar SAS

The model incorporates non-linear exponential compliances, pressure-dependent 
linear resorption into dural veins, and coupling to cardiac (arterial/venous) 
and respiratory (epidural) driving pressure pulsations.

Usage:
    python stepanik_csf_simulation.py
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Stepanik2009Model:
    """2nd Generation 5-compartment craniospinal CSF model (Štěpáník 2009)."""

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
        self.C1A0 = 1.5e-4   # Arteries - Ventricles
        self.C2A0 = 0.20     # Arteries - Intracranial SAS
        self.C1V0 = 0.60     # Veins - Ventricles
        self.C2V0 = 0.60     # Veins - Intracranial SAS
        self.C3C0 = 1.125    # Cervical epidural interface
        self.C4T0 = 1.500    # Thoracic epidural interface
        self.C5L0 = 2.250    # Lumbar epidural interface

        # Compliance non-linearity parameters: curvature exponent (a) and slope (r)
        self.a1A, self.r1A = 0.3, 1.0
        self.a2A, self.r2A = 0.3, 1.0
        self.a1V, self.r1V = 0.6, 0.6
        self.a2V, self.r2V = 0.6, 0.6
        self.a3C, self.r3C = 0.6, 0.6
        self.a4T, self.r4T = 0.6, 0.6
        self.a5L, self.r5L = 0.6, 0.6

        # Baseline resting volumes [mL]
        self.V10 = 30.0   # Ventricular system
        self.V20 = 90.0   # Intracranial SAS
        self.V30 = 10.0   # Cervical SAS
        self.V40 = 10.0   # Thoracic SAS
        self.V50 = 10.0   # Lumbar SAS

        # Baseline relative pressure differentials [mmHg]
        self.p1A0 = 11.0 - 80.0  # -69.0 mmHg (Ventricles - Arterial)
        self.p2A0 = 10.0 - 80.0  # -70.0 mmHg (Intracranial SAS - Arterial)
        self.p1V0 = 10.0 - 2.0   # 8.0 mmHg (Ventricles - Venous)
        self.p2V0 = 10.0 - 2.0   # 8.0 mmHg (Intracranial SAS - Venous)
        self.p3C0 = 10.0 - 2.0   # 8.0 mmHg (Cervical - Cervical Epidural)
        self.p4T0 = 10.0 - 2.0   # 8.0 mmHg (Thoracic - Thoracic Epidural)
        self.p5L0 = 10.0 - 2.0   # 8.0 mmHg (Lumbar - Lumbar Epidural)

        # CSF Formation & Resorption
        self.S1 = 550.0 / 86400.0  # CSF production in ventricles (~0.006366 mL/s)
        self.S2 = 0.0
        self.S3 = 0.0
        self.S4 = 0.0
        self.S5 = 0.0

        self.peq = 10.0            # Equilibrium ICP [mmHg]
        self.alpha = 0.11 / 60.0   # Resorption conductance [mL / (mmHg * s)]
        self.Pmin = (self.alpha * self.peq - self.S1) / self.alpha  # Threshold (~2.0 mmHg)

        # Infusion configuration (5 compartments)
        self.infusion_rate = np.zeros(5)      # Rate [mL/s]
        self.infusion_start = np.zeros(5)     # Start time [s]
        self.infusion_duration = np.zeros(5)  # Duration [s]

        # Solver & Mode parameters
        self.Tmax = 1.0
        self.RelTol = 1e-3
        self.AbsTol = 1e-3
        self.InitialStep = 1e-3
        self.MaxStep = 0.1
        self.interpolate_results = False
        self.interpolation_period = 0.001
        
        # Mode switch: False = physiological p2 = p2V + pV, True = strict original MATLAB formula
        self.strict_matlab_mode = False

    def resorption(self, pressure):
        """Linear pressure-dependent CSF resorption into dural veins."""
        if pressure <= self.Pmin:
            return 0.0
        return -self.alpha * (pressure - self.Pmin)

    def infusion(self, t, index):
        """Calculates constant fluid infusion rate for compartment index (0-4) at time t."""
        start = self.infusion_start[index]
        duration = self.infusion_duration[index]
        if t < start or t > (start + duration):
            return 0.0
        return self.infusion_rate[index]

    def derivatives(self, t, p, interpolators):
        """
        Calculates time derivatives dp/dt for relative pressure differentials p.
        State vector p = [p1A, p2V, p3C, p4T, p5L]
        Uses shape-preserving PCHIP interpolation for driving boundary signals.
        """
        # Interpolate driving pressure differences at time t
        pAVi = float(interpolators['pAV'](t))
        pVCi = float(interpolators['pVC'](t))
        pCTi = float(interpolators['pCT'](t))
        pTLi = float(interpolators['pTL'](t))

        # Numerical derivative of pAVi (d(pA - pV)/dt)
        step_size = (np.finfo(float).eps)**(1.0 / 3.0) * max(1.0, abs(t))
        pAVii = float(interpolators['pAV'](t + step_size))
        dpAVi = (pAVii - pAVi) / step_size

        # Absolute boundary pressures
        pAi = float(interpolators['pA'](t))
        pVi = float(interpolators['pV'](t))

        # Active non-linear compartmental compliances
        C1Ai = self.C1A0 * np.exp(-self.r1A * (abs(p[0])**self.a1A))
        C2Ai = self.C2A0 * np.exp(-self.r2A * (abs(p[1] - pAVi)**self.a2A))
        C1Vi = self.C1V0 * np.exp(-self.r1V * (abs(p[0] + pAVi)**self.a1V))
        C2Vi = self.C2V0 * np.exp(-self.r2V * (abs(p[1])**self.a2V))
        C3Ci = self.C3C0 * np.exp(-self.r3C * (abs(p[2])**self.a3C))
        C4Ti = self.C4T0 * np.exp(-self.r4T * (abs(p[3])**self.a4T))
        C5Li = self.C5L0 * np.exp(-self.r5L * (abs(p[4])**self.a5L))

        # Resorption target pressure
        resorption_p = (p[1] + pAi) if self.strict_matlab_mode else (p[1] + pVi)

        # Relative pressure derivatives dp/dt
        dpdt = np.zeros(5)

        # Ventricles (1)
        dpdt[0] = (1.0 / (C1Ai + C1Vi)) * (
            -self.Z12 * p[0] + self.Z12 * p[1] - self.Z12 * pAVi
            + self.S1 + self.infusion(t, 0) - C1Vi * dpAVi
        )

        # Intracranial SAS (2)
        dpdt[1] = (1.0 / (C2Ai + C2Vi)) * (
            self.Z12 * p[0] - (self.Z12 + self.Z23) * p[1] + self.Z23 * p[2]
            + self.Z12 * pAVi - self.Z23 * pVCi + self.S2
            + self.infusion(t, 1) + self.resorption(resorption_p) + C2Ai * dpAVi
        )

        # Cervical SAS (3)
        dpdt[2] = (1.0 / C3Ci) * (
            self.Z23 * p[1] - (self.Z23 + self.Z34) * p[2] + self.Z34 * p[3]
            + self.Z23 * pVCi - self.Z34 * pCTi + self.S3 + self.infusion(t, 2)
        )

        # Thoracic SAS (4)
        dpdt[3] = (1.0 / C4Ti) * (
            self.Z34 * p[2] - (self.Z34 + self.Z45) * p[3] + self.Z45 * p[4]
            + self.Z34 * pCTi - self.Z45 * pTLi + self.S4 + self.infusion(t, 3)
        )

        # Lumbar SAS (5)
        dpdt[4] = (1.0 / C5Li) * (
            self.Z45 * p[3] - self.Z45 * p[4] + self.Z45 * pTLi
            + self.S5 + self.infusion(t, 4)
        )

        return dpdt


def run_stepanik_simulation(model=None, duration_s=30.0, save_plot_path=None):
    """
    Executes a complete simulation of Štěpáník's 2nd generation craniospinal model.
    """
    if model is None:
        model = Stepanik2009Model()
    
    model.Tmax = duration_s

    # Generate time grid (500 Hz resolution)
    time = np.linspace(0, duration_s, int(duration_s * 500))

    # Synthetic boundary driving pressure signals (Cardiac 1.0 Hz + Respiratory 0.25 Hz)
    pA = 80.0 + 10.0 * np.sin(2.0 * np.pi * 1.0 * time)  # Arterial pressure
    pV = 2.0 + 0.5 * np.sin(2.0 * np.pi * 0.25 * time)   # Venous pressure
    pC = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time)   # Cervical epidural
    pT = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time)   # Thoracic epidural (Respiration)
    pL = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time)   # Lumbar epidural

    pAV = pA - pV
    pVC = pV - pC
    pCT = pC - pT
    pTL = pT - pL

    # Pre-build shape-preserving PCHIP interpolators
    interpolators = {
        'pA': PchipInterpolator(time, pA),
        'pV': PchipInterpolator(time, pV),
        'pC': PchipInterpolator(time, pC),
        'pT': PchipInterpolator(time, pT),
        'pL': PchipInterpolator(time, pL),
        'pAV': PchipInterpolator(time, pAV),
        'pVC': PchipInterpolator(time, pVC),
        'pCT': PchipInterpolator(time, pCT),
        'pTL': PchipInterpolator(time, pTL),
    }

    # Initial state vector p0 = [p1A, p2V, p3C, p4T, p5L]
    p0 = np.array([model.p1A0, model.p2V0, model.p3C0, model.p4T0, model.p5L0], dtype=float)

    # Solve system of ODEs
    sol = solve_ivp(
        fun=lambda t, y: model.derivatives(t, y, interpolators),
        t_span=(0.0, float(duration_s)),
        y0=p0,
        method='RK45',
        rtol=model.RelTol,
        atol=model.AbsTol,
        max_step=model.MaxStep,
        first_step=model.InitialStep
    )

    t = sol.t
    p = sol.y.T  # Shape (N, 5)

    results = {'t': t, 'p': p}

    # Interpolate output if requested
    if model.interpolate_results:
        t_interp = np.arange(0, model.Tmax + model.interpolation_period, model.interpolation_period)
        p = PchipInterpolator(t, p, axis=0)(t_interp)
        t = t_interp
        results['t'] = t
        results['p'] = p

    # Reconstruct absolute driving pressures on simulation time grid
    results['pA'] = interpolators['pA'](t)
    results['pV'] = interpolators['pV'](t)
    results['pC'] = interpolators['pC'](t)
    results['pT'] = interpolators['pT'](t)
    results['pL'] = interpolators['pL'](t)

    # Reconstruct absolute compartmental pressures [mmHg]
    results['p1'] = p[:, 0] + results['pA']
    results['p2'] = p[:, 1] + results['pV']
    results['p3'] = p[:, 2] + results['pC']
    results['p4'] = p[:, 3] + results['pT']
    results['p5'] = p[:, 4] + results['pL']

    # Inter-compartmental fluid flows [mL/s]
    results['F12'] = (results['p1'] - results['p2']) / model.R12
    results['F23'] = (results['p2'] - results['p3']) / model.R23
    results['F34'] = (results['p3'] - results['p4']) / model.R34
    results['F45'] = (results['p4'] - results['p5']) / model.R45

    # Formation, Resorption, and Infusion rates [mL/s]
    n_points = len(t)
    results['formation'] = model.S1 * np.ones(n_points)
    
    if model.strict_matlab_mode:
        results['resorption'] = np.array([model.resorption(p[i, 1] + results['pA'][i]) for i in range(n_points)])
    else:
        results['resorption'] = np.array([model.resorption(results['p2'][i]) for i in range(n_points)])

    results['infusion1'] = np.array([model.infusion(t[i], 0) for i in range(n_points)])
    results['infusion2'] = np.array([model.infusion(t[i], 1) for i in range(n_points)])
    results['infusion3'] = np.array([model.infusion(t[i], 2) for i in range(n_points)])
    results['infusion4'] = np.array([model.infusion(t[i], 3) for i in range(n_points)])
    results['infusion5'] = np.array([model.infusion(t[i], 4) for i in range(n_points)])

    # Compartmental volumes [mL] using cumulative trapezoidal integration
    d_V1 = -results['F12'] + model.S1 + results['infusion1']
    d_V2 =  results['F12'] - results['F23'] + model.S2 + results['resorption'] + results['infusion2']
    d_V3 =  results['F23'] - results['F34'] + model.S3 + results['infusion3']
    d_V4 =  results['F34'] - results['F45'] + model.S4 + results['infusion4']
    d_V5 =  results['F45'] + model.S5 + results['infusion5']

    results['V1'] = cumulative_trapezoid(d_V1, t, initial=0) + model.V10
    results['V2'] = cumulative_trapezoid(d_V2, t, initial=0) + model.V20
    results['V3'] = cumulative_trapezoid(d_V3, t, initial=0) + model.V30
    results['V4'] = cumulative_trapezoid(d_V4, t, initial=0) + model.V40
    results['V5'] = cumulative_trapezoid(d_V5, t, initial=0) + model.V50

    # Non-linear compartmental compliances [mL/mmHg]
    results['C1A'] = model.C1A0 * np.exp(-model.r1A * (abs(results['p1'] - results['pA'])**model.a1A))
    results['C2A'] = model.C2A0 * np.exp(-model.r2A * (abs(results['p2'] - results['pA'])**model.a2A))
    results['C1V'] = model.C1V0 * np.exp(-model.r1V * (abs(results['p1'] - results['pV'])**model.a1V))
    results['C2V'] = model.C2V0 * np.exp(-model.r2V * (abs(results['p2'] - results['pV'])**model.a2V))
    results['C3C'] = model.C3C0 * np.exp(-model.r3C * (abs(results['p3'] - results['pC'])**model.a3C))
    results['C4T'] = model.C4T0 * np.exp(-model.r4T * (abs(results['p4'] - results['pT'])**model.a4T))
    results['C5L'] = model.C5L0 * np.exp(-model.r5L * (abs(results['p5'] - results['pL'])**model.a5L))

    if save_plot_path:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Plot Pressures
        ax = axes[0, 0]
        ax.plot(results['t'], results['p1'], label='p1 (Ventricles)')
        ax.plot(results['t'], results['p2'], label='p2 (Intracranial SAS / ICP)', color='crimson')
        ax.plot(results['t'], results['p3'], label='p3 (Cervical SAS)')
        ax.plot(results['t'], results['p4'], label='p4 (Thoracic SAS)')
        ax.plot(results['t'], results['p5'], label='p5 (Lumbar SAS)')
        ax.set_title("Craniospinal Pressures [mmHg]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [mmHg]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        # Plot Inter-compartmental Flows
        ax = axes[0, 1]
        ax.plot(results['t'], results['F12'], label='F12 (Ventr -> Cranial SAS)')
        ax.plot(results['t'], results['F23'], label='F23 (Cranial -> Cervical SAS)')
        ax.plot(results['t'], results['F34'], label='F34 (Cervical -> Thoracic SAS)')
        ax.plot(results['t'], results['F45'], label='F45 (Thoracic -> Lumbar SAS)')
        ax.set_title("Inter-compartmental CSF Flows [mL/s]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flow [mL/s]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        # Plot Volumes
        ax = axes[1, 0]
        ax.plot(results['t'], results['V1'], label='V1 (Ventricles)')
        ax.plot(results['t'], results['V2'], label='V2 (Intracranial SAS)')
        ax.plot(results['t'], results['V3'], label='V3 (Cervical)')
        ax.plot(results['t'], results['V4'], label='V4 (Thoracic)')
        ax.plot(results['t'], results['V5'], label='V5 (Lumbar)')
        ax.set_title("Compartmental Volumes [mL]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Volume [mL]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        # Plot Formation and Resorption
        ax = axes[1, 1]
        ax.plot(results['t'], results['formation'] * 60, label='Formation (S1)', color='green')
        ax.plot(results['t'], -results['resorption'] * 60, label='Resorption (-Fout)', color='red')
        ax.set_title("Formation & Resorption Mass Balance [mL/min]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Rate [mL/min]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        # Plot Global P-V Curve
        ax = axes[2, 0]
        dV_total = (results['V1'] + results['V2'] + results['V3'] + results['V4'] + results['V5']) - \
                   (model.V10 + model.V20 + model.V30 + model.V40 + model.V50)
        ax.plot(dV_total, results['p2'], color='purple', label='Global P-V Curve')
        ax.set_title("Global Pressure-Volume Relationship")
        ax.set_xlabel("Total CSF Volume Change ΔV [mL]")
        ax.set_ylabel("ICP p2 [mmHg]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        # Plot Compliances
        ax = axes[2, 1]
        ax.plot(results['t'], results['C1A'], label='C1A (Artery-Ventricle)')
        ax.plot(results['t'], results['C2A'], label='C2A (Artery-Cranial SAS)')
        ax.plot(results['t'], results['C3C'], label='C3C (Cervical)')
        ax.plot(results['t'], results['C4T'], label='C4T (Thoracic)')
        ax.plot(results['t'], results['C5L'], label='C5L (Lumbar)')
        ax.set_title("Compartmental Compliances [mL/mmHg]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Compliance [mL/mmHg]")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=150)
        plt.close()

    return results


def main():
    print("=== Zbyšek Štěpáník (2009) 2nd Generation Craniospinal Model Simulation ===")
    results = run_stepanik_simulation(duration_s=10.0)
    
    print("\nBaseline Simulation Results (Mean Values over 10s):")
    print(f"  - Ventricular Pressure (p1):      {np.mean(results['p1']):.2f} mmHg")
    print(f"  - Intracranial SAS Pressure (p2): {np.mean(results['p2']):.2f} mmHg (ICP Target: ~10.0)")
    print(f"  - Lumbar SAS Pressure (p5):       {np.mean(results['p5']):.2f} mmHg")
    print(f"  - Peak Cervical CSF Flow (F23):   {np.max(abs(results['F23'])):.2f} mL/s")


if __name__ == "__main__":
    main()
