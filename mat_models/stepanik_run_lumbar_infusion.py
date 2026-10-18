"""
===============================================================================
Zbyšek Štěpáník (2009) 2nd Generation Craniospinal CSF Model — Lumbar Infusion Test
===============================================================================
Executes a standard Lumbar Infusion Test (LIT) on Zbyšek Štěpáník's (2009) 2nd 
generation 5-compartment model of craniospinal cerebrospinal fluid (CSF) dynamics.

Protocol matching run_vanek_lumbar_infusion.py:
  - Baseline Phase: t = 0 to 60s
  - Infusion Phase: t = 60 to 360s (1.5 mL/min = 0.025 mL/s into Lumbar SAS / compartment 5)
  - Recovery Phase: t = 360 to 500s

Usage:
    python stepanik_run_lumbar_infusion.py
"""

import sys
sys.path.append('/workspace/artifacts')
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from stepanik_2009_csf_model import Stepanik2009Model, run_stepanik_simulation


def run_stepanik_lumbar_infusion_test():
    """
    Executes a standard Lumbar Infusion Test (LIT) on Zbyšek Štěpáník's (2009) 2nd Generation Model.
    
    Protocol:
    - Baseline Phase: t = 0 to 60s
    - Infusion Phase: t = 60 to 360s (1.5 mL/min = 0.025 mL/s into Lumbar SAS / compartment 5)
    - Recovery Phase: t = 360 to 500s
    """
    model = Stepanik2009Model()
    
    duration_s = 500.0
    model.Tmax = duration_s
    
    # Configure infusion into compartment 5 (Lumbar SAS, index 4)
    model.infusion_rate[4] = 0.025      # 1.5 mL/min = 0.025 mL/s
    model.infusion_start[4] = 60.0       # start at t = 60s
    model.infusion_duration[4] = 300.0   # duration = 300s (ends at t = 360s)
    
    print("Running Lumbar Infusion Test simulation on Štěpáník (2009) model (500s)...")
    
    # Execute full 500s simulation
    results = run_stepanik_simulation(model=model, duration_s=duration_s, save_plot_path=None)
    
    t = results['t']
    p1, p2, p3, p4, p5 = results['p1'], results['p2'], results['p3'], results['p4'], results['p5']
    F12, F23, F34, F45 = results['F12'], results['F23'], results['F34'], results['F45']
    V1, V2, V3, V4, V5 = results['V1'], results['V2'], results['V3'], results['V4'], results['V5']
    
    resorption = -results['resorption']  # Convert negative resorption to positive magnitude [mL/s]
    infusion_i5 = results['infusion5']
    
    C1A, C1V = results['C1A'], results['C1V']
    C2A, C2V = results['C2A'], results['C2V']
    C3C, C4T, C5L = results['C3C'], results['C4T'], results['C5L']

    # Calculate key clinical metrics
    baseline_mask = (t >= 10) & (t <= 50)
    infusion_plateau_mask = (t >= 300) & (t <= 350)

    p0_icp = np.mean(p2[baseline_mask])
    pinf_icp = np.mean(p2[infusion_plateau_mask])
    dp_icp = pinf_icp - p0_icp

    p0_lumb = np.mean(p5[baseline_mask])
    pinf_lumb = np.mean(p5[infusion_plateau_mask])
    dp_lumb = pinf_lumb - p0_lumb

    I_inf = 0.025  # mL/s
    R_out_calc = dp_icp / I_inf  # mmHg * s / mL
    R_out_clinical = R_out_calc / 60.0  # mmHg / (mL/min)

    print("\n--- Infusion Test Metrics ---")
    print(f"Baseline ICP (P2): {p0_icp:.2f} mmHg ({p0_icp * 13.6:.1f} mmH2O)")
    print(f"Plateau ICP (P2):  {pinf_icp:.2f} mmHg ({pinf_icp * 13.6:.1f} mmH2O)")
    print(f"ICP Elevation (ΔP): {dp_icp:.2f} mmHg ({dp_icp * 13.6:.1f} mmH2O)")
    print(f"Baseline Lumbar Pressure (P5): {p0_lumb:.2f} mmHg ({p0_lumb * 13.6:.1f} mmH2O)")
    print(f"Plateau Lumbar Pressure (P5):  {pinf_lumb:.2f} mmHg ({pinf_lumb * 13.6:.1f} mmH2O)")
    print(f"Calculated Outflow Resistance Rout: {R_out_calc:.2f} mmHg*s/mL ({R_out_clinical:.2f} mmHg/(mL/min))")

    # Generate 6-panel visualization figure matching run_vanek_lumbar_infusion style
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle("Lumbar Infusion Test (LIT) Simulation — Štěpáník (2009) 2nd Gen Model", fontsize=14, fontweight='bold')

    # Panel 1: Craniospinal Pressures
    ax = axes[0, 0]
    ax.plot(t, p1, label='P1 (Ventricles)', alpha=0.8)
    ax.plot(t, p2, label='P2 (Intracranial SAS / ICP)', linewidth=1.5, color='crimson')
    ax.plot(t, p5, label='P5 (Lumbar SAS)', linewidth=1.5, color='darkblue', linestyle='--')
    ax.axvspan(60, 360, color='orange', alpha=0.15, label='Infusion Phase (1.5 mL/min)')
    ax.set_title("Craniospinal Pressures During Infusion [mmHg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 2: Fluid Mass Balance
    ax = axes[0, 1]
    ax.plot(t, results['formation'] * 60, label='Formation (S1)', color='green')
    ax.plot(t, resorption * 60, label='Resorption (F_res)', color='red')
    ax.plot(t, infusion_i5 * 60, label='Lumbar Infusion (I5)', color='orange', linewidth=2)
    ax.set_title("Fluid Mass Balance [mL/min]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rate [mL/min]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='center right', fontsize=8)

    # Panel 3: Inter-compartmental Flows
    ax = axes[1, 0]
    ax.plot(t, F12 * 60, label='F12 (Ventr -> Cranial SAS)', alpha=0.7)
    ax.plot(t, F23 * 60, label='F23 (Cranial -> Cervical SAS)', alpha=0.7)
    ax.plot(t, F34 * 60, label='F34 (Cervical -> Thoracic SAS)', alpha=0.7)
    ax.plot(t, F45 * 60, label='F45 (Thoracic -> Lumbar SAS)', alpha=0.7)
    ax.set_title("Inter-compartmental CSF Flows [mL/min]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flow [mL/min]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 4: Compartmental Volumes
    ax = axes[1, 1]
    ax.plot(t, V1, label='V1 (Ventricles)')
    ax.plot(t, V2, label='V2 (Intracranial SAS)')
    ax.plot(t, V3, label='V3 (Cervical)')
    ax.plot(t, V4, label='V4 (Thoracic)')
    ax.plot(t, V5, label='V5 (Lumbar)')
    ax.set_title("Compartmental Volumes [mL]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Volume [mL]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 5: Non-linear Compliances
    ax = axes[2, 0]
    ax.plot(t, C1A, label='C1A (Arterial-Ventricle)', alpha=0.7)
    ax.plot(t, C1V, label='C1V (Venous-Ventricle)', alpha=0.7)
    ax.plot(t, C2A, label='C2A (Arterial-Cranial SAS)', alpha=0.7)
    ax.plot(t, C2V, label='C2V (Venous-Cranial SAS)', alpha=0.7)
    ax.plot(t, C5L, label='C5L (Lumbar)', alpha=0.7)
    ax.set_title("Non-linear Compliances [mL/mmHg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Compliance [mL/mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 6: Dynamic Pressure-Volume Curve
    ax = axes[2, 1]
    dV_total = (V1 + V2 + V3 + V4 + V5) - (model.V10 + model.V20 + model.V30 + model.V40 + model.V50)
    ax.plot(dV_total, p2, color='purple', label='Global P-V Curve')
    ax.set_title("Dynamic Pressure-Volume (P-V) Curve")
    ax.set_xlabel("Total CSF Volume Change ΔV [mL]")
    ax.set_ylabel("ICP P2 [mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = "stepanik_lumbar_infusion_test.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot successfully saved to {plot_path}")

    return {
        't': t, 'p1': p1, 'p2': p2, 'p5': p5,
        'p0_icp': p0_icp, 'pinf_icp': pinf_icp, 'R_out_clinical': R_out_clinical
    }


if __name__ == "__main__":
    run_stepanik_lumbar_infusion_test()
