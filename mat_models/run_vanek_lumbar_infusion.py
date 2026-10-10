import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from vanek_2015_model import Vanek2015Model

def run_vanek_lumbar_infusion_test():
    """
    Executes a standard Lumbar Infusion Test (LIT) on Petr Vaněk's (2015) 3rd Generation Model.
    
    Protocol matching run_lumbar_infusion.txt:
    - Baseline Phase: t = 0 to 60s
    - Infusion Phase: t = 60 to 360s (1.5 mL/min = 0.025 mL/s into Lumbar SAS / compartment 5)
    - Recovery Phase: t = 360 to 500s
    """
    model = Vanek2015Model()
    
    duration_s = 500.0
    time = np.linspace(0, duration_s, int(duration_s * 100))  # 100 Hz sampling
    
    # Driving boundary pressures (Cardiac 1.2 Hz + Respiratory 0.25 Hz)
    Pa = 80.0 + 10.0 * np.sin(2.0 * np.pi * 1.2 * time)  # Arterial pressure
    Pb = 10.0 + 1.5 * np.sin(2.0 * np.pi * 1.2 * time)   # Venous pressure (b)
    PC = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time)   # Cervical epidural
    PT = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time)   # Thoracic epidural
    PL = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time)   # Lumbar epidural

    interpolators = {
        'Pa': PchipInterpolator(time, Pa),
        'Pb': PchipInterpolator(time, Pb),
        'PC': PchipInterpolator(time, PC),
        'PT': PchipInterpolator(time, PT),
        'PL': PchipInterpolator(time, PL),
    }

    # Infusion function (0.025 mL/s = 1.5 mL/min between t=60s and t=360s into Lumbar SAS)
    def infusion_rate_func(t):
        inf = np.zeros(5)
        if 60.0 <= t <= 360.0:
            inf[4] = 0.025  # mL/s
        return inf

    # Initial resting pressures [P1, P2, P3, P4, P5]
    P0 = np.array([11.0, 10.0, 10.0, 10.0, 10.0])

    print("Running Lumbar Infusion Test simulation on Vaněk (2015) model (500s)...")
    
    # Solve system of ODEs
    sol = solve_ivp(
        fun=lambda t, P: model.derivatives(t, P, interpolators, infusion_rates=infusion_rate_func(t)),
        t_span=(0.0, duration_s),
        y0=P0,
        method='Radau',
        rtol=1e-5,
        atol=1e-5
    )

    t = sol.t
    P = sol.y  # Shape (5, N)
    p1, p2, p3, p4, p5 = P[0], P[1], P[2], P[3], P[4]

    # Reconstruct flows, volumes, and resorption
    F12 = (p1 - p2) * model.Z12
    F23 = (p2 - p3) * model.Z23
    F34 = (p3 - p4) * model.Z34
    F45 = (p4 - p5) * model.Z45

    resorption = np.array([model.resorption(p2_i) for p2_i in p2])
    infusion_i5 = np.array([infusion_rate_func(t_i)[4] for t_i in t])

    # Reconstruct drive arterial/venous pressures on solver time grid
    Pa_t = interpolators['Pa'](t)
    Pb_t = interpolators['Pb'](t)
    PC_t = interpolators['PC'](t)
    PT_t = interpolators['PT'](t)
    PL_t = interpolators['PL'](t)

    # Compute active compliances over time
    compliances = [model.compute_compliances(p1[i], p2[i], p3[i], p4[i], p5[i], Pa_t[i], Pb_t[i], PC_t[i], PT_t[i], PL_t[i]) for i in range(len(t))]
    C1a = np.array([c[0] for c in compliances])
    C2a = np.array([c[1] for c in compliances])
    C1b = np.array([c[2] for c in compliances])
    C2b = np.array([c[3] for c in compliances])
    C3C = np.array([c[4] for c in compliances])
    C4T = np.array([c[5] for c in compliances])
    C5L = np.array([c[6] for c in compliances])

    # Compute volume changes via cumulative integration
    dV1_dt = -F12 + model.S1
    dV2_dt = F12 - F23 - resorption
    dV3_dt = F23 - F34
    dV4_dt = F34 - F45
    dV5_dt = F45 + infusion_i5

    V1 = cumulative_trapezoid(dV1_dt, t, initial=0) + model.V10
    V2 = cumulative_trapezoid(dV2_dt, t, initial=0) + model.V20
    V3 = cumulative_trapezoid(dV3_dt, t, initial=0) + model.V30
    V4 = cumulative_trapezoid(dV4_dt, t, initial=0) + model.V40
    V5 = cumulative_trapezoid(dV5_dt, t, initial=0) + model.V50

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

    # Generate plot figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle("Lumbar Infusion Test (LIT) Simulation — Vaněk (2015) 3rd Gen Model", fontsize=14, fontweight='bold')

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
    ax.plot(t, np.ones_like(t) * model.S1 * 60, label='Formation (S1)', color='green')
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
    ax.plot(t, C1a, label='C1a (Arterial-Ventricle)', alpha=0.7)
    ax.plot(t, C1b, label='C1b (Venous-Ventricle)', alpha=0.7)
    ax.plot(t, C2a, label='C2a (Arterial-Cranial SAS)', alpha=0.7)
    ax.plot(t, C2b, label='C2b (Venous-Cranial SAS)', alpha=0.7)
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
    plot_path = "vanek_lumbar_infusion_test.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot successfully saved to {plot_path}")

    return {
        't': t, 'p1': p1, 'p2': p2, 'p5': p5,
        'p0_icp': p0_icp, 'pinf_icp': pinf_icp, 'R_out_clinical': R_out_clinical
    }

if __name__ == "__main__":
    run_vanek_lumbar_infusion_test()
