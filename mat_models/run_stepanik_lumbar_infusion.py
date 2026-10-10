import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# Load stepanik_csf_simulation.py dynamically
spec = importlib.util.spec_from_file_location("csf_sim", "./stepanik_csf_simulation.py")
csf_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(csf_sim)

CSFAttributes = csf_sim.CSFAttributes
CSF_model = csf_sim.CSF_model

def run_lumbar_infusion_test():
    # Configure parameters
    params = CSFAttributes()
    
    # Infusion parameters: 0.025 ml/s (= 1.5 ml/min standard LIT) into compartment 5 (Lumbar SAS, index 4)
    # Baseline phase: 0 to 60s
    # Infusion phase: 60s to 360s (300s duration = 5 min)
    # Recovery phase: 360s to 500s
    duration_s = 500.0
    params.Tmax = duration_s
    
    # Compartment index 4 = Lumbar SAS
    params.rychlost_infuze[4] = 0.025  # ml/s
    params.start_infuze[4] = 60.0      # s
    params.doba_infuze[4] = 300.0      # s
    
    # Generate time grid and driving pressures (100 Hz sampling)
    time = np.linspace(0, duration_s, int(duration_s * 100))
    
    # Cardiac (1 Hz) and Respiratory (0.25 Hz) driving pressures
    pA = 80.0 + 8.0 * np.sin(2.0 * np.pi * 1.0 * time)
    pV = 2.0 + 0.5 * np.sin(2.0 * np.pi * 0.25 * time)
    pC = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time)
    pT = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time)
    pL = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time)
    
    inputs = {
        'time': time,
        'pA': pA,
        'pV': pV,
        'pC': pC,
        'pT': pT,
        'pL': pL
    }
    
    settings = {'parameters': params, 'inputs': inputs}
    
    print("Running Lumbar Infusion Test simulation on Stepanik (2009) model (500s)...")
    results = CSF_model(settings)
    
    t = results['t']
    p1 = results['p1'] # Ventricles
    p2 = results['p2'] # Intracranial SAS (ICP)
    p5 = results['p5'] # Lumbar SAS
    
    # Calculate key metrics
    baseline_mask = (t >= 10) & (t <= 50)
    infusion_plateau_mask = (t >= 300) & (t <= 350)
    
    p0_icp = np.mean(p2[baseline_mask])
    pinf_icp = np.mean(p2[infusion_plateau_mask])
    dp_icp = pinf_icp - p0_icp
    
    p0_lumb = np.mean(p5[baseline_mask])
    pinf_lumb = np.mean(p5[infusion_plateau_mask])
    dp_lumb = pinf_lumb - p0_lumb
    
    # Outflow resistance calculation: Rout = (P_inf - P_0) / I_inf
    I_inf = 0.025 # ml/s
    R_out_calc = dp_icp / I_inf # mmHg * s / ml
    R_out_clinical = R_out_calc / 60.0 # mmHg / (ml/min)
    
    print("\n--- Infusion Test Metrics ---")
    print(f"Baseline ICP (p2): {p0_icp:.2f} mmHg ({p0_icp * 13.6:.1f} mmH2O)")
    print(f"Plateau ICP (p2): {pinf_icp:.2f} mmHg ({pinf_icp * 13.6:.1f} mmH2O)")
    print(f"ICP Elevation (ΔP): {dp_icp:.2f} mmHg ({dp_icp * 13.6:.1f} mmH2O)")
    print(f"Baseline Lumbar Pressure (p5): {p0_lumb:.2f} mmHg ({p0_lumb * 13.6:.1f} mmH2O)")
    print(f"Plateau Lumbar Pressure (p5): {pinf_lumb:.2f} mmHg ({pinf_lumb * 13.6:.1f} mmH2O)")
    print(f"Calculated Outflow Resistance Rout: {R_out_calc:.2f} mmHg*s/ml ({R_out_clinical:.2f} mmHg/(ml/min))")
    
    # Create high-quality figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle("Lumbar Infusion Test (LIT) Simulation — 5-Compartment Model", fontsize=15, fontweight='bold')
    
    # Panel 1: Craniospinal Pressures
    ax = axes[0, 0]
    ax.plot(t, p1, label='p1 (Ventricles)', alpha=0.8)
    ax.plot(t, p2, label='p2 (Intracranial SAS / ICP)', linewidth=1.5, color='crimson')
    ax.plot(t, p5, label='p5 (Lumbar SAS)', linewidth=1.5, color='darkblue', linestyle='--')
    ax.axvspan(60, 360, color='orange', alpha=0.15, label='Infusion Phase (1.5 ml/min)')
    ax.set_title("Craniospinal Pressures During Infusion [mmHg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel 2: Infusion and Resorption Rates
    ax = axes[0, 1]
    ax.plot(t, results['tvorba'] * 60, label='Formation (S1)', color='green')
    ax.plot(t, -results['vstrebavani'] * 60, label='Resorption (-Fout)', color='red')
    ax.plot(t, results['infuze5'] * 60, label='Lumbar Infusion (I5)', color='orange', linewidth=2)
    ax.set_title("Fluid Mass Balance [ml/min]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rate [ml/min]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='center right', fontsize=8)
    
    # Panel 3: Inter-compartmental Flows
    ax = axes[1, 0]
    ax.plot(t, results['F12'] * 60, label='F12 (Ventr -> Cranial SAS)', alpha=0.7)
    ax.plot(t, results['F23'] * 60, label='F23 (Cranial -> Cervical SAS)', alpha=0.7)
    ax.plot(t, results['F34'] * 60, label='F34 (Cervical -> Thoracic SAS)', alpha=0.7)
    ax.plot(t, results['F45'] * 60, label='F45 (Thoracic -> Lumbar SAS)', alpha=0.7)
    ax.set_title("Inter-compartmental CSF Flows [ml/min]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flow [ml/min]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel 4: Compartmental Volumes
    ax = axes[1, 1]
    ax.plot(t, results['V1'], label='V1 (Ventricles)')
    ax.plot(t, results['V2'], label='V2 (Intracranial SAS)')
    ax.plot(t, results['V3'], label='V3 (Cervical)')
    ax.plot(t, results['V4'], label='V4 (Thoracic)')
    ax.plot(t, results['V5'], label='V5 (Lumbar)')
    ax.set_title("Compartmental Volumes [ml]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Volume [ml]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel 5: Non-linear Compliances
    ax = axes[2, 0]
    ax.plot(t, results['C1A'], label='C1A (Vascular-Ventricle)', alpha=0.7)
    ax.plot(t, results['C2A'], label='C2A (Vascular-Cranial SAS)', alpha=0.7)
    ax.plot(t, results['C3C'], label='C3C (Cervical)', alpha=0.7)
    ax.plot(t, results['C4T'], label='C4T (Thoracic)', alpha=0.7)
    ax.plot(t, results['C5L'], label='C5L (Lumbar)', alpha=0.7)
    ax.set_title("Non-linear Compartmental Compliances [ml/mmHg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Compliance [ml/mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel 6: Pressure-Volume Relationship
    ax = axes[2, 1]
    dV_total = (results['V1'] + results['V2'] + results['V3'] + results['V4'] + results['V5']) - \
               (params.V10 + params.V20 + params.V30 + params.V40 + params.V50)
    ax.plot(dV_total, p2, color='purple', label='Global P-V Curve')
    ax.set_title("Dynamic Pressure-Volume (P-V) Curve")
    ax.set_xlabel("Total CSF Volume Change ΔV [ml]")
    ax.set_ylabel("ICP p2 [mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = "./lumbar_infusion_test_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot successfully generated and saved to {plot_path}")
    
    return results

if __name__ == "__main__":
    run_lumbar_infusion_test()
