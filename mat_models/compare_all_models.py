import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from ursino_2010_model import Ursino2010Model
from stepanik_2009_csf_model import Stepanik2009Model, run_stepanik_simulation
from vanek_2015_model import Vanek2015Model, run_vanek_simulation

print("===========================================================================")
print("                      MULTI-MODEL COMPARISON SCRIPT")
print("===========================================================================")

# %% [1] Run Ursino (2010) Cerebrovascular Simulation
print("\n1. Running Ursino (2010) 3-Phase Simulation (Baseline -> Stenosis -> Hypercapnia)...")
ursino_model = Ursino2010Model()
y0_u = ursino_model.get_initial_state()

# Phase 1: Baseline (0 to 50s) -> Steady state baseline
sol_u1 = solve_ivp(lambda t, y: ursino_model.derivatives(t, y), [0, 50], y0_u, method='Radau')
y_ss = sol_u1.y[:, -1]

# Phase 2: Left ICA Occlusion (50 to 100s) -> Steady state occluded
sol_u2 = solve_ivp(lambda t, y: ursino_model.derivatives(t, y, ica_stenosis_l=500.0), [50, 100], y_ss, method='Radau')
y_occ = sol_u2.y[:, -1]

# Phase 3: Hypercapnia Challenge (100 to 250s, PaCO2 = 60 mmHg) -> Steady state hypercapnia
sol_u3 = solve_ivp(lambda t, y: ursino_model.derivatives(t, y, PaCO2=60.0, ica_stenosis_l=500.0), [100, 250], y_occ, method='Radau')
y_hyp = sol_u3.y[:, -1]

t_u = np.concatenate([sol_u1.t, sol_u2.t, sol_u3.t])
y_u = np.hstack([sol_u1.y, sol_u2.y, sol_u3.y])

q_mca_l, q_mca_r = [], []
for i in range(len(t_u)):
    st = y_u[:, i]
    sten = 500.0 if t_u[i] >= 50 else 1.0
    fl = ursino_model.get_flow_dict(st, ica_stenosis_l=sten)
    q_mca_l.append(fl['q_mca_l'])
    q_mca_r.append(fl['q_mca_r'])

f_base = ursino_model.get_flow_dict(y_ss)
f_occ = ursino_model.get_flow_dict(y_occ, ica_stenosis_l=500.0)
f_hyp = ursino_model.get_flow_dict(y_hyp, ica_stenosis_l=500.0)

print(f"   - Left MCA Baseline Flow:					{f_base['q_mca_l']:.2f} mL/s")
print(f"   - Left MCA Occluded Flow:					{f_occ['q_mca_l']:.2f} mL/s")
print(f"   - Left MCA Post-Hypercapnia Flow:			{f_hyp['q_mca_l']:.2f} mL/s (Steal phenomenon)")
print(f"   - Contralateral MCA Post-Hypercapnia Flow:   {f_hyp['q_mca_r']:.2f} mL/s (Contralateral surge)")

# %% [2] Run Štěpáník (2009) and Vaněk (2015) CSF Infusion Simulations
print("\n2. Running Lumbar Infusion Test (1.5 mL/min for 300s) on Štěpáník vs. Vaněk...")
duration_s = 500.0

# 2a. Štěpáník (2009) Model under Native Boundary (pV = 2.0 mmHg)
stepanik_model = Stepanik2009Model()
stepanik_model.infusion_rate[4] = 0.025  # 1.5 mL/min = 0.025 mL/s into Lumbar SAS
stepanik_model.infusion_start[4] = 60.0
stepanik_model.infusion_duration[4] = 300.0

res_step = run_stepanik_simulation(model=stepanik_model, duration_s=duration_s)

# 2b. Vaněk (2015) Model under Native Boundary (Pb = 10.0 mmHg)
res_vanek = run_vanek_simulation(duration_s=duration_s, infusion_rate_ml_min=1.5)

# Calculate metrics
t_s = res_step['t']
mask_base_s = (t_s >= 10) & (t_s <= 50)
mask_plat_s = (t_s >= 300) & (t_s <= 350)
p0_s = np.mean(res_step['p2'][mask_base_s])
pinf_s = np.mean(res_step['p2'][mask_plat_s])
dp_s = pinf_s - p0_s
Rout_s = (dp_s / 0.025) / 60.0

t_v = res_vanek['t']
mask_base_v = (t_v >= 10) & (t_v <= 50)
mask_plat_v = (t_v >= 300) & (t_v <= 350)
p0_v = np.mean(res_vanek['P2'][mask_base_v])
pinf_v = np.mean(res_vanek['P2'][mask_plat_v])
dp_v = pinf_v - p0_v
Rout_v = (dp_v / 0.025) / 60.0

# %% [3] Print Quantitative Comparison Summary Table
print("\n" + "="*80)
print(f"{'Metric / Parameter':<38} | {'Štěpáník (2009)':<17} | {'Vaněk (2015)':<17}")
print("="*80)
print(f"{'Native Dural Sinus Pressure (pV/Pb)':<38} | {'2.0 mmHg':<17} | {'10.0 mmHg':<17}")
print(f"{'Baseline ICP (P2_0)':<38} | {p0_s:<17.2f} | {p0_v:<17.2f} mmHg")
print(f"{'5-Min Peak Infusion ICP (P2_inf)':<38} | {pinf_s:<17.2f} | {pinf_v:<17.2f} mmHg")
print(f"{'5-Min Peak ICP in mmH2O':<38} | {pinf_s*13.6:<17.1f} | {pinf_v*13.6:<17.1f} mmH2O")
print(f"{'ICP Elevation (ΔP)':<38} | {dp_s:<17.2f} | {dp_v:<17.2f} mmHg")
print(f"{'Calculated Outflow Resistance Rout':<38} | {Rout_s:<17.2f} | {Rout_v:<17.2f} mmHg/(mL/min)")
print("="*80)

# %% [4] Generate Visual Comparison Figures
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: Ursino Flow Steal
ax = axes[0, 0]
ax.plot(t_u, q_mca_l, 'r-', label='Left MCA Flow (Ischemic)', linewidth=1.8)
ax.plot(t_u, q_mca_r, 'b--', label='Right MCA Flow (Healthy)', linewidth=1.8)
ax.axvline(50, color='k', linestyle=':', label='100% Left ICA Occlusion')
ax.axvline(100, color='g', linestyle=':', label='Hypercapnia (PaCO2=60 mmHg)')
ax.set_title("Ursino (2010): Microvascular Steal Phenomenon", fontweight='bold')
ax.set_xlabel("Time [s]"); ax.set_ylabel("Blood Flow [mL/s]")
ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8)

# Panel 2: Ursino ICP
ax = axes[0, 1]
ax.plot(t_u, y_u[0], 'purple', label='Intracranial Pressure (Pic)', linewidth=1.8)
ax.axvline(50, color='k', linestyle=':')
ax.axvline(100, color='g', linestyle=':')
ax.set_title("Ursino (2010): Intracranial Pressure (Pic)", fontweight='bold')
ax.set_xlabel("Time [s]"); ax.set_ylabel("Pressure [mmHg]")
ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8)

# Panel 3: Štěpáník vs. Vaněk ICP Comparison
ax = axes[1, 0]
ax.plot(t_s, res_step['p2'], 'r-', label='Štěpáník (2009) P2 (ICP)', linewidth=2.0)
ax.plot(t_v, res_vanek['P2'], 'g--', label='Vaněk (2015) P2 (ICP)', linewidth=2.0)
ax.axvspan(60, 360, color='orange', alpha=0.15, label='Infusion Phase (1.5 mL/min)')
ax.set_title("Štěpáník vs. Vaněk: Native ICP Elevation During Infusion", fontweight='bold')
ax.set_xlabel("Time [s]"); ax.set_ylabel("ICP P2 [mmHg]")
ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8)

# Panel 4: Štěpáník vs. Vaněk Lumbar Pressure Comparison
ax = axes[1, 1]
ax.plot(t_s, res_step['p5'], 'm-', label='Štěpáník (2009) P5 (Lumbar)', linewidth=1.8)
ax.plot(t_v, res_vanek['P5'], 'c--', label='Vaněk (2015) P5 (Lumbar)', linewidth=1.8)
ax.axvspan(60, 360, color='orange', alpha=0.15)
ax.set_title("Štěpáník vs. Vaněk: Lumbar SAS Pressure (P5)", fontweight='bold')
ax.set_xlabel("Time [s]"); ax.set_ylabel("Lumbar Pressure P5 [mmHg]")
ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize=8)

plt.tight_layout()
plot_path = "model_comparison_plots.png"
plt.savefig(plot_path, dpi=150)
print("\nPlot saved successfully to ./model_comparison_plots.png!")
