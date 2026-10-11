import sys
sys.path.append('/workspace/artifacts')
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from ursino_2010_model import Ursino2010Model
from stepanik_2009_csf_model import Stepanik2009Model
from vanek_2015_model import Vanek2015Model

print("All model modules successfully loaded!")

# %% [2] Run Ursino (2010) Cerebrovascular Simulation
print("\n1. Running Ursino (2010) 3-Phase Simulation (Baseline -> Stenosis -> Hypercapnia)...")
ursino_model = Ursino2010Model()
y0_u = ursino_model.get_initial_state()

sol_u1 = solve_ivp(lambda t, y: ursino_model.derivatives(t, y), [0, 50], y0_u, method='Radau')
y_ss = sol_u1.y[:, -1]

sol_u2 = solve_ivp(lambda t, y: ursino_model.derivatives(t, y, ica_stenosis_l=500.0), [50, 100], y_ss, method='Radau')

sol_u3 = solve_ivp(lambda t, y: ursino_model.derivatives(t, y, PaCO2=60.0, ica_stenosis_l=500.0), [100, 250], sol_u2.y[:, -1], method='Radau')

t_u = np.concatenate([sol_u1.t, sol_u2.t, sol_u3.t])
y_u = np.hstack([sol_u1.y, sol_u2.y, sol_u3.y])

q_mca_l, q_mca_r = [], []
for i in range(len(t_u)):
    st = y_u[:, i]
    sten = 500.0 if t_u[i] >= 50 else 1.0
    fl = ursino_model.get_flow_dict(st, ica_stenosis_l=sten)
    q_mca_l.append(fl['q_mca_l'])
    q_mca_r.append(fl['q_mca_r'])

print(f"   - Left MCA Baseline Flow: {q_mca_l[0]:.2f} mL/s")
print(f"   - Left MCA Occluded Flow: {q_mca_l[len(sol_u1.t)+10]:.2f} mL/s")
print(f"   - Left MCA Post-Hypercapnia Flow: {q_mca_l[-1]:.2f} mL/s (Steal phenomenon)")

# %% [3] Run Štěpáník (2009) and Vaněk (2015) CSF Infusion Simulations (Native Boundaries)
print("\n2. Running Lumbar Infusion Test (1.5 mL/min for 300s) on Štěpáník vs. Vaněk (Native Boundaries)...")
duration_s = 500.0
time_grid = np.linspace(0, duration_s, int(duration_s * 100))

# Vaněk (2015) Native Driving Signals (Pb mean = 10.0 mmHg)
Pa_v = 80.0 + 10.0 * np.sin(2.0 * np.pi * 1.2 * time_grid)
Pb_v = 10.0 + 1.5 * np.sin(2.0 * np.pi * 1.2 * time_grid)
PC_v = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time_grid)
PT_v = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time_grid)
PL_v = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time_grid)

interp_vanek = {
    'Pa': PchipInterpolator(time_grid, Pa_v),
    'Pb': PchipInterpolator(time_grid, Pb_v),
    'PC': PchipInterpolator(time_grid, PC_v),
    'PT': PchipInterpolator(time_grid, PT_v),
    'PL': PchipInterpolator(time_grid, PL_v),
}

def inf_func(t):
    inf = np.zeros(5)
    if 60.0 <= t <= 360.0:
        inf[4] = 0.025
    return inf

vanek_model = Vanek2015Model()
sol_vanek = solve_ivp(
    fun=lambda t, P: vanek_model.derivatives(t, P, interp_vanek, infusion_rates=inf_func(t)),
    t_span=(0.0, duration_s),
    y0=np.array([11.0, 10.0, 10.0, 10.0, 10.0]),
    method='Radau', rtol=1e-5, atol=1e-5
)

# Štěpáník (2009) Native Driving Signals (pV mean = 2.0 mmHg)
Pa_s = 80.0 + 8.0 * np.sin(2.0 * np.pi * 1.0 * time_grid)
pV_s = 2.0 + 0.5 * np.sin(2.0 * np.pi * 0.25 * time_grid)
pC_s = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time_grid)
pT_s = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time_grid)
pL_s = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time_grid)

pAV_s = Pa_s - pV_s; pVC_s = pV_s - pC_s; pCT_s = pC_s - pT_s; pTL_s = pT_s - pL_s
interp_step = {
    'pA': PchipInterpolator(time_grid, Pa_s),
    'pV': PchipInterpolator(time_grid, pV_s),
    'pC': PchipInterpolator(time_grid, pC_s),
    'pT': PchipInterpolator(time_grid, pT_s),
    'pL': PchipInterpolator(time_grid, pL_s),
    'pAV': PchipInterpolator(time_grid, pAV_s),
    'pVC': PchipInterpolator(time_grid, pVC_s),
    'pCT': PchipInterpolator(time_grid, pCT_s),
    'pTL': PchipInterpolator(time_grid, pTL_s)
}

stepanik_model = Stepanik2009Model()
stepanik_model.infusion_rate[4] = 0.025
stepanik_model.infusion_start[4] = 60.0
stepanik_model.infusion_duration[4] = 300.0

p0_s = np.array([stepanik_model.p1A0, stepanik_model.p2V0, stepanik_model.p3C0, stepanik_model.p4T0, stepanik_model.p5L0])
sol_step = solve_ivp(
    fun=lambda t, p: stepanik_model.derivatives(t, p, interp_step),
    t_span=(0.0, duration_s), y0=p0_s, method='Radau', rtol=1e-4, atol=1e-4
)

p2_step = sol_step.y[1] + interp_step['pV'](sol_step.t)
p5_step = sol_step.y[4] + interp_step['pL'](sol_step.t)

mask_base_v = (sol_vanek.t >= 10) & (sol_vanek.t <= 50)
mask_plat_v = (sol_vanek.t >= 340) & (sol_vanek.t <= 360)
p0_v = np.mean(sol_vanek.y[1][mask_base_v])
pinf_v = np.mean(sol_vanek.y[1][mask_plat_v])
dp_v = pinf_v - p0_v
Rout_v = (dp_v / 0.025) / 60.0

mask_base_s = (sol_step.t >= 10) & (sol_step.t <= 50)
mask_plat_s = (sol_step.t >= 340) & (sol_step.t <= 360)
p0_s_val = np.mean(p2_step[mask_base_s])
pinf_s_val = np.mean(p2_step[mask_plat_s])
dp_s_val = pinf_s_val - p0_s_val
Rout_s = (dp_s_val / 0.025) / 60.0

# %% [4] Print Quantitative Comparison Summary
print("\n" + "="*75)
print(f"{'Metric / Parameter':<35} | {'Štěpáník (2009)':<16} | {'Vaněk (2015)':<16}")
print("="*75)
print(f"{'Baseline ICP (P2_0)':<35} | {p0_s_val:<16.2f} | {p0_v:<16.2f} mmHg")
print(f"{'5-Min Peak Infusion ICP (P2_inf)':<35} | {pinf_s_val:<16.2f} | {pinf_v:<16.2f} mmHg")
print(f"{'5-Min Peak ICP in mmH2O':<35} | {pinf_s_val*13.6:<16.1f} | {pinf_v*13.6:<16.1f} mmH2O")
print(f"{'ICP Elevation (ΔP)':<35} | {dp_s_val:<16.2f} | {dp_v:<16.2f} mmHg")
print(f"{'Calculated Outflow Resistance Rout':<35} | {Rout_s:<16.2f} | {Rout_v:<16.2f} mmHg/(mL/min)")
print("="*75)

# %% [5] Generate Visual Comparison Figures
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(t_u, q_mca_l, 'r-', label='Left MCA Flow (Ischemic)')
ax.plot(t_u, q_mca_r, 'b--', label='Right MCA Flow (Healthy)')
ax.axvline(50, color='k', linestyle=':', label='ICA Occlusion')
ax.axvline(100, color='g', linestyle=':', label='Hypercapnia (PaCO2=60)')
ax.set_title("Ursino (2010): Microvascular Steal Phenomenon")
ax.set_xlabel("Time [s]"); ax.set_ylabel("Blood Flow [mL/s]")
ax.grid(True); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(t_u, y_u[0], 'purple', label='Intracranial Pressure (Pic)')
ax.axvline(50, color='k', linestyle=':')
ax.axvline(100, color='g', linestyle=':')
ax.set_title("Ursino (2010): Intracranial Pressure (Pic)")
ax.set_xlabel("Time [s]"); ax.set_ylabel("Pressure [mmHg]")
ax.grid(True); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(sol_step.t, p2_step, 'r-', label='Štěpáník (2009) P2 (ICP, pV=2.0 mmHg)', linewidth=1.5)
ax.plot(sol_vanek.t, sol_vanek.y[1], 'g--', label='Vaněk (2015) P2 (ICP, Pb=10.0 mmHg)', linewidth=1.5)
ax.axvspan(60, 360, color='orange', alpha=0.15, label='Infusion Phase (1.5 mL/min)')
ax.set_title("Štěpáník vs. Vaněk: Native Boundary ICP Elevation")
ax.set_xlabel("Time [s]"); ax.set_ylabel("ICP P2 [mmHg]")
ax.grid(True); ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(sol_step.t, p5_step, 'm-', label='Štěpáník (2009) P5 (Lumbar)')
ax.plot(sol_vanek.t, sol_vanek.y[4], 'c--', label='Vaněk (2015) P5 (Lumbar)')
ax.axvspan(60, 360, color='orange', alpha=0.15)
ax.set_title("Štěpáník vs. Vaněk: Lumbar SAS Pressure (P5)")
ax.set_xlabel("Time [s]"); ax.set_ylabel("Lumbar Pressure P5 [mmHg]")
ax.grid(True); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("./model_comparison_plots.png", dpi=150)
print("\nPlot saved successfully to ./model_comparison_plots.png!")
