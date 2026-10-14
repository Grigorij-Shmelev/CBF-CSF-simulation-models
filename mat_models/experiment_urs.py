import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from ursino_2010_model import Ursino2010Model

# 1. Setup Normal Model
model_normal = Ursino2010Model()

# 2. Setup High-Elastance / Low-Compliance Model
model_stiff = Ursino2010Model()
model_stiff.kE = 0.15  # Double elastance coefficient (kE = 0.15 vs 0.077)
model_stiff.Ro = 1000.0  # Higher CSF outflow resistance (Ro = 1000 vs 526)

# Initial state
y0 = model_normal.get_initial_state()

print("Simulating baseline and ICA occlusion under two skull compliance states...")

# Run 100% Left ICA Occlusion for both
sol_normal = solve_ivp(
    lambda t, y: model_normal.derivatives(t, y, ica_stenosis_l=500.0),
    [0, 60],
    y0,
    method="Radau",
)
sol_stiff = solve_ivp(
    lambda t, y: model_stiff.derivatives(t, y, ica_stenosis_l=500.0),
    [0, 60],
    y0,
    method="Radau",
)

# Plot Intracranial Pressure (Pic) comparison
plt.figure(figsize=(9, 5))
plt.plot(sol_normal.t, sol_normal.y[0], "b-", label="Normal Skull (kE = 0.077)")
plt.plot(
    sol_stiff.t,
    sol_stiff.y[0],
    "r--",
    label="Stiff / Low-Compliance Skull (kE = 0.15)",
)

plt.title("Ursino Model: ICP Response to Carotid Occlusion under Different Compliances")
plt.xlabel("Time (s)")
plt.ylabel("Intracranial Pressure Pic (mmHg)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ursino_compliance_experiment.png", dpi=150)
print("Plot saved as ursino_compliance_experiment.png!")
