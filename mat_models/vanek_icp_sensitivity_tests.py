import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from vanek_2015_model import Vanek2015Model


def run_custom_infusion(
    c2b0_val=0.60, alpha_divider=60.0, r23_val=0.30, duration_s=400.0
):
  # Instantiate model and override parameters
  model = Vanek2015Model()
  model.C2b0 = c2b0_val  # Venous compliance
  model.alpha = 0.11 / alpha_divider  # Outflow conductance
  model.R23 = r23_val  # Cervical SAS resistance
  model.Z23 = 1.0 / model.R23

  time = np.linspace(0, duration_s, int(duration_s * 100))

  # Synthetic boundary pressures
  Pa = 80.0 + 10.0 * np.sin(2.0 * np.pi * 1.2 * time)
  Pb = 10.0 + 1.5 * np.sin(2.0 * np.pi * 1.2 * time)
  PC = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time)
  PT = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time)
  PL = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time)

  interpolators = {
      "Pa": PchipInterpolator(time, Pa),
      "Pb": PchipInterpolator(time, Pb),
      "PC": PchipInterpolator(time, PC),
      "PT": PchipInterpolator(time, PT),
      "PL": PchipInterpolator(time, PL),
  }

  # Infusion rate: 1.5 mL/min (0.025 mL/s) between t=50s and t=300s
  def infusion_func(t):
    inf = np.zeros(5)
    if 50.0 <= t <= 300.0:
      inf[4] = 0.025
    return inf

  P0 = np.array([11.0, 10.0, 10.0, 10.0, 10.0])

  sol = solve_ivp(
      fun=lambda t, P: model.derivatives(
          t, P, interpolators, infusion_rates=infusion_func(t)
      ),
      t_span=(0.0, duration_s),
      y0=P0,
      method="Radau",
      rtol=1e-5,
      atol=1e-5,
  )
  return sol.t, sol.y[1]  # Return time and ICP (P2)


# --- Run 3 Scenarios ---
print("Running Vaněk parameter sensitivity experiments...")
t_norm, p2_norm = run_custom_infusion()  # Default baseline
t_hydro, p2_hydro = run_custom_infusion(
    alpha_divider=180.0
)  # High outflow resistance (Hydrocephalus)
t_stiff, p2_stiff = run_custom_infusion(
    c2b0_val=0.05
)  # Collapsed venous cushion (Stiff veins)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(t_norm, p2_norm, "g-", label="Normal Baseline")
plt.plot(
    t_hydro,
    p2_hydro,
    "r--",
    label="Hydrocephalus (High Outflow Resistance Rout)",
)
plt.plot(
    t_stiff,
    p2_stiff,
    "m-.",
    label="Stiff Venous Bed (C2b0 = 0.05 vs 0.60 mL/mmHg)",
)

plt.axvspan(50, 300, color="orange", alpha=0.15, label="Infusion Phase")
plt.title("Vaněk Model: Sensitivity of ICP Plateau to Compliance & Resistance")
plt.xlabel("Time (s)")
plt.ylabel("Intracranial Pressure P2 (mmHg)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("vanek_icp_plots.png", dpi=150)
print("Plot saved as vanek_icp_plots.png!")