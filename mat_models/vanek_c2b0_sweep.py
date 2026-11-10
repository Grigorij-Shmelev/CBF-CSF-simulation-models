"""
vanek_c2b0_sweep.py -- the two roles of Vanek's venous-wall compliance C2b0.

For each C2b0 value:
  (1) lumbar infusion test as in vanek_icp_sensitivity_tests.py (1.5 mL/min into compartment 5 between
      50 and 300 s): rise of mean P2 = mean(280-300 s) - mean(30-50 s);
  (2) cardiac pulse path: P2 peak-to-peak (40-60 s) when only Pa pulses (80 +- 10, Pb = 10) and when only
      Pb pulses (10 +- 1.5, Pa = 80); respiratory inputs constant.
vanek_2015_model.py is NOT modified (C2b0 is set on the instance). Saves vanek_c2b0_sweep.png.
Run from mat_models/  (about 1-2 min).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from vanek_2015_model import Vanek2015Model

VALUES = (0.6, 0.1, 0.02, 0.005, 0.001)


def run(c2b0, T, pa, pb, resp, infusion):
    m = Vanek2015Model(); m.C2b0 = c2b0
    t = np.linspace(0.0, T, int(T * 100))
    s = lambda f, A: A * np.sin(2 * np.pi * f * t)
    sig = {"Pa": pa(t), "Pb": pb(t), "PC": 2 + resp * s(0.25, 0.3),
           "PT": 2 + resp * s(0.25, 1.2), "PL": 2 + resp * s(0.25, 0.4)}
    it = {k: PchipInterpolator(t, v) for k, v in sig.items()}
    inf = (lambda tt: np.array([0, 0, 0, 0, 0.025]) if 50 <= tt <= 300 else np.zeros(5)) if infusion \
        else (lambda tt: np.zeros(5))
    r = solve_ivp(lambda tt, P: m.derivatives(tt, P, it, infusion_rates=inf(tt)), (0.0, T),
                  np.array([11.0, 10.0, 10.0, 10.0, 10.0]), method="Radau",
                  t_eval=np.arange(0.0, T, 0.05), rtol=1e-5, atol=1e-5)
    return r.t, r.y[1]


def pulse(t, mean, A):
    return mean + A * np.sin(2 * np.pi * 1.2 * t)


rows = []
print("C2b0    infusion rise   P2 pulse (Pa only)   P2 pulse (Pb only)")
for c in VALUES:
    t, p = run(c, 400.0, lambda t: pulse(t, 80, 10), lambda t: pulse(t, 10, 1.5), 1.0, True)
    rise = p[(t >= 280) & (t < 300)].mean() - p[(t >= 30) & (t < 50)].mean()
    t, p = run(c, 60.0, lambda t: pulse(t, 80, 10), lambda t: np.full_like(t, 10.0), 0.0, False)
    via_a = np.ptp(p[(t >= 40) & (t < 60)])
    t, p = run(c, 60.0, lambda t: np.full_like(t, 80.0), lambda t: pulse(t, 10, 1.5), 0.0, False)
    via_b = np.ptp(p[(t >= 40) & (t < 60)])
    rows.append((c, rise, via_a, via_b))
    print(f"{c:<7g} {rise:8.2f} mmHg   {via_a:10.3f} mmHg        {via_b:10.3f} mmHg", flush=True)

c, rise, va, vb = map(np.array, zip(*rows))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogx(c, rise, "o-"); ax[0].set_title("Lumbar infusion: rise of mean P2 [mmHg]")
ax[1].loglog(c, va, "o-", label="pulse from Pa only (±10 mmHg)")
ax[1].loglog(c, vb, "s-", label="pulse from Pb only (±1.5 mmHg)")
ax[1].set_title("Cardiac pulse in P2, peak-to-peak [mmHg]"); ax[1].legend(fontsize=8)
for a in ax:
    a.set_xlabel("C2b0 [mL/mmHg]"); a.grid(alpha=0.3, which="both"); a.axvline(0.6, color="k", ls=":", lw=0.8)
fig.tight_layout(); fig.savefig("vanek_c2b0_sweep.png", dpi=150)
print("Saved vanek_c2b0_sweep.png")
