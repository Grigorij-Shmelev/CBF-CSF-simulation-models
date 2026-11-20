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
from cbfcsf.checks.vanek import lit_run
from cbfcsf.models.vanek_2015 import Vanek2015Model, synthetic_signals
from cbfcsf.paths import out_path

VALUES = (0.6, 0.1, 0.02, 0.005, 0.001)


def run(c2b0, T, pa, pb, resp, infusion):
    m = Vanek2015Model(); m.C2b0 = c2b0

    def signals(t):
        sig = synthetic_signals(t, resp=resp)
        sig["Pa"], sig["Pb"] = pa(t), pb(t)
        return sig

    window = (50.0, 300.0) if infusion else (None, None)
    r = lit_run(m, T, *window, signals=signals, t_eval=np.arange(0.0, T, 0.05))[0]
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
fig.tight_layout(); fig.savefig(out_path("vanek", "vanek_c2b0_sweep.png"), dpi=150)
print("Saved vanek_c2b0_sweep.png")
