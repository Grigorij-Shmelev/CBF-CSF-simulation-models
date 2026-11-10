"""
urs_cd_sigmoid.py -- Ursino's pial compliance sigmoid C_d(x)/C_dn for the three district types.

Uses the same formula as Ursino2010Model.derivatives():
    C_d = C_dn * ((1 - sat) + (1 + sat) * u) / (1 + u),  u = exp(x / k_C),  k_C = C_dn * sat / 2,
    sat = sat2 (x > 0) or sat1 (x < 0).
Prints the x at which C_d = 0 and saves urs_cd_sigmoid.png. Run from mat_models/.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ursino_2010_model import Ursino2010Model

m = Ursino2010Model()


def cd_ratio(cdn, x):
    s = np.where(x < 0, m.sat1, m.sat2)
    u = np.exp(x / (cdn * s / 2.0))
    return ((1 - s) + (1 + s) * u) / (1 + u)


fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for d, lab in (("ml", "MCA"), ("pl", "PCA"), ("al", "ACA")):
    cdn = m.Cdn[d]
    x1 = np.linspace(-1.0, 0.1, 2201)
    x2 = np.linspace(-0.05, 0.05, 1001)
    ax[0].plot(x1, cd_ratio(cdn, x1), label=f"{lab} district ($C_{{dn}}$ = {cdn:.3f} mL/mmHg)")
    ax[1].plot(x2, cd_ratio(cdn, x2), label=f"{lab} district")
    print(f"{lab}: C_d = 0 at x = {cdn * m.sat1 / 2 * np.log(0.75):+.4f}")
for a in ax:
    a.axhline(0, color="k", lw=0.8, ls=":")
    a.axhline(1 + m.sat2, color="gray", lw=0.8, ls="--")
    a.set_xlabel("x = x_CO2 − x_aut"); a.set_ylabel("C_d / C_dn"); a.grid(alpha=0.3)
ax[1].axvspan(-0.0021, 0.05, color="tab:green", alpha=0.08, label="range of x in our runs")
ax[0].set_title("Full range: ceiling 1 + sat2 = 1.4, floor 1 − sat1 = −6")
ax[1].set_title("Zoom around x = 0 (below the dotted line: C_d < 0)")
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("urs_cd_sigmoid.png", dpi=150)
print("Saved urs_cd_sigmoid.png")
