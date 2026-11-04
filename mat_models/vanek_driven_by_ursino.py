"""
vanek_driven_by_ursino.py  --  Task D3.3 (A5): first file-based, one-way coupling

Ursino -> CSV -> Vanek. The CSV comes from:
    python urs_dynamic_pa_test.py export     (-> ursino_to_vanek_D100.csv / _D80.csv)

vanek_2015_model.py is NOT modified: only the 'Pa' (and optionally 'Pb')
interpolators passed to Vanek2015Model.derivatives() are replaced.

Usage (from mat_models/):
    python vanek_driven_by_ursino.py            -> uses ursino_to_vanek_D100.csv
    python vanek_driven_by_ursino.py D80        -> uses ursino_to_vanek_D80.csv
"""
import sys
import time as clock
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from vanek_2015_model import Vanek2015Model

SCEN = sys.argv[1] if len(sys.argv) > 1 else "D100"
CSV = f"ursino_to_vanek_{SCEN}.csv"
T_HYPER = 50.0          # must match the export (hypercapnia onset in the Ursino run)
T_END = 399.0           # CSV ends at ~400 s
DT_OUT = 0.005
WIN = 6 / 1.2           # 6 cardiac cycles = 5 s
WINDOWS = {"pre": (T_HYPER - WIN, T_HYPER), "late": (T_END - WIN, T_END)}

# Each run: which signal drives Pa and Pb.
#   pa: None -> Vanek's own synthetic Pa (reference), or a CSV column name
#   pb: "synthetic" (Vanek default, 1.2 Hz) | "const" (10 mmHg, diagnostic) | a CSV column
RUNS = {
    "ref: synthetic Pa (80)":      {"pa": None,         "pb": "synthetic"},
    "Pa = PICA_l":                 {"pa": "PICA_l",     "pb": "synthetic"},
    "Pa = PMCA_end_l":             {"pa": "PMCA_end_l", "pb": "synthetic"},
    "Pa = Pd_ml":                  {"pa": "Pd_ml",      "pb": "synthetic"},
    "diag: Pa = PICA_l, Pb const": {"pa": "PICA_l",     "pb": "const"},
    "Pa = PICA_l, Pb = Pv (Ursino)": {"pa": "PICA_l",   "pb": "Pv"},
}


def synthetic(t):
    """Boundary signals exactly as in vanek_2015_model.run_vanek_simulation."""
    return {"Pa": 80.0 + 10.0 * np.sin(2 * np.pi * 1.2 * t),
            "Pb": 10.0 + 1.5 * np.sin(2 * np.pi * 1.2 * t),
            "PC": 2.0 + 0.3 * np.sin(2 * np.pi * 0.25 * t),
            "PT": 2.0 + 1.2 * np.sin(2 * np.pi * 0.25 * t),
            "PL": 2.0 + 0.4 * np.sin(2 * np.pi * 0.25 * t)}


def make_interpolators(df, run):
    t = df["t"].to_numpy()
    sig = synthetic(t)
    if run["pa"] is not None:
        sig["Pa"] = df[run["pa"]].to_numpy()
    if run["pb"] == "const":
        sig["Pb"] = np.full_like(t, 10.0)
    elif run["pb"] != "synthetic":
        sig["Pb"] = df[run["pb"]].to_numpy()
    return {k: PchipInterpolator(t, v) for k, v in sig.items()}


def run_vanek(model, interp):
    t_eval = np.arange(0.0, T_END, DT_OUT)
    sol = solve_ivp(lambda t, P: model.derivatives(t, P, interp),
                    (0.0, T_END), np.array([11.0, 10.0, 10.0, 10.0, 10.0]),   # P0 as in the repo
                    method="Radau", t_eval=t_eval, rtol=1e-5, atol=1e-5)
    if not sol.success:
        raise RuntimeError(sol.message)
    P = sol.y
    return sol.t, {"P1": P[0], "P2": P[1], "F23": (P[1] - P[2]) * model.Z23,
                   "Pa_in": interp["Pa"](sol.t), "Pb_in": interp["Pb"](sol.t)}


def main():
    try:
        df = pd.read_csv(CSV)
    except FileNotFoundError:
        sys.exit(f"{CSV} not found -> run 'python urs_dynamic_pa_test.py export' first")
    model = Vanek2015Model()
    results, rows = {}, []
    for name, run in RUNS.items():
        tic = clock.time()
        t, out = run_vanek(model, make_interpolators(df, run))
        results[name] = (t, out)
        r = {"run": name}
        for w, (a, b) in WINDOWS.items():
            m = (t >= a) & (t < b)
            for k in ("Pa_in", "P2", "F23"):
                r[f"{w}_{k}_mean"], r[f"{w}_{k}_ptp"] = out[k][m].mean(), np.ptp(out[k][m])
        rows.append(r)
        print(f"{name}: {clock.time() - tic:.0f} s")

    tab = pd.DataFrame(rows)
    tab.to_csv(f"vanek_driven_by_ursino_{SCEN}.csv", index=False, float_format="%.4f")
    show = ["run", "pre_Pa_in_mean", "late_Pa_in_mean", "pre_P2_mean", "late_P2_mean",
            "pre_P2_ptp", "late_P2_ptp", "pre_F23_ptp", "late_F23_ptp"]
    with pd.option_context("display.float_format", "{:7.3f}".format, "display.width", 160):
        print(f"\nInput CSV: {CSV}   (P in mmHg, F23 in mL/s; ptp = peak-to-peak)")
        print(tab[show].to_string(index=False))

    # Plot: Pa input, P2 (ICP) and F23; full run and 3 s zoom in the late window
    keys = [("Pa_in", "Vanek input Pa [mmHg]"), ("P2", "P2 = intracranial SAS / ICP [mmHg]"),
            ("F23", "F23 = CSF flow cranial -> cervical [mL/s]")]
    fig, ax = plt.subplots(len(keys), 2, figsize=(14, 10), sharex="col")
    for name, (t, out) in results.items():
        z = (t >= T_END - 3.0)
        for r, (k, lab) in enumerate(keys):
            ax[r, 0].plot(t, out[k], lw=0.5, label=name)
            ax[r, 1].plot(t[z], out[k][z], lw=1.0, label=name)
            ax[r, 0].set_ylabel(lab, fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    for a in ax[:, 0]:
        a.axvline(T_HYPER, color="k", ls=":", lw=0.8)
    ax[0, 1].legend(fontsize=7)
    ax[0, 0].set_title("Full run (hypercapnia in Ursino at dotted line)")
    ax[0, 1].set_title("Zoom: last 3 s")
    ax[-1, 0].set_xlabel("t [s]"); ax[-1, 1].set_xlabel("t [s]")
    fig.suptitle(f"Vanek (2015) driven by Ursino output ({CSV}), one-way, file-based")
    fig.tight_layout()
    fig.savefig(f"vanek_driven_by_ursino_{SCEN}.png", dpi=150)
    print(f"\nSaved vanek_driven_by_ursino_{SCEN}.csv and .png")


if __name__ == "__main__":
    main()
