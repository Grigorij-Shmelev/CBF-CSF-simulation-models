"""
vanek_pb_step_test.py -- does a lasting change of Vanek's venous input Pb change the mean P2 (ICP)?

Test: Pb is held at 14 mmHg (control), or ramped from 14 to 20 mmHg between t = 50 and 70 s and then held.
Pa is constant (97.3 mmHg), all other inputs as in run_vanek_simulation. vanek_2015_model.py is NOT modified.
Prints the mean of P2 over 20 s windows (whole 0.25 Hz respiratory cycles) and saves vanek_pb_step_test.png.

Run from mat_models/:   python vanek_pb_step_test.py        (about 1-2 min)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from vanek_2015_model import Vanek2015Model

T_END, DT_IN, DT_OUT = 4000.0, 0.05, 0.5
t = np.arange(0.0, T_END + DT_IN, DT_IN)
WINDOWS = (40, 100, 200, 400, 800, 1500, 2500, 3900)        # window starts [s], length 20 s


def synthetic(t):
    """Boundary signals as in vanek_2015_model.run_vanek_simulation (Pa replaced below)."""
    return {"Pa": np.full_like(t, 97.3),
            "Pb": 10.0 + 1.5 * np.sin(2 * np.pi * 1.2 * t),
            "PC": 2.0 + 0.3 * np.sin(2 * np.pi * 0.25 * t),
            "PT": 2.0 + 1.2 * np.sin(2 * np.pi * 0.25 * t),
            "PL": 2.0 + 0.4 * np.sin(2 * np.pi * 0.25 * t)}


def run(model, pb):
    sig = synthetic(t)
    sig["Pb"] = pb
    interp = {k: PchipInterpolator(t, v) for k, v in sig.items()}
    sol = solve_ivp(lambda tt, P: model.derivatives(tt, P, interp), (0.0, T_END),
                    np.array([11.0, 10.0, 10.0, 10.0, 10.0]), method="Radau",
                    t_eval=np.arange(0.0, T_END, DT_OUT), rtol=1e-5, atol=1e-5)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y[1]


def main():
    model = Vanek2015Model()
    cases = {"Pb = 14 (control)": np.full_like(t, 14.0),
             "Pb 14 -> 20 (ramp 50-70 s, then held)": np.where(t < 50, 14.0, np.where(t < 70, 14.0 + 6.0 * (t - 50) / 20, 20.0))}
    print("mean P2 [mmHg] over 20 s windows starting at t =", WINDOWS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, pb in cases.items():
        ts, p2 = run(model, pb)
        means = [p2[(ts >= a) & (ts < a + 20)].mean() for a in WINDOWS]
        print(f"{label:42s}", "  ".join(f"{m:6.3f}" for m in means))
        k = int(20 / DT_OUT)                                  # 20 s moving mean for the plot
        ax.plot(ts[: len(ts) - k + 1], np.convolve(p2, np.ones(k) / k, mode="valid"), label=label)
    print(f"equilibrium from the constants: P2 = Pmin + S1/alpha = {model.Pmin + model.S1 / model.alpha:.3f}")
    ax.set_xlabel("t [s]"); ax.set_ylabel("P2, 20 s moving mean [mmHg]"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Vanek: effect of a lasting step in Pb on the mean ICP (P2)")
    fig.tight_layout(); fig.savefig("vanek_pb_step_test.png", dpi=150)
    print("Saved vanek_pb_step_test.png")


if __name__ == "__main__":
    main()
