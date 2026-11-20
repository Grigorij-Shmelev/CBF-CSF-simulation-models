"""Checks of the Ursino & Giannessi 2010 model (python run.py ursino ...).

steal(): ICA occlusion and hypercapnia, left vs right MCA flow (the old urs_steal_test.py).
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from cbfcsf.models.ursino_2010 import Ursino2010Model
from cbfcsf.paths import out_path


def steal_run(model):
    """Baseline 0-50 s, left ICA occluded (resistance x 500) 50-100 s, then PaCO2 = 60 mmHg 100-250 s.
    Returns the three solutions joined, the left and right MCA flows at every solver point, and the end states."""
    y0 = model.get_initial_state()

    # Phase 1: Baseline (0 to 50s)
    sol_base = solve_ivp(
        lambda t, y: model.derivatives(t, y), [0, 50], y0, method="Radau"
    )
    y_ss = sol_base.y[:, -1]

    # Phase 2: 100% Left ICA Occlusion (50s to 100s)
    sol_occl = solve_ivp(
        lambda t, y: model.derivatives(t, y, ica_stenosis_l=500.0),
        [50, 100],
        y_ss,
        method="Radau",
    )

    # Phase 3: Hypercapnia Challenge (PaCO2 = 60 mmHg) (100s to 250s)
    sol_hyper = solve_ivp(
        lambda t, y: model.derivatives(t, y, PaCO2=60.0, ica_stenosis_l=500.0),
        [100, 250],
        sol_occl.y[:, -1],
        method="Radau",
    )

    # Combine time and state histories
    t_all = np.concatenate([sol_base.t, sol_occl.t, sol_hyper.t])
    y_all = np.hstack([sol_base.y, sol_occl.y, sol_hyper.y])

    # Calculate Left MCA (Ischemic) and Right MCA (Healthy) Flows over time
    q_mca_left = []
    q_mca_right = []
    for i in range(len(t_all)):
        state = y_all[:, i]
        # Adjust stenosis multiplier depending on time phase
        stenosis = 500.0 if t_all[i] >= 50 else 1.0
        flows = model.get_flow_dict(state, ica_stenosis_l=stenosis)
        q_mca_left.append(flows["q_mca_l"])
        q_mca_right.append(flows["q_mca_r"])
    return dict(t=t_all, y=y_all, q_mca_l=q_mca_left, q_mca_r=q_mca_right,
                y_ss=y_ss, y_occ=sol_occl.y[:, -1], y_hyp=sol_hyper.y[:, -1])


def steal():
    """Steal phenomenon: saves outputs/ursino/urs_steal_phenomenon.png."""
    model = Ursino2010Model()
    print("Simulating Steal Phenomenon & Hypercapnia Transients...")
    r = steal_run(model)

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(r["t"], r["q_mca_l"], "r-", label="Left MCA Flow (Ischemic side)")
    plt.plot(r["t"], r["q_mca_r"], "b--", label="Right MCA Flow (Healthy side)")
    plt.axvline(
        50, color="k", linestyle=":", label="Left ICA Occlusion (100% Stenosis)"
    )
    plt.axvline(100, color="g", linestyle=":", label="Hypercapnia (PaCO2 = 60 mmHg)")

    plt.title("Hemodynamic Response: ICA Occlusion & Steal Phenomenon")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Blood Flow (mL/s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("ursino", "urs_steal_phenomenon.png"), dpi=300, bbox_inches="tight")
    print("Plot saved as steal_phenomenon.png!")


if __name__ == "__main__":
    steal()
