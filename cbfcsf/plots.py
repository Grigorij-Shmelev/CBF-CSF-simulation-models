"""Figures of the checks. pyplot is imported inside the functions (as in the original new_model.py): importing it
resets Python's record of warnings already shown, so importing it earlier would change what the runs print on stderr."""
from cbfcsf.fit_lit import LIT_TRUTH, lit_curve, lit_protocol
from cbfcsf.models.new_model import NAMES


def plot_lit_figure(model, t, p1, p2, p5, formation, resorption, infusion_i5, flows, volumes, compliances, title,
                    plot_path):
    """6-panel figure of a lumbar infusion test in a 5-compartment CSF model (Vanek or Stepanik).
    flows = (F12, F23, F34, F45), volumes = (V1..V5), compliances = [(array, label), ...] (5 curves);
    formation = S1 on the time grid [mL/s]."""
    import matplotlib.pyplot as plt
    F12, F23, F34, F45 = flows
    V1, V2, V3, V4, V5 = volumes
    # Generate plot figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Panel 1: Craniospinal Pressures
    ax = axes[0, 0]
    ax.plot(t, p1, label='P1 (Ventricles)', alpha=0.8)
    ax.plot(t, p2, label='P2 (Intracranial SAS / ICP)', linewidth=1.5, color='crimson')
    ax.plot(t, p5, label='P5 (Lumbar SAS)', linewidth=1.5, color='darkblue', linestyle='--')
    ax.axvspan(60, 360, color='orange', alpha=0.15, label='Infusion Phase (1.5 mL/min)')
    ax.set_title("Craniospinal Pressures During Infusion [mmHg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 2: Fluid Mass Balance
    ax = axes[0, 1]
    ax.plot(t, formation * 60, label='Formation (S1)', color='green')
    ax.plot(t, resorption * 60, label='Resorption (F_res)', color='red')
    ax.plot(t, infusion_i5 * 60, label='Lumbar Infusion (I5)', color='orange', linewidth=2)
    ax.set_title("Fluid Mass Balance [mL/min]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rate [mL/min]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='center right', fontsize=8)

    # Panel 3: Inter-compartmental Flows
    ax = axes[1, 0]
    ax.plot(t, F12 * 60, label='F12 (Ventr -> Cranial SAS)', alpha=0.7)
    ax.plot(t, F23 * 60, label='F23 (Cranial -> Cervical SAS)', alpha=0.7)
    ax.plot(t, F34 * 60, label='F34 (Cervical -> Thoracic SAS)', alpha=0.7)
    ax.plot(t, F45 * 60, label='F45 (Thoracic -> Lumbar SAS)', alpha=0.7)
    ax.set_title("Inter-compartmental CSF Flows [mL/min]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flow [mL/min]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 4: Compartmental Volumes
    ax = axes[1, 1]
    ax.plot(t, V1, label='V1 (Ventricles)')
    ax.plot(t, V2, label='V2 (Intracranial SAS)')
    ax.plot(t, V3, label='V3 (Cervical)')
    ax.plot(t, V4, label='V4 (Thoracic)')
    ax.plot(t, V5, label='V5 (Lumbar)')
    ax.set_title("Compartmental Volumes [mL]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Volume [mL]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 5: Non-linear Compliances
    ax = axes[2, 0]
    for c, label in compliances:
        ax.plot(t, c, label=label, alpha=0.7)
    ax.set_title("Non-linear Compliances [mL/mmHg]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Compliance [mL/mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

    # Panel 6: Dynamic Pressure-Volume Curve
    ax = axes[2, 1]
    dV_total = (V1 + V2 + V3 + V4 + V5) - (model.V10 + model.V20 + model.V30 + model.V40 + model.V50)
    ax.plot(dV_total, p2, color='purple', label='Global P-V Curve')
    ax.set_title("Dynamic Pressure-Volume (P-V) Curve")
    ax.set_xlabel("Total CSF Volume Change ΔV [mL]")
    ax.set_ylabel("ICP P2 [mmHg]")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot successfully saved to {plot_path}")


def plot_run(r, m, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = r["t"] / 60.0
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    ax[0, 0].plot(t, r["CBF"], "k", label="total")
    for j, n in enumerate(NAMES):
        ax[0, 0].plot(t, r["q_k"][:, j], label=n, ls="-" if j < 2 else "--")
    ax[0, 0].set_title("Flow [mL/s]"); ax[0, 0].legend(fontsize=8, ncol=2)
    for j, n in enumerate(NAMES):
        ax[0, 1].plot(t, r["Va"][:, j], label=n, ls="-" if j < 2 else "--")
    ax[0, 1].plot(t, r["Vv"], color="tab:purple", label="venous")
    ax[0, 1].set_title("Stressed blood volume per compartment [mL]"); ax[0, 1].legend(fontsize=8, ncol=2)
    ax[0, 2].plot(t, r["ICP"], label="ICP = P2"); ax[0, 2].plot(t, r["P5"], "--", label="P5 lumbar")
    ax[0, 2].axhline(m.Pvs_n, color="gray", lw=0.8, ls=":"); ax[0, 2].set_title("CSF pressure [mmHg]")
    ax[0, 2].legend(fontsize=8)
    ax[1, 0].plot(t, r["Vblood"] - r["Vblood"][0], label="blood (stressed CBV)")
    ax[1, 0].plot(t, r["V1"] + r["V2"] - (r["V1"][0] + r["V2"][0]), label="cranial CSF")
    ax[1, 0].plot(t, r["dVspinal"], label="spinal CSF")
    ax[1, 0].set_title("Volume change [mL]"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, r["x_CO2"][:, 0] - r["x_aut"][:, 0], label="mL"); ax[1, 1].plot(t, r["x_CO2"][:, 1] - r["x_aut"][:, 1], label="rL")
    ax[1, 1].plot(t, r["x_CO2"][:, 2] - r["x_aut"][:, 2], "--", label="mR")
    ax[1, 1].set_title("Control x = x_CO2 - x_aut (> 0: dilation)"); ax[1, 1].legend(fontsize=8)
    ax[1, 2].plot(t, r["Pk"][:, 0], label="P arterial mL"); ax[1, 2].plot(t, r["Pk"][:, 1], label="P arterial rL")
    ax[1, 2].plot(t, r["Pc"], label="P capillary"); ax[1, 2].plot(t, r["Pv"], label="P venous")
    ax[1, 2].set_title("Blood pressures [mmHg]"); ax[1, 2].legend(fontsize=8)
    for a in ax.ravel():
        a.axvspan(r["span"][0] / 60.0, r["span"][1] / 60.0, color="tab:orange", alpha=0.10); a.grid(alpha=0.3)
    for a in ax[1]:
        a.set_xlabel("time [min]")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_fit(f, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(f["t"], f["y"], ".", ms=3, color="0.6", label="synthetic data (truth + 0.3 mmHg noise)")
    ax.plot(f["t"], lit_curve(LIT_TRUTH, f["t"], lit_protocol()), "k", lw=1.2, label="truth")
    ax.plot(f["t"], f["fit"], "r--", lw=1.4, label="fit")
    R, sc, P0 = f["theta"]
    ax.set_title(f"LIT fit on synthetic data: R_out {R:.1f} (truth {LIT_TRUTH[0]}) mmHg min/mL, "
                 f"scale {sc:.2f} ({LIT_TRUTH[1]}), P0 {P0:.2f} ({LIT_TRUTH[2]}) mmHg", fontsize=9)
    ax.set_xlabel("time [s]"); ax.set_ylabel("lumbar pressure P5 [mmHg]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_recording(vc, vb, path):
    """K2: ICP pulse of the model (default and Vanek values) driven by the data file, vs the ICP of the data file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    w = (vc["tr"] >= 20) & (vc["tr"] <= 30)
    ax.plot(vc["tr"][w], vc["icp_rec"][w] - vc["icp_rec_mean"], "k", label="Stepanik data file (one beat repeated)")
    for v, lab in ((vc, "model, default (storage calibrated)"), (vb, "model, Vanek values")):
        w2 = (v["t"] >= 20) & (v["t"] <= 30)
        ax.plot(v["t"][w2], v["icp_mod"][w2] - v["icp_mod_mean"], label=lab)
    ax.set_xlabel("time [s]"); ax.set_ylabel("ICP - mean [mmHg]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("ICP pulse: model driven by the arterial pressure of the data file vs its ICP")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
