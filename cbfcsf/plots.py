"""Figures shared by several checks."""
import matplotlib.pyplot as plt


def plot_lit_figure(model, t, p1, p2, p5, formation, resorption, infusion_i5, flows, volumes, compliances, title,
                    plot_path):
    """6-panel figure of a lumbar infusion test in a 5-compartment CSF model (Vanek or Stepanik).
    flows = (F12, F23, F34, F45), volumes = (V1..V5), compliances = [(array, label), ...] (5 curves);
    formation = S1 on the time grid [mL/s]."""
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
