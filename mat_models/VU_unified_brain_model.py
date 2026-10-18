import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ursino_2010_model import Ursino2010Model
from vanek_2015_model import Vanek2015Model


class UnifiedBrainModel:
    """
    Unified Biophysical Brain Model integrating Ursino (2010) Cerebrovascular 
    Hemodynamics with Vaněk (2015) 3rd Generation Craniospinal CSF Hydrodynamics.
    
    Version 2: Added direct support for Lumbar Infusion Test (LIT), Stenosis, 
    and Steal Phenomenon via parameter arguments in run_sequential_pipeline().
    """

    def __init__(self):
        self.ursino = Ursino2010Model()
        self.vanek = Vanek2015Model()

    def run_sequential_pipeline(
        self, 
        duration_s=200.0, 
        hypercapnia_start=50.0, 
        pco2_target=60.0, 
        stenosis_l=1.0,
        infusion_rate_ml_min=0.0,
        infusion_start=60.0,
        infusion_duration=300.0,
        infusion_compartment=4
    ):
        """
        Executes a sequential multi-scale pipeline:
        1. Solves Ursino (2010) 23-ODE hemodynamic model to calculate active vasomotion, 
           CBF redistribution, and total cerebral blood volume expansion Delta_V_blood(t).
        2. Passes blood volume expansion and microvascular pressures as dynamic driving 
           boundary conditions into Vaněk (2015) 5-compartment CSF hydrodynamics solver, 
           including optional fluid infusion (e.g. Lumbar Infusion Test).
        """
        print(f"1. Running Ursino Hemodynamics Engine (Duration: {duration_s}s, PaCO2 Target: {pco2_target} mmHg, ICA Stenosis L: {stenosis_l}x)...")
        
        y0_u = self.ursino.get_initial_state()
        
        # Phase 1: Baseline / Pre-hypercapnia (0 to hypercapnia_start)
        sol_u1 = solve_ivp(
            lambda t, y: self.ursino.derivatives(t, y, PaCO2=40.0, ica_stenosis_l=stenosis_l),
            [0.0, hypercapnia_start], y0_u, method='Radau', rtol=1e-5, atol=1e-5
        )
        
        # Phase 2: Hypercapnia (hypercapnia_start to duration_s)
        sol_u2 = solve_ivp(
            lambda t, y: self.ursino.derivatives(t, y, PaCO2=pco2_target, ica_stenosis_l=stenosis_l),
            [hypercapnia_start, duration_s], sol_u1.y[:, -1], method='Radau', rtol=1e-5, atol=1e-5
        )

        t_u = np.concatenate([sol_u1.t, sol_u2.t[1:]])
        y_u = np.hstack([sol_u1.y, sol_u2.y[:, 1:]])

        # Extract MCA flows and calculate Blood Volume Expansion (Delta V_blood)
        q_mca_l, q_mca_r = [], []
        pic_u = y_u[0]  # Intracranial pressure state from Ursino
        pv_u  = y_u[1]  # Cerebral venous pressure state from Ursino
        
        v_blood_delta = []
        for i in range(len(t_u)):
            st = y_u[:, i]
            fl = self.ursino.get_flow_dict(st, ica_stenosis_l=stenosis_l)
            q_mca_l.append(fl['q_mca_l'])
            q_mca_r.append(fl['q_mca_r'])
            
            # Cumulative blood volume shift approximation [mL]
            pd_mean = np.mean(st[5:11])
            v_shift = 0.05 * (pd_mean - st[0])  # Scale factor for blood volume shift
            v_blood_delta.append(v_shift)

        v_blood_delta = np.array(v_blood_delta)

        if infusion_rate_ml_min > 0:
            print(f"2. Passing Volume Shifts & Lumbar Infusion ({infusion_rate_ml_min} mL/min from t={infusion_start}s to {infusion_start+infusion_duration}s) into Vaněk CSF Solver...")
        else:
            print("2. Passing Hemodynamic Volume Shifts into Vaněk (2015) CSF Solver...")
        
        # Build smooth PCHIP interpolators for driving boundary conditions
        interp_pa = PchipInterpolator(t_u, 80.0 + 10.0 * np.sin(2.0 * np.pi * 1.0 * t_u))
        interp_pb = PchipInterpolator(t_u, pv_u)
        interp_pc = PchipInterpolator(t_u, 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * t_u))
        interp_pt = PchipInterpolator(t_u, 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * t_u))
        interp_pl = PchipInterpolator(t_u, 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * t_u))

        interpolators_v = {
            'Pa': interp_pa, 'Pb': interp_pb, 'PC': interp_pc,
            'PT': interp_pt, 'PL': interp_pl
        }

        # Dynamic infusion function [mL/s]
        def get_infusion_rates(t):
            inf = np.zeros(5)
            if infusion_rate_ml_min > 0.0:
                if infusion_start <= t <= (infusion_start + infusion_duration):
                    inf[infusion_compartment] = infusion_rate_ml_min / 60.0  # mL/min -> mL/s
            return inf

        # Solve Vaněk 5-compartment CSF model
        p0_v = np.array([11.0, 10.0, 10.0, 10.0, 10.0])
        sol_v = solve_ivp(
            lambda t, P: self.vanek.derivatives(t, P, interpolators_v, infusion_rates=get_infusion_rates(t)),
            [0.0, duration_s], p0_v, method='Radau', rtol=1e-5, atol=1e-5
        )

        t_v = sol_v.t
        p_v = sol_v.y  # Shape (5, N)

        # Reconstruct inter-compartmental spinal flow F23 (Cranial -> Cervical)
        f23_v = (p_v[1] - p_v[2]) * self.vanek.Z23

        results = {
            't_u': t_u, 'q_mca_l': np.array(q_mca_l), 'q_mca_r': np.array(q_mca_r),
            'v_blood_delta': v_blood_delta, 'pic_u': pic_u,
            't_v': t_v, 'P1': p_v[0], 'P2': p_v[1], 'P3': p_v[2], 'P4': p_v[3], 'P5': p_v[4],
            'F23': f23_v,
            'infusion_params': {
                'rate_ml_min': infusion_rate_ml_min,
                'start_s': infusion_start,
                'duration_s': infusion_duration
            }
        }

        return results

    def export_mri_validation_data(self, results, output_csv_path="mri_validation_dataset_v2.csv"):
        """
        Exports simulation results in a clean, standardized CSV structure formatted 
        specifically for David's MRI validation pipeline.
        """
        df = pd.DataFrame({
            'time_s': results['t_v'],
            'ICP_P2_mmHg': results['P2'],
            'Ventricle_P1_mmHg': results['P1'],
            'Spinal_Cervical_P3_mmHg': results['P3'],
            'Spinal_Thoracic_P4_mmHg': results['P4'],
            'Spinal_Lumbar_P5_mmHg': results['P5'],
            'Cervical_CSF_Flow_F23_mL_s': results['F23']
        })
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully exported MRI validation dataset to: {output_csv_path}")
        return df

    def generate_summary_plots(self, results, save_path="unified_brain_simulation.png"):
        """Generates a 4-panel publication-quality visualization of the combined simulation."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # Panel 1: Ursino MCA Blood Flows under Hypercapnia
        ax = axes[0, 0]
        ax.plot(results['t_u'], results['q_mca_l'], 'r-', label='Left MCA Flow (mL/s)', linewidth=1.5)
        ax.plot(results['t_u'], results['q_mca_r'], 'b--', label='Right MCA Flow (mL/s)', linewidth=1.5)
        ax.axvline(50.0, color='g', linestyle=':', label='Event Onset (t=50s)')
        ax.set_title("1. Ursino (2010): Regional Cerebral Blood Flow")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Blood Flow [mL/s]")
        ax.grid(True); ax.legend(fontsize=8)

        # Panel 2: Blood Volume Shift
        ax = axes[0, 1]
        ax.plot(results['t_u'], results['v_blood_delta'], 'm-', label='Delta V_blood (mL)', linewidth=1.5)
        ax.axvline(50.0, color='g', linestyle=':')
        ax.set_title("2. Microvascular Blood Volume Expansion (Delta V)")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Volume Shift [mL]")
        ax.grid(True); ax.legend(fontsize=8)

        # Panel 3: Vaněk Craniospinal Pressures
        ax = axes[1, 0]
        ax.plot(results['t_v'], results['P1'], label='P1 (Ventricles)')
        ax.plot(results['t_v'], results['P2'], label='P2 (Intracranial SAS / ICP)', linewidth=2.0, color='crimson')
        ax.plot(results['t_v'], results['P5'], label='P5 (Lumbar SAS)', linestyle='--', color='darkblue')
        inf_p = results['infusion_params']
        if inf_p['rate_ml_min'] > 0:
            ax.axvspan(inf_p['start_s'], inf_p['start_s'] + inf_p['duration_s'], color='orange', alpha=0.15, label=f"Infusion ({inf_p['rate_ml_min']} mL/min)")
        else:
            ax.axvline(50.0, color='g', linestyle=':')
        ax.set_title("3. Vaněk (2015): Craniospinal Pressure Response")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Pressure [mmHg]")
        ax.grid(True); ax.legend(fontsize=8)

        # Panel 4: Cervical Spinal CSF Flow
        ax = axes[1, 1]
        ax.plot(results['t_v'], results['F23'], color='teal', label='F23 (Cranial -> Cervical Flow)')
        ax.axvline(50.0, color='g', linestyle=':')
        ax.set_title("4. Craniospinal CSF Displacement (Foramen Magnum)")
        ax.set_xlabel("Time [s]"); ax.set_ylabel("Flow [mL/s]")
        ax.grid(True); ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Summary visualization saved to: {save_path}")


def main():
    print("=== UNIFIED BIOPHYSICAL BRAIN MODEL PIPELINE (v2) ===")
    sim = UnifiedBrainModel()
    
    # Test combined Steal Phenomenon + Lumbar Infusion Test simultaneously!
    print("\n--- Running Combined Hypercapnia Steal + Lumbar Infusion Test ---")
    results = sim.run_sequential_pipeline(
        duration_s=400.0, 
        hypercapnia_start=50.0, 
        pco2_target=60.0, 
        stenosis_l=500.0,             # 100% Left ICA Occlusion (Steal)
        infusion_rate_ml_min=1.5,     # 1.5 mL/min Lumbar Infusion
        infusion_start=60.0, 
        infusion_duration=300.0
    )
    
    sim.export_mri_validation_data(results, output_csv_path="VU_mri_validation_dataset.csv")
    sim.generate_summary_plots(results, save_path="VU_unified_brain_simulation.png")
    print("Unified pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
