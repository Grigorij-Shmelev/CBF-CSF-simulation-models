"""
===============================================================================
Zbyšek Štěpáník (2009) 2nd Generation Craniospinal CSF Model — Lumbar Infusion Test
===============================================================================
Executes a standard Lumbar Infusion Test (LIT) on Zbyšek Štěpáník's (2009) 2nd 
generation 5-compartment model of craniospinal cerebrospinal fluid (CSF) dynamics.

Protocol matching run_vanek_lumbar_infusion.py:
  - Baseline Phase: t = 0 to 60s
  - Infusion Phase: t = 60 to 360s (1.5 mL/min = 0.025 mL/s into Lumbar SAS / compartment 5)
  - Recovery Phase: t = 360 to 500s

Usage:
    python stepanik_run_lumbar_infusion.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from cbfcsf.models.stepanik_2009 import Stepanik2009Model, run_stepanik_simulation
from cbfcsf.common.lit import lit_metrics, print_lit_metrics
from cbfcsf.paths import out_path
from cbfcsf.plots import plot_lit_figure


def lit_model():
    """Stepanik's model configured for the lumbar infusion test: 0.025 mL/s into compartment 5 from 60 s for 300 s."""
    model = Stepanik2009Model()
    # Configure infusion into compartment 5 (Lumbar SAS, index 4)
    model.infusion_rate[4] = 0.025      # 1.5 mL/min = 0.025 mL/s
    model.infusion_start[4] = 60.0       # start at t = 60s
    model.infusion_duration[4] = 300.0   # duration = 300s (ends at t = 360s)
    return model


def run_stepanik_lumbar_infusion_test():
    """
    Executes a standard Lumbar Infusion Test (LIT) on Zbyšek Štěpáník's (2009) 2nd Generation Model.
    
    Protocol:
    - Baseline Phase: t = 0 to 60s
    - Infusion Phase: t = 60 to 360s (1.5 mL/min = 0.025 mL/s into Lumbar SAS / compartment 5)
    - Recovery Phase: t = 360 to 500s
    """
    model = lit_model()

    duration_s = 500.0
    model.Tmax = duration_s

    print("Running Lumbar Infusion Test simulation on Štěpáník (2009) model (500s)...")
    
    # Execute full 500s simulation
    results = run_stepanik_simulation(model=model, duration_s=duration_s, save_plot_path=None)
    
    t = results['t']
    p1, p2, p3, p4, p5 = results['p1'], results['p2'], results['p3'], results['p4'], results['p5']
    F12, F23, F34, F45 = results['F12'], results['F23'], results['F34'], results['F45']
    V1, V2, V3, V4, V5 = results['V1'], results['V2'], results['V3'], results['V4'], results['V5']
    
    resorption = -results['resorption']  # Convert negative resorption to positive magnitude [mL/s]
    infusion_i5 = results['infusion5']
    
    C1A, C1V = results['C1A'], results['C1V']
    C2A, C2V = results['C2A'], results['C2V']
    C3C, C4T, C5L = results['C3C'], results['C4T'], results['C5L']

    # Calculate key clinical metrics
    m = lit_metrics(t, p2, p5)
    print_lit_metrics(m)

    plot_lit_figure(model, t, p1, p2, p5, results['formation'], resorption, infusion_i5, (F12, F23, F34, F45),
                    (V1, V2, V3, V4, V5),
                    [(C1A, 'C1A (Arterial-Ventricle)'), (C1V, 'C1V (Venous-Ventricle)'),
                     (C2A, 'C2A (Arterial-Cranial SAS)'), (C2V, 'C2V (Venous-Cranial SAS)'), (C5L, 'C5L (Lumbar)')],
                    "Lumbar Infusion Test (LIT) Simulation — Štěpáník (2009) 2nd Gen Model",
                    out_path("stepanik", "stepanik_lumbar_infusion_test.png"))

    return {
        't': t, 'p1': p1, 'p2': p2, 'p5': p5,
        'p0_icp': m['p0_icp'], 'pinf_icp': m['pinf_icp'], 'R_out_clinical': m['R_out_clinical']
    }


if __name__ == "__main__":
    run_stepanik_lumbar_infusion_test()
