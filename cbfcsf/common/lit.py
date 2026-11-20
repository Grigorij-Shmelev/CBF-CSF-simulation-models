"""Lumbar infusion test: the metrics of the Vanek and Stepanik LIT checks (baseline 10-50 s, plateau 300-350 s)."""
import numpy as np


def lit_metrics(t, p2, p5, I_inf=0.025):
    """Mean ICP p2 and lumbar pressure p5 at baseline and plateau, outflow resistance from the ICP rise."""
    baseline_mask = (t >= 10) & (t <= 50)
    infusion_plateau_mask = (t >= 300) & (t <= 350)

    p0_icp = np.mean(p2[baseline_mask])
    pinf_icp = np.mean(p2[infusion_plateau_mask])
    dp_icp = pinf_icp - p0_icp

    p0_lumb = np.mean(p5[baseline_mask])
    pinf_lumb = np.mean(p5[infusion_plateau_mask])
    dp_lumb = pinf_lumb - p0_lumb

    R_out_calc = dp_icp / I_inf  # mmHg * s / mL
    R_out_clinical = R_out_calc / 60.0  # mmHg / (mL/min)
    return dict(p0_icp=p0_icp, pinf_icp=pinf_icp, dp_icp=dp_icp, p0_lumb=p0_lumb, pinf_lumb=pinf_lumb,
                dp_lumb=dp_lumb, R_out_calc=R_out_calc, R_out_clinical=R_out_clinical)


def print_lit_metrics(m):
    print("\n--- Infusion Test Metrics ---")
    print(f"Baseline ICP (P2): {m['p0_icp']:.2f} mmHg ({m['p0_icp'] * 13.6:.1f} mmH2O)")
    print(f"Plateau ICP (P2):  {m['pinf_icp']:.2f} mmHg ({m['pinf_icp'] * 13.6:.1f} mmH2O)")
    print(f"ICP Elevation (ΔP): {m['dp_icp']:.2f} mmHg ({m['dp_icp'] * 13.6:.1f} mmH2O)")
    print(f"Baseline Lumbar Pressure (P5): {m['p0_lumb']:.2f} mmHg ({m['p0_lumb'] * 13.6:.1f} mmH2O)")
    print(f"Plateau Lumbar Pressure (P5):  {m['pinf_lumb']:.2f} mmHg ({m['pinf_lumb'] * 13.6:.1f} mmH2O)")
    print(f"Calculated Outflow Resistance Rout: {m['R_out_calc']:.2f} mmHg*s/mL ({m['R_out_clinical']:.2f} mmHg/(mL/min))")
