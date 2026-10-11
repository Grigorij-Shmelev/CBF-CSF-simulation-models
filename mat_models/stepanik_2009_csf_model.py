import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

class CSFAttributes:
    """Container for parameters matching Settings.parameters in MATLAB code."""
    def __init__(self):
        # Odpory [mmHg*s/ml]
        self.R12 = 1.60e+2
        self.R23 = 3.00e-1
        self.R34 = 4.00e-1
        self.R45 = 3.00e-1
        
        self.Z12 = 1.0 / self.R12
        self.Z23 = 1.0 / self.R23
        self.Z34 = 1.0 / self.R34
        self.Z45 = 1.0 / self.R45

        # Poddajnosti [ml/mmHg] - maximalni poddajnost v 0
        self.C1A0 = 1.500e-4   # rozhrani arterie - komory 
        self.C2A0 = 2.000e-1   # rozhrani arterie - SAS
        self.C1V0 = 6e-1       # rozhrani zily - komory
        self.C2V0 = 6e-1       # rozhrani zily - SAS
        self.C3C0 = 1.125e+0   # krcni
        self.C4T0 = 1.500e+0   # hrudni
        self.C5L0 = 2.250e+0   # bederni

        # Poddajnosti - krivost exponencialy (a)
        self.a1A = 3e-1
        self.a2A = 3e-1
        self.a1V = 6e-1
        self.a2V = 6e-1
        self.a3C = 6e-1
        self.a4T = 6e-1
        self.a5L = 6e-1

        # Poddajnosti - sklon (r)
        self.r1A = 10e-1
        self.r2A = 10e-1
        self.r1V = 6e-1
        self.r2V = 6e-1
        self.r3C = 6e-1
        self.r4T = 6e-1
        self.r5L = 6e-1

        # Objemy [ml]
        self.V10 = 30.0   # komorovy system
        self.V20 = 90.0   # SAS
        self.V30 = 10.0   # krcni
        self.V40 = 10.0   # hrudni
        self.V50 = 10.0   # bederni

        # Tlaky initial relative differentials [mmHg]
        self.p1A0 = 11.0 - 80.0  # -69
        self.p2A0 = 10.0 - 80.0  # -70
        self.p1V0 = 10.0 - 2.0   # 8
        self.p2V0 = 10.0 - 2.0   # 8
        self.p3C0 = 10.0 - 2.0   # 8
        self.p4T0 = 10.0 - 2.0   # 8
        self.p5L0 = 10.0 - 2.0   # 8

        # Zdroje [ml/s]
        self.S1 = 550.0 / 24.0 / 3600.0  # ~0.00636574 ml/s
        self.S2 = 0.0
        self.S3 = 0.0
        self.S4 = 0.0
        self.S5 = 0.0

        # Resorpce
        self.peq = 10.0                  # mmHg
        self.alpha = 0.11 / 60.0         # ml/(mmHg*s)
        self.pmin = (self.alpha * self.peq - self.S1) / self.alpha  # mmHg

        # Infuze (0-indexed in Python: 5 elements)
        self.rychlost_infuze = np.zeros(5)  # ml/s
        self.start_infuze = np.zeros(5)     # s
        self.doba_infuze = np.zeros(5)      # s

        # Ostatni solver parameters
        self.Tmax = 1.0
        self.RelTol = 1e-3
        self.AbsTol = 1e-3
        self.InitialStep = 1e-3
        self.MaxStep = 1e-1
        self.interpolovat = 0
        self.inter_per = 0.001
        
        # Mode switch: True = exact MATLAB p(2)+pAi formula, False = physiological p2 = p(2)+pVi
        self.strict_matlab_mode = False


def resorption(tlak, parameters):
    """Linear pressure-dependent resorption function matching MATLAB resorption()."""
    if tlak <= parameters.pmin:
        return 0.0
    else:
        return -parameters.alpha * (tlak - parameters.pmin)


def infusion(tau, index, parameters):
    """Constant-flow infusion function matching MATLAB infusion(). Note: index is 0-4."""
    start = parameters.start_infuze[index]
    doba = parameters.doba_infuze[index]
    if tau > (start + doba) or tau < start:
        return 0.0
    else:
        return parameters.rychlost_infuze[index]


def d_P(t, p, settings, interpolators):
    """
    Derivative function dp/dt corresponding to d_P(t, p, Settings) in MATLAB.
    State vector p = [p1A, p2V, p3C, p4T, p5L]
    Uses PchipInterpolator for shape-preserving, overshoot-free waveform interpolation.
    """
    params = settings['parameters']

    # Interpolate driving pressure differences at current time t using PCHIP
    pAVi = float(interpolators['pAV'](t))
    pVCi = float(interpolators['pVC'](t))
    pCTi = float(interpolators['pCT'](t))
    pTLi = float(interpolators['pTL'](t))

    # Numerical derivative of pAVi (matching MATLAB eps^(1/3) step)
    ms = (np.finfo(float).eps)**(1.0/3.0) * max(1.0, abs(t))
    pAVii = float(interpolators['pAV'](t + ms))
    dpAVi = (pAVii - pAVi) / ms

    # Absolute driving pressures
    pAi = float(interpolators['pA'](t))
    pVi = float(interpolators['pV'](t))

    # Non-linear compartmental compliances
    C1Ai = params.C1A0 * np.exp(-params.r1A * (abs(p[0])**params.a1A))
    C2Ai = params.C2A0 * np.exp(-params.r2A * (abs(p[1] - pAVi)**params.a2A))
    C1Vi = params.C1V0 * np.exp(-params.r1V * (abs(p[0] + pAVi)**params.a1V))
    C2Vi = params.C2V0 * np.exp(-params.r2V * (abs(p[1])**params.a2V))
    C3Ci = params.C3C0 * np.exp(-params.r3C * (abs(p[2])**params.a3C))
    C4Ti = params.C4T0 * np.exp(-params.r4T * (abs(p[3])**params.a4T))
    C5Li = params.C5L0 * np.exp(-params.r5L * (abs(p[4])**params.a5L))

    # Resorption pressure target:
    # MATLAB original line 351: resorption(p(2)+pAi, Settings) -> p[1] + pAi
    # Physiological correction: p2 = p2V + pV = p[1] + pVi
    if params.strict_matlab_mode:
        resorption_p = p[1] + pAi
    else:
        resorption_p = p[1] + pVi

    dpdt = np.zeros(5)
    dpdt[0] = 1.0 / (C1Ai + C1Vi) * (
        -params.Z12 * p[0] + params.Z12 * p[1] - params.Z12 * pAVi
        + params.S1 + infusion(t, 0, params) - C1Vi * dpAVi
    )
    dpdt[1] = 1.0 / (C2Ai + C2Vi) * (
        params.Z12 * p[0] - (params.Z12 + params.Z23) * p[1] + params.Z23 * p[2]
        + params.Z12 * pAVi - params.Z23 * pVCi + params.S2
        + infusion(t, 1, params) + resorption(resorption_p, params) + C2Ai * dpAVi
    )
    dpdt[2] = 1.0 / C3Ci * (
        params.Z23 * p[1] - (params.Z23 + params.Z34) * p[2] + params.Z34 * p[3]
        + params.Z23 * pVCi - params.Z34 * pCTi + params.S3 + infusion(t, 2, params)
    )
    dpdt[3] = 1.0 / C4Ti * (
        params.Z34 * p[2] - (params.Z34 + params.Z45) * p[3] + params.Z45 * p[4]
        + params.Z34 * pCTi - params.Z45 * pTLi + params.S4 + infusion(t, 3, params)
    )
    dpdt[4] = 1.0 / C5Li * (
        params.Z45 * p[3] - params.Z45 * p[4] + params.Z45 * pTLi
        + params.S5 + infusion(t, 4, params)
    )

    return dpdt


def CSF_model(settings):
    """
    Main simulation execution function corresponding 1:1 to CSF_model(Settings) in MATLAB.
    Uses PchipInterpolator for exact shape-preserving 1:1 MATLAB pchip behavior.
    """
    params = settings['parameters']
    inputs = settings['inputs']

    # Calculate input pressure differences
    params.pAV = inputs['pA'] - inputs['pV']
    params.pVC = inputs['pV'] - inputs['pC']
    params.pCT = inputs['pC'] - inputs['pT']
    params.pTL = inputs['pT'] - inputs['pL']

    # Pre-build PCHIP interpolators for input pressure signals (exact match to MATLAB pchip)
    time_arr = inputs['time']
    interpolators = {
        'pA': PchipInterpolator(time_arr, inputs['pA']),
        'pV': PchipInterpolator(time_arr, inputs['pV']),
        'pC': PchipInterpolator(time_arr, inputs['pC']),
        'pT': PchipInterpolator(time_arr, inputs['pT']),
        'pL': PchipInterpolator(time_arr, inputs['pL']),
        'pAV': PchipInterpolator(time_arr, params.pAV),
        'pVC': PchipInterpolator(time_arr, params.pVC),
        'pCT': PchipInterpolator(time_arr, params.pCT),
        'pTL': PchipInterpolator(time_arr, params.pTL),
    }

    # Define time span
    if params.Tmax > 0:
        tspan = (0.0, float(params.Tmax))
    else:
        tspan = (0.0, float(inputs['time'][-1]))

    p0 = np.array([params.p1A0, params.p2V0, params.p3C0, params.p4T0, params.p5L0], dtype=float)

    # Solve system of ODEs
    sol = solve_ivp(
        fun=lambda t, y: d_P(t, y, settings, interpolators),
        t_span=tspan,
        y0=p0,
        method='RK45',
        rtol=params.RelTol,
        atol=params.AbsTol,
        max_step=params.MaxStep,
        first_step=params.InitialStep
    )

    t = sol.t
    p = sol.y.T  # shape (N, 5)

    results = {'t': t, 'p': p}

    # Interpolate results if requested
    if params.interpolovat == 1:
        t_interp = np.arange(0, params.Tmax + params.inter_per, params.inter_per)
        p_interp = PchipInterpolator(t, p, axis=0)(t_interp)
        t = t_interp
        p = p_interp
        results['t'] = t
        results['p'] = p

    # Reconstruct input pressures on simulation time grid using PCHIP
    results['pA'] = interpolators['pA'](t)
    results['pV'] = interpolators['pV'](t)
    results['pC'] = interpolators['pC'](t)
    results['pT'] = interpolators['pT'](t)
    results['pL'] = interpolators['pL'](t)

    # Reconstruct absolute compartmental pressures
    results['p1'] = p[:, 0] + results['pA']
    results['p2'] = p[:, 1] + results['pV']
    results['p3'] = p[:, 2] + results['pC']
    results['p4'] = p[:, 3] + results['pT']
    results['p5'] = p[:, 4] + results['pL']

    # Inter-compartmental fluid flows [ml/s]
    results['F12'] = (results['p1'] - results['p2']) / params.R12
    results['F23'] = (results['p2'] - results['p3']) / params.R23
    results['F34'] = (results['p3'] - results['p4']) / params.R34
    results['F45'] = (results['p4'] - results['p5']) / params.R45

    # CSF formation, resorption, and infusions
    n_points = len(t)
    results['tvorba'] = params.S1 * np.ones(n_points)
    
    if params.strict_matlab_mode:
        results['vstrebavani'] = np.array([resorption(p[i, 1] + results['pA'][i], params) for i in range(n_points)])
    else:
        results['vstrebavani'] = np.array([resorption(results['p2'][i], params) for i in range(n_points)])

    results['infuze1'] = np.array([infusion(t[i], 0, params) for i in range(n_points)])
    results['infuze2'] = np.array([infusion(t[i], 1, params) for i in range(n_points)])
    results['infuze3'] = np.array([infusion(t[i], 2, params) for i in range(n_points)])
    results['infuze4'] = np.array([infusion(t[i], 3, params) for i in range(n_points)])
    results['infuze5'] = np.array([infusion(t[i], 4, params) for i in range(n_points)])

    # Compartmental volumes [ml] using cumulative trapezoidal integration
    d_V1 = -results['F12'] + params.S1 + results['infuze1']
    d_V2 = results['F12'] - results['F23'] + params.S2 + results['vstrebavani'] + results['infuze2']
    d_V3 = results['F23'] - results['F34'] + params.S3 + results['infuze3']
    d_V4 = results['F34'] - results['F45'] + params.S4 + results['infuze4']
    d_V5 = results['F45'] + params.S5 + results['infuze5']

    results['V1'] = cumulative_trapezoid(d_V1, t, initial=0) + params.V10
    results['V2'] = cumulative_trapezoid(d_V2, t, initial=0) + params.V20
    results['V3'] = cumulative_trapezoid(d_V3, t, initial=0) + params.V30
    results['V4'] = cumulative_trapezoid(d_V4, t, initial=0) + params.V40
    results['V5'] = cumulative_trapezoid(d_V5, t, initial=0) + params.V50

    # Compartmental compliances [ml/mmHg]
    results['C1A'] = params.C1A0 * np.exp(-params.r1A * (abs(results['p1'] - results['pA'])**params.a1A))
    results['C2A'] = params.C2A0 * np.exp(-params.r2A * (abs(results['p2'] - results['pA'])**params.a2A))
    results['C1V'] = params.C1V0 * np.exp(-params.r1V * (abs(results['p1'] - results['pV'])**params.a1V))
    results['C2V'] = params.C2V0 * np.exp(-params.r2V * (abs(results['p2'] - results['pV'])**params.a2V))
    results['C3C'] = params.C3C0 * np.exp(-params.r3C * (abs(results['p3'] - results['pC'])**params.a3C))
    results['C4T'] = params.C4T0 * np.exp(-params.r4T * (abs(results['p4'] - results['pT'])**params.a4T))
    results['C5L'] = params.C5L0 * np.exp(-params.r5L * (abs(results['p5'] - results['pL'])**params.a5L))

    return results


def run_default_simulation(duration_s=30.0, save_plot_path=None, strict_matlab_mode=False):
    """
    Helper function to generate default inputs, configure settings, run simulation, and optionally plot.
    """
    params = CSFAttributes()
    params.Tmax = duration_s
    params.strict_matlab_mode = strict_matlab_mode

    # Generate synthetic input pressure waveforms (Cardiac 1 Hz + Respiratory 0.25 Hz)
    time = np.linspace(0, duration_s, int(duration_s * 500))
    
    # Arterial pressure pA (mean 80, pulse amp 10)
    pA = 80.0 + 10.0 * np.sin(2.0 * np.pi * 1.0 * time)
    # Venous pressure pV (mean 2)
    pV = 2.0 + 0.5 * np.sin(2.0 * np.pi * 0.25 * time)
    # Epidural pressures
    pC = 2.0 + 0.3 * np.sin(2.0 * np.pi * 0.25 * time)
    pT = 2.0 + 1.2 * np.sin(2.0 * np.pi * 0.25 * time)  # respiration effect
    pL = 2.0 + 0.4 * np.sin(2.0 * np.pi * 0.25 * time)

    inputs = {
        'time': time,
        'pA': pA,
        'pV': pV,
        'pC': pC,
        'pT': pT,
        'pL': pL
    }

    settings = {'parameters': params, 'inputs': inputs}
    results = CSF_model(settings)

    if save_plot_path:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        # Plot Pressures
        ax = axes[0, 0]
        ax.plot(results['t'], results['p1'], label='p1 (Ventricles)')
        ax.plot(results['t'], results['p2'], label='p2 (Intracranial SAS)')
        ax.plot(results['t'], results['p3'], label='p3 (Cervical SAS)')
        ax.plot(results['t'], results['p4'], label='p4 (Thoracic SAS)')
        ax.plot(results['t'], results['p5'], label='p5 (Lumbar SAS)')
        ax.set_title("Craniospinal Pressures [mmHg]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pressure [mmHg]")
        ax.grid(True)
        ax.legend()

        # Plot Flows
        ax = axes[0, 1]
        ax.plot(results['t'], results['F12'], label='F12 (Ventr -> SAS)')
        ax.plot(results['t'], results['F23'], label='F23 (Cranial -> Cerv)')
        ax.plot(results['t'], results['F34'], label='F34 (Cerv -> Thor)')
        ax.plot(results['t'], results['F45'], label='F45 (Thor -> Lumb)')
        ax.set_title("Inter-compartmental CSF Flows [ml/s]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flow [ml/s]")
        ax.grid(True)
        ax.legend()

        # Plot Volumes
        ax = axes[1, 0]
        ax.plot(results['t'], results['V1'], label='V1')
        ax.plot(results['t'], results['V2'], label='V2')
        ax.plot(results['t'], results['V3'], label='V3')
        ax.plot(results['t'], results['V4'], label='V4')
        ax.plot(results['t'], results['V5'], label='V5')
        ax.set_title("Compartmental Volumes [ml]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Volume [ml]")
        ax.grid(True)
        ax.legend()

        # Plot Formation and Resorption
        ax = axes[1, 1]
        ax.plot(results['t'], results['tvorba'], label='Formation (S1)')
        ax.plot(results['t'], -results['vstrebavani'], label='Resorption (-Fout)')
        ax.set_title("Formation & Resorption [ml/s]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flow [ml/s]")
        ax.grid(True)
        ax.legend()

        # Plot Global P-V Curve
        ax = axes[2, 0]
        dV_total = (results['V1'] + results['V2'] + results['V3'] + results['V4'] + results['V5']) - \
                   (params.V10 + params.V20 + params.V30 + params.V40 + params.V50)
        ax.plot(dV_total, results['p2'], label='P-V Curve')
        ax.set_title("Global Pressure-Volume Relationship")
        ax.set_xlabel("ΔV Total [ml]")
        ax.set_ylabel("ICP p2 [mmHg]")
        ax.grid(True)
        ax.legend()

        # Plot Compliances
        ax = axes[2, 1]
        ax.plot(results['t'], results['C1A'], label='C1A')
        ax.plot(results['t'], results['C2A'], label='C2A')
        ax.plot(results['t'], results['C3C'], label='C3C')
        ax.plot(results['t'], results['C4T'], label='C4T')
        ax.plot(results['t'], results['C5L'], label='C5L')
        ax.set_title("Compartmental Compliances [ml/mmHg]")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Compliance [ml/mmHg]")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=150)
        plt.close()

    return results


if __name__ == "__main__":
    print("Testing translated CSF simulation model v3 (with PCHIP)...")
    res = run_default_simulation(duration_s=10.0, save_plot_path="./stepanik_csf_verification_plot.png")
    print("Simulation finished successfully!")
    print(f"Mean p1 (Ventricles): {np.mean(res['p1']):.2f} mmHg")
    print(f"Mean p2 (ICP): {np.mean(res['p2']):.2f} mmHg")
    print(f"Mean p5 (Lumbar): {np.mean(res['p5']):.2f} mmHg")
