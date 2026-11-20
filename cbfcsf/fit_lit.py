"""LIT fitting tool of the new model (spec 7.10, 7.12-7.14): fit R_out, spinal scale and resting pressure to a
lumbar infusion recording; tests on synthetic data."""
import os

import numpy as np
from scipy.integrate import solve_ivp

from cbfcsf.common.signals import pulse
from cbfcsf.models.new_model import Inputs, NewModel


# ---- LIT fitting tool ---------------------------------------------------------------------------------------------
LIT_S1 = 550.0 / 86400.0
LIT_TRUTH = (13.0, 2.5, 11.0)           # synthetic "patient": R_out [mmHg min/mL], spinal scale s, resting ICP P0 [mmHg]
LIT_DEFAULT = (1.0 / 0.11, 2.0896, 9.472)                                  # start of the fit: default model (R_out = 1/0.11)


def lit_protocol(rate_ml_min=1.5, t_base=300.0, t_inf=600.0, t_post=300.0):
    return dict(rate=rate_ml_min / 60.0, t_on=t_base, t_off=t_base + t_inf, t_end=t_base + t_inf + t_post)


def lit_curve(theta, t_eval, proto, **extra):
    """Lumbar pressure P5(t) of the model for theta = (R_out [mmHg min/mL], s, P0 [mmHg]); constant-rate infusion.
    extra: further NewModel parameters, e.g. kappa_v=0.2 to generate data from a modified model."""
    R_out, s, P0 = theta
    alpha = 1.0 / (60.0 * R_out)
    m = NewModel(alpha=alpha, Pvs_n=P0 - LIT_S1 / alpha, c_sp_scale=s, **extra)
    r0, t_on, t_off, t_end = proto["rate"], proto["t_on"], proto["t_off"], proto["t_end"]
    inf = lambda t: np.array([0, 0, 0, 0, pulse(t, t_on, t_off, t_end, 0, r0, 5)])
    inp = Inputs(Pvs=m.Pvs_n, infusion=inf)
    sol = solve_ivp(lambda t, y: m.rhs(t, y, inp), (0.0, t_end), m.initial_state(), method="Radau",
                    t_eval=t_eval, rtol=1e-6, atol=1e-8, max_step=10.0)
    return sol.y[9]


def fit_lit(t, p, proto, theta0=LIT_DEFAULT):
    """Fit (R_out, s, P0) to a measured LIT curve (time [s], pressure [mmHg]). Errors from the Jacobian at the optimum."""
    from scipy.optimize import least_squares
    resid = lambda x: lit_curve((np.exp(x[0]), np.exp(x[1]), x[2]), t, proto) - p
    x0 = np.array([np.log(theta0[0]), np.log(theta0[1]), theta0[2]])
    res = least_squares(resid, x0, x_scale=[0.2, 0.2, 1.0], diff_step=1e-3, ftol=1e-10, xtol=1e-10, max_nfev=60,
                        bounds=([np.log(2.0), np.log(0.3), 0.0], [np.log(60.0), np.log(10.0), 30.0]))
    dof = max(len(p) - 3, 1)
    sigma2 = 2.0 * res.cost / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(res.jac.T @ res.jac)))
    return dict(theta=(float(np.exp(res.x[0])), float(np.exp(res.x[1])), float(res.x[2])),
                se_rel=(float(se[0]), float(se[1])), se_P0=float(se[2]), rms=float(np.sqrt(sigma2)),
                nfev=int(res.nfev), x=res.x, se=se)


def resample_mean(t, p, dt=2.0):
    """Average a recording into dt-second bins (the fit needs about one point per 2 s).
    Bins without a sample (gaps, or sampling slower than dt) and non-finite samples are dropped; they must not enter
    the fit as zeros. Each bin is placed at the mean time of its samples."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    idx = np.floor((t - t[0]) / dt).astype(int)
    n = idx.max() + 1
    cnt = np.bincount(idx, minlength=n)
    keep = cnt > 0
    return (np.bincount(idx, weights=t, minlength=n)[keep] / cnt[keep],
            np.bincount(idx, weights=p, minlength=n)[keep] / cnt[keep])


def load_lit_csv(path):
    """CSV with a header and two columns: time [s], pressure [mmHg] (assumed format, spec 7.10)."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return data[data.dtype.names[0]], data[data.dtype.names[1]]


def _fit_task(kind, seed):
    """One fit, run in a worker process. kind: 'clean', 'noisy', 'long'."""
    if kind == "long":
        proto = lit_protocol(t_inf=1800.0)
    else:
        proto = lit_protocol()
    t = np.arange(0.0, proto["t_end"], 2.0)
    y = lit_curve(LIT_TRUTH, t, proto)
    if kind != "clean":
        y = y + 0.3 * np.random.default_rng(seed).standard_normal(len(t))
    out = fit_lit(t, y, proto)
    out.update(kind=kind, seed=seed, t=t, y=y, fit=lit_curve(out["theta"], t, proto))
    return out


def _mismatch_task(kappa):
    """Data from a model with venous squeeze kappa, fitted with the default model (spec 7.13)."""
    proto = lit_protocol()
    t = np.arange(0.0, proto["t_end"], 2.0)
    y = lit_curve(LIT_TRUTH, t, proto, kappa_v=kappa) + 0.3 * np.random.default_rng(1).standard_normal(len(t))
    out = fit_lit(t, y, proto)
    out.update(kappa=kappa)
    return out


def _one_thread_per_worker():
    """One BLAS thread per worker process: the linear algebra here is tiny and many threads per process slow it down."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def test_fit_mismatch(rep, kappas=(0.1, 0.2)):
    """M1-M4 (spec 7.13)."""
    from concurrent.futures import ProcessPoolExecutor
    _one_thread_per_worker()
    with ProcessPoolExecutor(max_workers=len(kappas)) as ex:
        out = list(ex.map(_mismatch_task, kappas))
    for f in out:
        rep.check(f"M1 kappa_v = {f['kappa']}: P0 error", f["theta"][2] - LIT_TRUTH[2], -0.3, 0.3, "mmHg")
        rep.check(f"M2 kappa_v = {f['kappa']}: R_out error", f["theta"][0] / LIT_TRUTH[0] - 1, -0.25, 0.25, "rel")
        rep.info(f"M3 kappa_v = {f['kappa']}: spinal scale error", f["theta"][1] / LIT_TRUTH[1] - 1, 0.0, np.inf, "rel")
        rep.info(f"M4 kappa_v = {f['kappa']}: rms residual", f["rms"], 0.0, 0.35, "mmHg")
    return out


def test_fit(rep, n_seeds=8, workers=8):
    """F1-F4 (spec 7.10): fit tool on synthetic LIT curves."""
    from concurrent.futures import ProcessPoolExecutor
    _one_thread_per_worker()
    tasks = [("clean", 0)] + [("noisy", sd) for sd in range(1, n_seeds + 1)] + [("long", 1)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_fit_task, *zip(*tasks)))
    f1, noisy, flong = out[0], out[1:-1], out[-1]
    rel = lambda a, b: a / b - 1.0
    rep.check("F1 noiseless: R_out", rel(f1["theta"][0], LIT_TRUTH[0]), -0.01, 0.01, "rel")
    rep.check("F1 noiseless: spinal scale", rel(f1["theta"][1], LIT_TRUTH[1]), -0.02, 0.02, "rel")
    rep.check("F1 noiseless: P0", f1["theta"][2] - LIT_TRUTH[2], -0.02, 0.02, "mmHg")
    f2 = noisy[0]
    rep.check("F2 noisy (seed 1): R_out", rel(f2["theta"][0], LIT_TRUTH[0]), -0.25, 0.25, "rel")
    rep.check("F2 noisy (seed 1): spinal scale", rel(f2["theta"][1], LIT_TRUTH[1]), -0.15, 0.15, "rel")
    rep.check("F2 noisy (seed 1): P0", f2["theta"][2] - LIT_TRUTH[2], -0.3, 0.3, "mmHg")
    xt = np.array([np.log(LIT_TRUTH[0]), np.log(LIT_TRUTH[1]), LIT_TRUTH[2]])
    z = np.abs(f2["x"] - xt) / f2["se"]
    rep.check("F2 noisy (seed 1): largest |error| / standard error", float(z.max()), 0, 3.0, "")
    ok = [abs(rel(f["theta"][0], LIT_TRUTH[0])) <= 0.25 and abs(rel(f["theta"][1], LIT_TRUTH[1])) <= 0.15
          and abs(f["theta"][2] - LIT_TRUTH[2]) <= 0.3 for f in noisy]
    rep.check(f"F3 seeds 1-{n_seeds}: number within the F2 bounds", float(sum(ok)), 0.75 * n_seeds, n_seeds, "")
    rep.info("F4 R_out standard error, 10-min / 30-min infusion", f2["se_rel"][0] / flong["se_rel"][0], 2.0, np.inf, "ratio")
    return f1, noisy, flong


def fit_file(path, rate_ml_min, t_on, t_off):
    """Fit a measured LIT (CSV: time [s], pressure [mmHg]; constant infusion rate [mL/min] from t_on to t_off [s])."""
    t, p = resample_mean(*load_lit_csv(path))
    proto = dict(rate=rate_ml_min / 60.0, t_on=t_on, t_off=t_off, t_end=float(t[-1]) + 1.0)
    f = fit_lit(t, p, proto)
    R, sc, P0 = f["theta"]
    m = NewModel(alpha=1 / (60 * R), Pvs_n=P0 - LIT_S1 * 60 * R, c_sp_scale=sc)
    print(f"R_out = {R:.2f} mmHg min/mL (+-{100 * f['se_rel'][0]:.0f} %)   spinal scale = {sc:.2f} (+-{100 * f['se_rel'][1]:.0f} %)   "
          f"P0 = {P0:.2f} +- {f['se_P0']:.2f} mmHg   storage at rest {m.storage_at_rest():.2f} mL/mmHg   rms residual {f['rms']:.2f} mmHg")
    return f
