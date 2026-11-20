"""Input signals."""
import numpy as np
from scipy.interpolate import PchipInterpolator


def pchip(t, signals):
    """Shape-preserving PCHIP interpolator of every signal sampled on the grid t: {name: f(t)}."""
    return {k: PchipInterpolator(t, v) for k, v in signals.items()}


def ramp(t, pts):
    """Piecewise-linear signal through pts = [(t0, v0), (t1, v1), ...]."""
    tt, vv = zip(*pts)
    return float(np.interp(t, tt, vv))


def pulse(t, t_on, t_off, t_end, base, level, rise):
    """Scenario input: base, ramp to level in `rise` s from t_on, hold, ramp back from t_off, base until t_end."""
    return ramp(t, [(0, base), (t_on, base), (t_on + rise, level), (t_off, level), (t_off + rise, base), (t_end, base)])
