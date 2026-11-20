"""Input signals."""
from scipy.interpolate import PchipInterpolator


def pchip(t, signals):
    """Shape-preserving PCHIP interpolator of every signal sampled on the grid t: {name: f(t)}."""
    return {k: PchipInterpolator(t, v) for k, v in signals.items()}
