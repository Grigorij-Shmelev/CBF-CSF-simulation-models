"""Physical laws used by more than one model."""
import numpy as np


def hagen_poiseuille(r, l, eta):
    """Hagen-Poiseuille resistance [mmHg s/mL] of a vessel of radius r and length l [cm]; eta: viscosity [mmHg s]."""
    return 8.0 * eta * l / (np.pi * r**4)
