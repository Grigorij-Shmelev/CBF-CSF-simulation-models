"""Physical laws used by more than one model."""
import numpy as np


def hagen_poiseuille(r, l, eta):
    """Hagen-Poiseuille resistance [mmHg s/mL] of a vessel of radius r and length l [cm]; eta: viscosity [mmHg s]."""
    return 8.0 * eta * l / (np.pi * r**4)


def wall_compliance(C0, r, a, dP):
    """Non-linear compliance of a wall between two spaces (Vanek, Stepanik): C0 exp(-r |dP|^a); dP: pressure difference."""
    return C0 * np.exp(-r * (abs(dP)**a))
