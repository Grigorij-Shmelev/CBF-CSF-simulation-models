"""Input signals."""
import csv
from dataclasses import dataclass

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


# ----------------------------------------------------------------------------------------------------------------------
# End-tidal-shaped CO2 input (spec 7.17)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EtCO2Protocol:
    """Inhaled-gas schedule and end-tidal shape of spec 7.17, all in one place.
    Protocol: 5 % CO2 inhaled for 3 min, 3 cycles (a hypercapnia test in the MRI scanner).
    Values marked PLACEHOLDER come from no source; they stand in until measured capnometer curves are available."""
    P0: float = 40.0          # baseline end-tidal CO2 [mmHg] = the model's PaCO2_n
    dP: float = 7.5           # PLACEHOLDER: plateau 47.5 mmHg, the student's assumption for 5 % CO2 (as test C)
    tau_on: float = 20.0      # PLACEHOLDER: time constant of the rise [s]
    tau_off: float = 20.0     # PLACEHOLDER: time constant of the decay [s]
    delay: float = 0.0        # transport delay [s] (option): the whole curve is shifted by it
    gap: float = 0.0          # arterial minus end-tidal [mmHg]. ASSUMPTION: end-tidal CO2 taken as arterial CO2
    t_rest: float = 120.0     # rest before the first inhalation [s] (as test C)
    n_cycles: int = 3         # number of inhalations (protocol)
    t_inhale: float = 180.0   # length of one inhalation [s] (protocol)
    t_air: float = 90.0       # air between inhalations [s]: the student's upper bound (21 Nov), exact value unknown
    t_after: float = 1200.0   # air after the last inhalation [s] (as test C)

    @property
    def onsets(self):
        return tuple(self.t_rest + k * (self.t_inhale + self.t_air) for k in range(self.n_cycles))

    @property
    def offsets(self):
        return tuple(t + self.t_inhale for t in self.onsets)

    @property
    def t_end(self):
        return self.offsets[-1] + self.t_after


def _scalar_or_array(out):
    return float(out) if np.ndim(out) == 0 else out


def etco2_curve(proto=EtCO2Protocol()):
    """End-tidal-shaped PaCO2(t) [mmHg] (spec 7.17). The schedule u(t) is a square wave (P0, and P0 + dP during the
    inhalations); the curve follows it with a first-order lag, in closed form on each segment between switch times t_s:
        P(t) = u + (P(t_s) - u) exp(-(t - t_s) / tau),   tau = tau_on rising, tau_off falling.
    Continuous, no ODE; a decay that has not finished carries over into the next inhalation. Returns P + gap.
    A plain function of t (scalar or array): a fitted end-tidal function can replace it in this one place."""
    ts, u, tau = [], [], []
    for on, off in zip(proto.onsets, proto.offsets):
        ts += [on, off]
        u += [proto.P0 + proto.dP, proto.P0]
        tau += [proto.tau_on, proto.tau_off]
    ts, u, tau = np.array(ts), np.array(u), np.array(tau)
    p_start = np.empty(len(ts))           # P(t_s): the value the previous segment has reached at the switch
    p = proto.P0
    for k in range(len(ts)):
        if k:
            p = u[k - 1] + (p_start[k - 1] - u[k - 1]) * np.exp(-(ts[k] - ts[k - 1]) / tau[k - 1])
        p_start[k] = p

    def f(t):
        t = np.asarray(t, float) - proto.delay
        i = np.searchsorted(ts, t, side="right") - 1
        j = np.maximum(i, 0)
        seg = u[j] + (p_start[j] - u[j]) * np.exp(-np.maximum(t - ts[j], 0.0) / tau[j])
        return _scalar_or_array(np.where(i < 0, proto.P0, seg) + proto.gap)
    return f


def etco2_step(proto=EtCO2Protocol(), rise=10.0):
    """The current switch-like input on the same schedule, for comparison: linear ramps of `rise` s (as pulse())."""
    pts = [(0.0, proto.P0)]
    for on, off in zip(proto.onsets, proto.offsets):
        pts += [(on, proto.P0), (on + rise, proto.P0 + proto.dP), (off, proto.P0 + proto.dP), (off + rise, proto.P0)]
    pts.append((max(proto.t_end, pts[-1][0]), proto.P0))
    tt, vv = (np.array(x) for x in zip(*pts))

    def f(t):
        return _scalar_or_array(np.interp(np.asarray(t, float) - proto.delay, tt, vv) + proto.gap)
    return f


# Units of a measured end-tidal curve (spec 7.17). The unit must be given; it is never guessed.
# kPa: 1 kPa = 7.50062 mmHg. percent: partial pressure of a gas fraction F in humidified alveolar gas, F/100 (P_B - 47),
# with 47 mmHg = water vapour pressure at 37 C. Standard physiology, NOT from the project sources.
ETCO2_UNITS = ("mmHg", "kPa", "percent")
KPA_TO_MMHG = 7.50062
P_H2O_37C = 47.0


@dataclass
class EtCO2Data:
    """A measured end-tidal curve, read by read_etco2_csv(). Times are in the CSV's own clock."""
    t: np.ndarray        # sample times [s]
    p: np.ndarray        # end-tidal CO2 [mmHg]
    baseline: float      # mean of the first 60 s of data [mmHg]
    dropped: int         # non-finite or unreadable samples that were dropped

    def to_model_time(self, t_csv, pad=120.0):
        """Model time of a CSV time: t = 0 lies `pad` s before the first sample."""
        return np.asarray(t_csv, float) - self.t[0] + pad

    def function(self, pad=120.0, gap=0.0):
        """PaCO2(t) in model time [mmHg]: PCHIP through the samples, the first and last values held outside the data
        range (PCHIP extrapolation runs away, and the solver steps past the last sample). The pad before the first
        sample holds the first value, so the model starts at rest."""
        interp = pchip(self.t, {"p": self.p})["p"]
        t0, t1 = self.t[0], self.t[-1]

        def f(t):
            return _scalar_or_array(interp(np.clip(np.asarray(t, float) - pad + t0, t0, t1)) + gap)
        return f

    def check_cycles(self, onsets, length):
        """Onsets (CSV clock) must increase, lie inside the data, and onset + length must not pass the last sample:
        otherwise the held last value would make the end of a window flat without warning."""
        onsets = list(onsets)
        if any(b <= a for a, b in zip(onsets, onsets[1:])):
            raise ValueError(f"--cycles must be strictly increasing, got {onsets}")
        for t_k in onsets:
            if t_k < self.t[0] or t_k + length > self.t[-1]:
                raise ValueError(f"cycle at {t_k:g} s with length {length:g} s does not fit inside the data "
                                 f"({self.t[0]:g} to {self.t[-1]:g} s, CSV clock)")


def read_etco2_csv(path, unit, barometric=760.0, baseline_window=60.0):
    """Read a measured end-tidal CO2 curve (spec 7.17): a CSV with the header `time_s, etco2`.
    unit: 'mmHg', 'kPa' or 'percent' (required). barometric: P_B [mmHg] for 'percent'.
    Non-finite and unreadable samples are dropped; time must then be strictly increasing."""
    if unit not in ETCO2_UNITS:
        raise ValueError(f"etco2 unit must be one of {', '.join(ETCO2_UNITS)}; got {unit!r}")
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]] if rows else []
    if header != ["time_s", "etco2"]:
        raise ValueError(f"{path}: the header must be 'time_s, etco2'; found {', '.join(header)!r}")

    def num(s):
        try:
            return float(s)
        except ValueError:
            return np.nan

    vals = []
    for n, row in enumerate(rows[1:], start=2):
        if not any(c.strip() for c in row):
            continue                                   # blank line
        if len(row) != 2:
            raise ValueError(f"{path}, line {n}: expected 2 columns (time_s, etco2), found {len(row)}")
        vals.append((num(row[0]), num(row[1])))
    a = np.array(vals, float).reshape(-1, 2)
    ok = np.isfinite(a).all(axis=1)
    t, v = a[ok, 0], a[ok, 1]
    if len(t) < 2:
        raise ValueError(f"{path}: fewer than 2 finite samples")
    bad = np.flatnonzero(np.diff(t) <= 0)
    if bad.size:
        i = bad[0]
        raise ValueError(f"{path}: time_s must be strictly increasing; {t[i + 1]:g} s follows {t[i]:g} s "
                         f"(finite samples {i + 1} and {i + 2})")
    if unit == "kPa":
        v = v * KPA_TO_MMHG
    elif unit == "percent":
        v = v / 100.0 * (barometric - P_H2O_37C)
    baseline = float(np.mean(v[t < t[0] + baseline_window]))
    return EtCO2Data(t=t, p=v, baseline=baseline, dropped=int((~ok).sum()))


def write_etco2_csv(path, t, p):
    """Write a curve in the format read_etco2_csv() reads (header time_s,etco2; mmHg)."""
    with open(path, "w", newline="") as fh:
        fh.write("time_s,etco2\n")
        for ti, pi in zip(t, p):
            fh.write(f"{ti:.3f},{pi:.12f}\n")
