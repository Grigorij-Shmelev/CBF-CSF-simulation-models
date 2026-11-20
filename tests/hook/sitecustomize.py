"""Capture hook for the golden-output tests (tests/capture_golden.py, tests/test_regression.py).

Python imports this file at start-up when tests/hook is on PYTHONPATH (the test harness sets that). It first runs
the system's own sitecustomize, then does nothing more unless the environment variable CAPTURE_DIR is set.
With CAPTURE_DIR set it records, without changing any computation:
  - every scipy.integrate.solve_ivp result: full arrays in the main process, a SHA-256 per call in worker
    processes (--fit runs its fits in a process pool; the order of calls between workers is not fixed);
  - the data of every matplotlib figure when it is saved (line data, patch geometry, texts) and the SHA-256 of
    the saved file;
  - the values returned by ProcessPoolExecutor.map in the main process.
The main process writes CAPTURE_DIR/main.npz and main.json at exit; workers append to CAPTURE_DIR/workers.txt.
"""
import os
import sys


def _chain_system_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in sys.path:
        if not p or os.path.abspath(p) == here:
            continue
        f = os.path.join(p, "sitecustomize.py")
        if os.path.isfile(f):
            with open(f) as fh:
                exec(compile(fh.read(), f, "exec"), {"__name__": "sitecustomize", "__file__": f})
            return


def _install():
    import atexit
    import functools
    import hashlib
    import json
    import numpy as np
    from capture_util import digest, flatten

    out_dir = os.environ["CAPTURE_DIR"]
    os.environ.setdefault("CAPTURE_MAIN_PID", str(os.getpid()))   # children inherit it: they are not main
    is_main = lambda: os.environ.get("CAPTURE_MAIN_PID") == str(os.getpid())
    arrays, texts = {}, {}
    count = {"solve": 0, "fig": 0, "pool": 0}

    # ---- solve_ivp ------------------------------------------------------------------------------------------------
    import scipy.integrate as si
    orig_solve = si.solve_ivp

    @functools.wraps(orig_solve)
    def solve_ivp(fun, t_span, y0, *args, **kwargs):
        sol = orig_solve(fun, t_span, y0, *args, **kwargs)
        stats = np.array([sol.status, sol.nfev, sol.njev, sol.nlu])
        if is_main():
            k = f"solve/{count['solve']:04d}"
            count["solve"] += 1
            arrays[f"{k}/t"], arrays[f"{k}/y"], arrays[f"{k}/stats"] = sol.t, sol.y, stats
            arrays[f"{k}/t_span"], arrays[f"{k}/y0"] = np.asarray(t_span, float), np.asarray(y0, float)
            if kwargs.get("t_eval") is not None:
                arrays[f"{k}/t_eval"] = np.asarray(kwargs["t_eval"], float)
            texts[f"{k}/options"] = {o: repr(kwargs.get(o)) for o in ("method", "rtol", "atol", "max_step", "first_step")}
            texts[f"{k}/message"] = sol.message
        else:
            line = f"{digest(sol.t, sol.y, stats)}\n".encode()
            fd = os.open(os.path.join(out_dir, "workers.txt"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        return sol

    si.solve_ivp = solve_ivp

    # ---- matplotlib savefig ---------------------------------------------------------------------------------------
    import matplotlib.colors as mcolors
    import matplotlib.figure as mfig
    orig_savefig = mfig.Figure.savefig

    def figure_data(fig, k):
        sup = fig._suptitle.get_text() if fig._suptitle is not None else None
        texts[f"{k}/figure"] = {"suptitle": sup, "size": [float(v) for v in fig.get_size_inches()]}
        for i, ax in enumerate(fig.axes):
            p = f"{k}/ax{i}"
            leg = ax.get_legend()
            texts[f"{p}/axes"] = {"title": [ax.get_title(loc) for loc in ("left", "center", "right")],
                                  "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                                  "scale": [ax.get_xscale(), ax.get_yscale()],
                                  "texts": [t.get_text() for t in ax.texts],
                                  "legend": [t.get_text() for t in leg.get_texts()] if leg is not None else None}
            for j, ln in enumerate(ax.get_lines()):
                q = f"{p}/line{j}"
                arrays[f"{q}/x"] = np.asarray(ln.get_xdata(), dtype=float)
                arrays[f"{q}/y"] = np.asarray(ln.get_ydata(), dtype=float)
                texts[f"{q}/style"] = {"label": ln.get_label(), "color": mcolors.to_hex(ln.get_color(), keep_alpha=True),
                                       "ls": str(ln.get_linestyle()), "lw": float(ln.get_linewidth()),
                                       "marker": str(ln.get_marker()), "ms": float(ln.get_markersize()),
                                       "alpha": ln.get_alpha(), "zorder": float(ln.get_zorder())}
            for j, pa in enumerate(ax.patches):
                q = f"{p}/patch{j}"
                arrays[f"{q}/xy"] = np.asarray(pa.get_patch_transform().transform(pa.get_path().vertices), float)
                texts[f"{q}/style"] = {"type": type(pa).__name__, "fc": mcolors.to_hex(pa.get_facecolor(), keep_alpha=True),
                                       "label": pa.get_label()}
            for j, co in enumerate(ax.collections):
                q = f"{p}/coll{j}"
                arrays[f"{q}/offsets"] = np.asarray(co.get_offsets(), float)
                texts[f"{q}/style"] = {"type": type(co).__name__, "label": co.get_label()}

    @functools.wraps(orig_savefig)
    def savefig(self, fname, *args, **kwargs):
        if not is_main():
            return orig_savefig(self, fname, *args, **kwargs)
        name = os.path.basename(os.fspath(fname)) if isinstance(fname, (str, os.PathLike)) else "<buffer>"
        k = f"fig/{count['fig']:02d}:{name}"
        count["fig"] += 1
        figure_data(self, k)
        res = orig_savefig(self, fname, *args, **kwargs)
        if name != "<buffer>" and os.path.isfile(fname):
            with open(fname, "rb") as fh:
                texts[f"{k}/png_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        return res

    mfig.Figure.savefig = savefig

    # ---- ProcessPoolExecutor.map (results that come back from worker processes) -----------------------------------
    import concurrent.futures as cf
    orig_map = cf.ProcessPoolExecutor.map

    @functools.wraps(orig_map)
    def pool_map(self, fn, *iterables, **kwargs):
        res = list(orig_map(self, fn, *iterables, **kwargs))
        if is_main():
            k = f"pool/{count['pool']:02d}:{getattr(fn, '__name__', '?')}"
            count["pool"] += 1
            flatten(res, k, arrays, texts)
        return iter(res)

    cf.ProcessPoolExecutor.map = pool_map

    # ---- dump at exit (main process only) -------------------------------------------------------------------------
    def dump():
        if not is_main():
            return
        np.savez(os.path.join(out_dir, "main.npz"), **arrays)
        with open(os.path.join(out_dir, "main.json"), "w") as fh:
            json.dump(texts, fh, indent=0, default=repr)

    atexit.register(dump)


_chain_system_sitecustomize()
if os.environ.get("CAPTURE_DIR"):
    _install()
