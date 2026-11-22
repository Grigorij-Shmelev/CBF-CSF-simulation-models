"""Capture the golden outputs of every command in tests/commands.py (run once, before the refactor).

    python tests/capture_golden.py [--only id,id] [--jobs 12] [--tmp DIR] [--make-fixture]

Every command runs twice, in separate temporary copies of the code; the two runs must agree exactly (otherwise the
golden outputs would not be reproducible and nothing is written). Run A then becomes tests/golden/<id>/.
Commands marked exclusive (--fit) run alone after the others, so that their runtime is honest.
--make-fixture writes tests/fixtures/lit_synthetic.csv (the input of --fit-csv) with the current code first.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
from commands import COMMANDS  # noqa: E402

FIXTURE = harness.REPO / "tests" / "fixtures" / "lit_synthetic.csv"


def make_fixture():
    """Synthetic LIT recording: lumbar pressure of the model for LIT_TRUTH, sampled at 2 Hz, noise 0.3 mmHg."""
    sys.path.insert(0, str(harness.REPO / "mat_models"))
    import new_model as nm
    proto = nm.lit_protocol()
    t = np.arange(0.0, proto["t_end"], 0.5)
    p = nm.lit_curve(nm.LIT_TRUTH, t, proto) + 0.3 * np.random.default_rng(11).standard_normal(len(t))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(FIXTURE, np.column_stack([t, p]), delimiter=",", header="time_s,pressure_mmHg", comments="", fmt="%.4f")
    print(f"wrote {FIXTURE.relative_to(harness.REPO)} ({len(t)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated command ids")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tmp", help="folder for the temporary copies (default: system temp)")
    ap.add_argument("--make-fixture", action="store_true")
    ap.add_argument("--reuse", metavar="DIR", help="only compare and write: runs A and B already in DIR")
    a = ap.parse_args()
    if a.make_fixture:
        make_fixture()
    cmds = [c for c in COMMANDS if not a.only or c.id in a.only.split(",")]
    if a.reuse:
        base = Path(a.reuse)
        return check_and_write(cmds, base)
    base = Path(tempfile.mkdtemp(prefix="golden_", dir=a.tmp))
    src = harness.snapshot_source(base)
    groups = {}
    for c in cmds:
        groups.setdefault(c.chain or c.id, []).append(c)
    shared = [(g, run) for g in groups.values() if not any(c.exclusive for c in g) for run in "AB"]
    alone = [(g, run) for run in "AB" for g in groups.values() if any(c.exclusive for c in g)]
    t0 = time.time()

    def one(job):
        g, run = job
        variant = "old" if all("old" in c.argv for c in g) else "new"    # commands added later have only the new form
        res = harness.run_chain(g, variant, base, base / run, sys.executable, src)
        for c in g:
            print(f"  run {run} {c.id:<26} done after {time.time() - t0:7.0f} s", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(one, sorted(shared, key=lambda j: -len(j[0]))))
    for job in alone:
        one(job)
    return check_and_write(cmds, base)


def check_and_write(cmds, base):
    print("\nDeterminism check (run A vs run B):")
    bad = 0
    for c in cmds:
        fails, warns = harness.compare(base / "A" / c.id, base / "B" / c.id, c.stderr_unordered)
        rt = [json.loads((base / r / c.id / "manifest.json").read_text())["runtime_s"] for r in "AB"]
        print(f"  {c.id:<26} {'OK' if not fails else 'DIFFERS':<8} runtime {rt[0]:7.1f} / {rt[1]:7.1f} s"
              + "".join(f"\n      FAIL {f}" for f in fails) + "".join(f"\n      warn {w}" for w in warns))
        bad += bool(fails)
    if bad:
        print(f"\n{bad} commands are not reproducible: golden outputs NOT written. Runs kept in {base}")
        return 1
    for c in cmds:
        dest = harness.GOLDEN / c.id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(base / "A" / c.id, dest)
    shutil.rmtree(base, ignore_errors=True)
    print(f"\nGolden outputs written to {harness.GOLDEN.relative_to(harness.REPO)}/ ({len(cmds)} commands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
