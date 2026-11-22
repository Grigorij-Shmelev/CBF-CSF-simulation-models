"""Regression test: re-run the commands and compare with the golden outputs (tests/golden/), exactly.

    python tests/test_regression.py                     all commands, the new form of each command
    python tests/test_regression.py --variant both      also the old commands (mat_models/ wrappers)
    python tests/test_regression.py --affected HEAD     only commands whose code changed since HEAD
    python tests/test_regression.py --only new.default,vanek.lit
    python tests/test_regression.py --affected HEAD --skip new.all --no-exclusive     quick check after a commit

Exit code 0 = everything identical. On a difference the report gives the max absolute and relative difference
of each differing array (needs the local tests/golden/*/arrays.npz, which are not committed).
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402
from commands import COMMANDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("new", "old", "both"), default="new",
                    help="new = command of the new layout (old if a command has no new form yet); old = the old "
                    "command (new if a command was added after the refactor and has no old form)")
    ap.add_argument("--only", help="comma-separated command ids")
    ap.add_argument("--affected", metavar="REV", help="only commands whose dependencies changed since REV")
    ap.add_argument("--skip", help="comma-separated command ids to leave out")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tmp", help="folder for the temporary copies (default: system temp)")
    ap.add_argument("--keep", action="store_true", help="keep the run folders")
    ap.add_argument("--no-exclusive", action="store_true", help="run --fit together with the others (faster, runtime "
                    "not representative)")
    a = ap.parse_args()

    cmds = [c for c in COMMANDS if (not a.only or c.id in a.only.split(","))
            and (not a.skip or c.id not in a.skip.split(","))]
    if a.affected:
        changed = harness.changed_since(a.affected)
        harness_files = ("tests/hook/", "tests/harness.py", "tests/direct.py", "tests/commands.py", "tests/fixtures/")
        if any(f.startswith(harness_files) for f in changed):
            print("test harness changed: running everything selected")
        else:
            keep = []
            for c in cmds:
                d = set().union(*(harness.deps(c, v) for v in c.argv))
                if d & changed:
                    keep.append(c)
            chains = {c.chain for c in keep if c.chain}
            cmds = [c for c in cmds if c in keep or (c.chain and c.chain in chains)]
        print(f"changed since {a.affected}: {sorted(changed)}")
    variants = {}
    for c in cmds:
        vs = ["old", "new"] if a.variant == "both" else [a.variant]
        variants[c.id] = [v for v in vs if v in c.argv] or [v for v in ("old", "new") if v in c.argv][:1]
    base = Path(tempfile.mkdtemp(prefix="regr_", dir=a.tmp))
    src = harness.snapshot_source(base)
    print(f"{len(cmds)} commands: {', '.join(c.id for c in cmds)}")
    t0 = time.time()
    results = {}
    for v in ("old", "new"):
        sel = [c for c in cmds if v in variants[c.id]]
        if sel:
            results[v] = harness.run_all(sel, v, base, base / v, a.jobs, src, exclusive=not a.no_exclusive)
    n_fail = 0
    print(f"\n{'command':<26}{'variant':<8}{'runtime':>9}  result")
    for c in cmds:
        for v in variants[c.id]:
            cur = results[v][c.id]
            fails, warns = harness.compare(harness.GOLDEN / c.id, cur, c.stderr_unordered)
            rt = json.loads((cur / "manifest.json").read_text())["runtime_s"]
            print(f"{c.id:<26}{v:<8}{rt:>8.1f}s  {'IDENTICAL' if not fails else 'FAIL'}"
                  + "".join(f"\n      FAIL {f}" for f in fails) + "".join(f"\n      warn {w}" for w in warns))
            n_fail += bool(fails)
    print(f"\n{n_fail} failing, wall time {time.time() - t0:.0f} s")
    if a.keep or n_fail:
        print(f"run folders kept in {base}")
    else:
        shutil.rmtree(base, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
