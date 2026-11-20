"""Shared code of tests/capture_golden.py and tests/test_regression.py.

Each command (tests/commands.py) runs in its own temporary copy of the code, so committed outputs are never
overwritten, with the matplotlib Agg backend and the capture hook (tests/hook/) on PYTHONPATH. One run gives:
  stdout.txt, stderr.txt   normalized: run root -> <ROOT>, output folder -> <OUT>, wall-clock times masked
  manifest.json            exit code, SHA-256 of every captured array, texts (figure labels, solver messages),
                           SHA-256 of every file the command wrote
  arrays.npz               the arrays themselves (kept locally only: used to report differences)
Comparison is exact. A PNG whose bytes differ is a warning only (pixel identity is not required; the plotted data
are compared exactly).
"""
import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "tests" / "hook"
GOLDEN = REPO / "tests" / "golden"
LOCAL_DATA = ["zbysek_matlab_code/data/inputs_OK.mat"]      # gitignored, needed by --calibrate (K2)
sys.path.insert(0, str(HOOK))
from capture_util import digest  # noqa: E402


# ---- copies of the code -----------------------------------------------------------------------------------------------
def code_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    keep = [f for f in out if f and (f.endswith(".py") or f.startswith("tests/fixtures/"))
            and not f.startswith("tests/golden/")]
    return [f for f in keep if (REPO / f).is_file()] + [f for f in LOCAL_DATA if (REPO / f).is_file()]


def snapshot_source(base):
    """Copy the code once, at the start of a test run: later edits in the repo do not reach running commands."""
    src = Path(base) / "src"
    for f in code_files():
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / f, src / f)
    return src


def make_root(base, src):
    root = Path(tempfile.mkdtemp(prefix="run_", dir=base))
    shutil.copytree(src, root, dirs_exist_ok=True)
    return root


def snapshot(root):
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and "_capture" not in p.parts:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---- normalization ----------------------------------------------------------------------------------------------------
def normalize(text, root, masks=()):
    text = text.replace(str(root), "<ROOT>")
    text = re.sub(r"<ROOT>/outputs/\w+/", "", text)                       # a file in an output folder: bare name
    text = re.sub(r"<ROOT>/(?:outputs/\w+|docs/figures)\b", "<OUT>", text)  # the output folder itself
    for pat, rep in masks:
        text = re.sub(pat, rep, text, flags=re.M)
    return text


def normalize_err(text, root):
    text = normalize(text, root)
    return re.sub(r"<ROOT>/[^\s:\"]+\.py(\", line |:)\d+", r"<SRC>\1<L>", text)   # moved code: other path/line


# ---- one run ----------------------------------------------------------------------------------------------------------
def run_chain(cmds, variant, base, out_base, python, src):
    """Run commands that share one copy of src; returns {id: result folder}."""
    root = make_root(base, src)
    results = {}
    for cmd in cmds:
        argv = cmd.argv[variant]
        cap = root / "_capture" / cmd.id
        cap.mkdir(parents=True)
        env = dict(os.environ, MPLBACKEND="Agg", CAPTURE_DIR=str(cap),
                   PYTHONPATH=os.pathsep.join([str(root / "tests" / "hook")] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
        env.pop("CAPTURE_MAIN_PID", None)
        full = [python] + [str(root / a[1:]) if a.startswith("@") else a for a in argv]
        before = snapshot(root)
        t0 = time.time()
        try:
            p = subprocess.run(full, cwd=root / cmd.cwd, env=env, capture_output=True, text=True, timeout=cmd.timeout)
            rc, so, se = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc, so, se = "timeout", e.stdout or "", e.stderr or ""
            so, se = (so.decode() if isinstance(so, bytes) else so), (se.decode() if isinstance(se, bytes) else se)
        runtime = time.time() - t0
        after = snapshot(root)
        written = {k: v for k, v in after.items() if before.get(k) != v}
        dest = Path(out_base) / cmd.id
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        collect(cmd, variant, root, cap, rc, so, se, runtime, written, dest)
        results[cmd.id] = dest
    shutil.rmtree(root, ignore_errors=True)
    return results


def collect(cmd, variant, root, cap, rc, so, se, runtime, written, dest):
    arrays, texts = {}, {}
    for stem in ("main", "direct"):
        if (cap / f"{stem}.npz").exists():
            with np.load(cap / f"{stem}.npz") as z:
                arrays.update({f"{stem}:{k}": z[k] for k in z.files})
        if (cap / f"{stem}.json").exists():
            texts.update({f"{stem}:{k}": v for k, v in json.loads((cap / f"{stem}.json").read_text()).items()})
    workers = sorted((cap / "workers.txt").read_text().split()) if (cap / "workers.txt").exists() else []
    files, seen = {}, {}
    for rel, sha in sorted(written.items()):
        name = os.path.basename(rel)
        seen[name] = seen.get(name, 0) + 1
        key = name if seen[name] == 1 else f"{name}#{seen[name]}"
        files[key] = sha
        if name.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(root / rel)
            for c in df.columns:
                col = df[c].to_numpy()
                if col.dtype.kind in "biuf":
                    arrays[f"csv:{key}/{c}"] = col
                else:
                    texts[f"csv:{key}/{c}"] = [str(v) for v in col]
    (dest / "stdout.txt").write_text(normalize(so, root, cmd.masks))
    (dest / "stderr.txt").write_text(normalize_err(se, root))
    np.savez(dest / "arrays.npz", **arrays)
    manifest = {"id": cmd.id, "variant": variant, "argv": cmd.argv[variant], "rc": rc, "runtime_s": round(runtime, 1),
                "arrays": {k: [digest(a), a.dtype.str, list(a.shape)] for k, a in sorted(arrays.items())},
                "texts": texts, "workers": {"count": len(workers), "sha256": hashlib.sha256("\n".join(workers).encode()).hexdigest()},
                "files": files}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=repr))


def run_all(cmds, variant, base, out_base, jobs, src, python=sys.executable, log=print, exclusive=True):
    """Run commands; chains run in one copy each, exclusive commands run alone at the end (if exclusive=True)."""
    groups = {}
    for c in cmds:
        groups.setdefault(c.chain or c.id, []).append(c)
    shared = [g for g in groups.values() if not (exclusive and any(c.exclusive for c in g))]
    alone = [g for g in groups.values() if exclusive and any(c.exclusive for c in g)]
    results = {}

    def one(g):
        r = run_chain(g, variant, base, out_base, python, src)
        for c in g:
            log(f"  done {c.id:<26} {json.loads((r[c.id] / 'manifest.json').read_text())['runtime_s']:>8.1f} s")
        return r

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(one, sorted(shared, key=lambda g: -len(g))):
            results.update(r)
    for g in alone:
        results.update(one(g))
    return results


# ---- comparison -------------------------------------------------------------------------------------------------------
def stderr_blocks(text):
    """Split stderr into blocks: a warning line plus the indented source line(s) that follow it."""
    blocks = []
    for line in text.splitlines():
        if line.startswith((" ", "\t")) and blocks:
            blocks[-1] += "\n" + line
        else:
            blocks.append(line)
    return Counter(blocks)


def compare(gold, cur, stderr_unordered=False):
    """Return (failures, warnings) between two result folders."""
    fails, warns = [], []
    g = json.loads((gold / "manifest.json").read_text())
    c = json.loads((cur / "manifest.json").read_text())
    if g["rc"] != c["rc"]:
        fails.append(f"exit code {c['rc']} (golden {g['rc']})")
    for f in ("stdout.txt", "stderr.txt"):
        a, b = (gold / f).read_text(), (cur / f).read_text()
        if f == "stderr.txt" and stderr_unordered and a != b:
            ca, cb = stderr_blocks(a), stderr_blocks(b)
            if ca == cb:
                continue
            fails.append(f"stderr blocks differ (order ignored): missing {list((ca - cb).items())[:3]}, "
                         f"extra {list((cb - ca).items())[:3]}")
            continue
        if a != b:
            d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "golden", "now", n=0, lineterm=""))
            fails.append(f"{f} differs:\n    " + "\n    ".join(d[:30]) + ("\n    ..." if len(d) > 30 else ""))
    ga, ca = g["arrays"], c["arrays"]
    if set(ga) != set(ca):
        miss, extra = sorted(set(ga) - set(ca)), sorted(set(ca) - set(ga))
        fails.append(f"array keys differ: missing {miss[:8]}{'...' if len(miss) > 8 else ''}, "
                     f"extra {extra[:8]}{'...' if len(extra) > 8 else ''}")
    bad = [k for k in sorted(set(ga) & set(ca)) if ga[k] != ca[k]]
    if bad:
        fails.append(f"{len(bad)} arrays differ: " + "; ".join(array_diff(gold, cur, k, ga[k], ca[k]) for k in bad[:10]))
    if g["texts"] != c["texts"]:
        keys = sorted(k for k in set(g["texts"]) | set(c["texts"]) if g["texts"].get(k) != c["texts"].get(k))
        png = [k for k in keys if k.endswith("png_sha256")]
        other = [k for k in keys if k not in png]
        if other:
            fails.append(f"{len(other)} texts differ: " + "; ".join(
                f"{k}: golden {g['texts'].get(k)!r} now {c['texts'].get(k)!r}" for k in other[:6]))
        if png:
            warns.append(f"{len(png)} PNG files differ in bytes (plotted data compared separately)")
    if g["workers"] != c["workers"]:
        fails.append(f"worker solve_ivp results differ: golden {g['workers']} now {c['workers']}")
    gf, cf = g["files"], c["files"]
    if set(gf) != set(cf):
        fails.append(f"written files differ: golden {sorted(gf)} now {sorted(cf)}")
    for k in sorted(set(gf) & set(cf)):
        if gf[k] != cf[k]:
            (warns if k.endswith(".png") else fails).append(f"file {k}: bytes differ")
    return fails, warns


def array_diff(gold, cur, key, gm, cm):
    msg = f"{key} (golden {gm[1]}{gm[2]}, now {cm[1]}{cm[2]})"
    try:
        with np.load(gold / "arrays.npz") as zg, np.load(cur / "arrays.npz") as zc:
            a, b = zg[key].astype(float), zc[key].astype(float)
        if a.shape == b.shape:
            d = np.abs(a - b)
            rel = d / np.maximum(np.abs(a), np.finfo(float).tiny)
            msg += f": max abs diff {np.nanmax(d):.3g}, max rel diff {np.nanmax(rel):.3g}"
    except (FileNotFoundError, KeyError, ValueError) as e:
        msg += f": arrays not available locally ({type(e).__name__})"
    return msg


# ---- which commands can a change affect -------------------------------------------------------------------------------
def deps(cmd, variant):
    """Files a command depends on: its scripts and everything they import from the repo (transitively)."""
    todo = [REPO / a[1:] for a in cmd.argv[variant] if a.startswith("@") and a.endswith(".py")]
    todo += [REPO / d for d in cmd.deps]
    seen = set()
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    pkg = f.parent
                    for _ in range(node.level - 1):
                        pkg = pkg.parent
                    base = ".".join(pkg.relative_to(REPO).parts)
                    mod = f"{base}.{node.module}" if node.module else base
                else:
                    mod = node.module
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in
                  ("run_module", "import_module") and node.args and isinstance(node.args[0], ast.Constant)):
                names = [str(node.args[0].value)]                 # runpy.run_module("experiments.x") etc.
            for n in names:
                for d in (f.parent, REPO, REPO / "mat_models"):
                    for cand in (d / (n.replace(".", "/") + ".py"), d / n.replace(".", "/") / "__init__.py"):
                        if cand.is_file():
                            todo.append(cand)
    return {str(p.relative_to(REPO)) for p in seen}


def changed_since(rev):
    out = subprocess.run(["git", "diff", "--name-only", rev], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    out += subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout.split()
    return set(out)
