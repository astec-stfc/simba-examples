"""
Run every example script, each in a fresh Python process, check that the
notebooks hold the current scripts (sync_notebooks.py --check), and run the
group notebooks; print a summary table.

    python check_all.py                 # tiers A and B (B is skipped without a container runtime)
    python check_all.py --tiers A       # pip-only examples
    python check_all.py --tiers A,B,C   # everything, including the specialist codes
    python check_all.py 01_basics/05_generate_beam.py   # just these
    python check_all.py --no-notebooks

A script passes when it exits with 0 (its physics checks are asserts). Lines
it prints starting with XFAIL (a known bug, see ISSUES.md) or XPASS (a known
bug that no longer shows) are counted in the table. Logs are written to
output/check_all/.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "output" / "check_all"
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"


def tier(script):
    match = re.search(r"^Tier:\s*(\S+)", script.read_text(), re.M)
    return match.group(1) if match else "?"


def containers():
    if shutil.which("apptainer"):
        return True
    if shutil.which("docker"):
        return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scripts", nargs="*", type=Path, help="scripts to run (default: all)")
    parser.add_argument("--tiers", default="A,B")
    parser.add_argument("--no-notebooks", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per script")
    args = parser.parse_args()

    tiers = set(args.tiers.upper().split(","))
    scripts = [p.resolve() for p in args.scripts] or sorted(ROOT.glob("[0-9][0-9]_*/[0-9][0-9]_*.py"))
    notebooks = [] if args.no_notebooks or args.scripts else sorted(ROOT.glob("[0-9][0-9]_*/*.ipynb"))
    have_containers = containers()
    LOGS.mkdir(parents=True, exist_ok=True)

    rows, failed = [], False
    for script in scripts:
        name, t = script.relative_to(ROOT).as_posix(), tier(script)
        if t[0] not in tiers:
            rows.append((name, t, "skip", f"tier {t} not selected", 0))
            continue
        if "B" in t and not have_containers:
            rows.append((name, t, "skip", "no container runtime or image", 0))
            continue
        start = time.time()
        try:
            run = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True,
                                 timeout=args.timeout)
            output, code = run.stdout + run.stderr, run.returncode
        except subprocess.TimeoutExpired as e:
            output, code = f"{e.stdout or ''}{e.stderr or ''}\nTIMEOUT after {args.timeout} s", -1
        took = time.time() - start
        (LOGS / f"{script.stem}.log").write_text(output)
        notes = [f"{len(re.findall(rf'^{w}', output, re.M))} {w}" for w in ("XFAIL", "XPASS")]
        notes = ", ".join(n for n in notes if not n.startswith("0 "))
        if code:
            failed = True
            last = [line for line in output.strip().splitlines() if line.strip()][-1:] or [""]
            notes = (notes + "; " if notes else "") + last[0][:100]
        rows.append((name, t, "FAIL" if code else "pass", notes, took))
        print(f"{rows[-1][2]:4s} {name} ({took:.0f} s)", flush=True)

    if notebooks:
        sync = subprocess.run([sys.executable, str(ROOT / "sync_notebooks.py"), "--check"], capture_output=True, text=True)
        failed |= bool(sync.returncode)
        rows.append(("sync_notebooks.py --check", "-", "FAIL" if sync.returncode else "pass", sync.stdout.strip()[:100], 0))
    for nb in notebooks:
        name = nb.relative_to(ROOT).as_posix()
        start = time.time()
        # "-m nbconvert", not "-m jupyter nbconvert": the latter finds jupyter-nbconvert on PATH,
        # which may be another Python's, and that one starts its own kernel.
        run = subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                              "--ExecutePreprocessor.timeout=3600", "--output-dir", str(LOGS), str(nb)],
                             capture_output=True, text=True)
        took = time.time() - start
        (LOGS / f"{nb.stem}.nbconvert.log").write_text(run.stdout + run.stderr)
        note = run.stderr.strip().splitlines()[-1][:100] if run.returncode else ""
        bad = bool(run.returncode)
        failed |= bad
        rows.append((name, "-", "FAIL" if bad else "pass", note, took))
        print(f"{rows[-1][2]:4s} {name} ({took:.0f} s)", flush=True)

    width = max(len(r[0]) for r in rows) if rows else 10
    print(f"\n{'example':{width}s}  tier  result  time   notes")
    for name, t, result, notes, took in rows:
        print(f"{name:{width}s}  {t:4s}  {result:6s}  {took:5.0f}s  {notes}")
    print(f"\nLogs in {LOGS}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
