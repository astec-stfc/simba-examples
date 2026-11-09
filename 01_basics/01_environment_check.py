"""
01 - Environment check
======================

Shows: which versions of SIMBA, LAURA and laura-lattices are installed (and
from where), which tracking codes Python can find, and whether a container
runtime is available for the compiled codes (elegant, ASTRA, OPAL, ...).

Run:    python 01_basics/01_environment_check.py
Tier:   A
Needs:  nothing beyond the install in README.md.
Output: a table on screen. Exits non-zero, with a hint, if SIMBA, LAURA or
        laura-lattices is missing or older than these examples need.
        Missing tracking codes and container runtimes are reported, not
        failures: they only decide which tier of examples you can run.
"""

import importlib.metadata as md
import importlib.util
import platform
import shutil
import subprocess
import sys

from packaging.version import Version

# Oldest releases these examples are tested against (SIMBA main 26aed74 is 0.7.1.post5).
REQUIRED = {
    "simba-accelerator": "0.7.1.post5",
    "laura-accelerator": "1.1.5",
    "laura-lattices": "1.0.0",
}
# Python tracking codes, by tier (see README.md). Module name -> what it is.
PYTHON_CODES = {
    "A": {"ocelot": "Ocelot", "cheetah": "Cheetah", "xtrack": "Xsuite", "cpymad": "MAD-X (cpymad)",
          "wake_t": "Wake-T"},
    "C": {"impactx": "ImpactX", "pytao": "Bmad/Tao (needs libtao)",
          "fbpic": "FBPIC", "pybdsim": "BDSIM (needs the bdsim binary)"},
}
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def source_of(module):
    """Where a package is imported from, plus the git commit if it is a checkout."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return "not importable"
    where = spec.origin.rsplit("/", 2)[0]
    try:
        commit = subprocess.run(["git", "-C", where, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except OSError:
        commit = ""
    return f"{where}" + (f" (git {commit})" if commit else "")


print(f"Python {platform.python_version()} at {sys.executable}\n")

problems = []
for dist, module in [("simba-accelerator", "simba"), ("laura-accelerator", "laura"),
                     ("laura-lattices", "laura_lattices")]:
    try:
        version = md.version(dist)
    except md.PackageNotFoundError:
        problems.append(f"{dist} is not installed")
        print(f"  {dist:20s} MISSING")
        continue
    ok = Version(version) >= Version(REQUIRED[dist])
    if not ok:
        problems.append(f"{dist} {version} is older than {REQUIRED[dist]}")
    print(f"  {dist:20s} {version:14s} {'ok ' if ok else 'OLD'} {source_of(module)}")

if importlib.util.find_spec("laura_lattices"):
    import laura_lattices

    print(f"\n  laura_lattices machines: {', '.join(laura_lattices.MACHINES)}")
    if "JFEL" not in laura_lattices.MACHINES:
        problems.append("laura_lattices has no JFEL: install the public laura-lattices")

print("\nPython tracking codes (installed = importable without running them):")
for tier, codes in PYTHON_CODES.items():
    for module, name in codes.items():
        found = importlib.util.find_spec(module) is not None
        print(f"  [{tier}] {name:32s} {'installed' if found else '-'}")

print("\nContainer runtimes for compiled codes (tier B):")
docker = shutil.which("docker") is not None
docker_running = docker and run(["docker", "info"])
image = docker_running and run(["docker", "image", "inspect", IMAGE])
apptainer = shutil.which("apptainer") is not None
print(f"  docker      {'found' if docker else '-'}"
      f"{', daemon running' if docker_running else (', daemon not reachable' if docker else '')}"
      f"{', image present' if image else ''}")
if docker_running and not image:
    print(f"              image missing: docker pull {IMAGE}  (~17 GB; SIMBA would pull it on first use)")
print(f"  apptainer   {'found' if apptainer else '-'}")
tier_b = image or apptainer
print(f"\nTier A (pip only): {'ready' if not problems else 'not ready'}"
      f"   Tier B (containers): {'ready' if tier_b else 'not available'}")

if problems:
    print("\nFAILED:")
    for p in problems:
        print(f"  - {p}")
    print("Fix: follow the install steps in README.md (editable installs of SIMBA main,"
          " LAURA main and the public laura-lattices).")
    sys.exit(1)
print("\nOK")
