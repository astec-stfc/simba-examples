"""
14 - Compare codes on a real machine
====================================

Shows: the JFEL linac (13's line, from the saved injector beam to the
undulator entrance) tracked in several codes from the same beam, compared at
every screen they have in common, with a plot of the beam along s in each
code. It's a cut-down version of JFEL's Scripts/compare_codes.py.

Run:    python 02_tracking/14_compare_codes.py [--codes ocelot cheetah madx ...]
Tier:   A (adds elegant if a container runtime with the simcodes image is there)
Needs:  SIMBA, LAURA, laura-lattices, Ocelot, Cheetah, cpymad (MAD-X).
Output: output/14_compare_codes/<code>/ (each run) and
        output/14_compare_codes/compare.png.

Units: E in MeV; dE (rms energy spread) in %; emittance (normalised) and
sizes in um.

Collective effects are off in every code, as in 13: the codes don't all
model the same ones.

Where the codes differ (JFEL README, "Code comparison"):
- elegant (and Xsuite) compress the bunch ~6% more than the others: a
  different RF model gives a slightly different chirp into the chicane.
- Ocelot (and Xsuite, Bmad) grow eps_ny by ~2% in the chicane, from how the
  dipole edges are treated in y.
- sigma_y differs by several % around S11, where the beam has a ~25 um
  vertical waist and a few um is a large fraction.

Check: for each pair of codes, the largest disagreement anywhere along the
line in each quantity must be below the ceiling in SIMBA's own cross-code
test (unit_tests/test_jfel_code_comparison.py). That test runs the chicane
at 0.1 rad; this .def runs it at 0.13 rad, near full compression, so three
ceilings are re-measured here by the test's own rule (see TOLERANCES).
Relative quantities are in % of the pair's mean; dE is in percentage points.
Xsuite (--codes ... xsuite) is in neither table, so its pairs are printed
but not checked. At least 57 screens must be in common, the minimum SIMBA's
test asks for, so the comparison can't quietly shrink.
"""

import argparse
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import JFEL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
JFEL_DIR = os.path.dirname(JFEL.__file__)
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"
SUFFIX = ".openpmd.hdf5"

# label: (function of the beam, compared relatively)
QUANTITIES = {
    "E": (lambda b: 1e-6 * np.mean(b.cp.val), True),
    "dE": (lambda b: 100 * np.std(b.cp.val) / np.mean(b.cp.val), False),
    "enx": (lambda b: 1e6 * b.Particles.emittance.enx.val, True),
    "eny": (lambda b: 1e6 * b.Particles.emittance.eny.val, True),
    "sx": (lambda b: 1e6 * b.Particles.sigmas.sigma_x.val, True),
    "sy": (lambda b: 1e6 * b.Particles.sigmas.sigma_y.val, True),
    "sz": (lambda b: 1e6 * b.Particles.sigmas.sigma_z.val, True),
}
# Ceilings (E, dE, enx, eny, sx, sy, sz): SIMBA's test_jfel_code_comparison.py
# (1.4x what it measured, with the chicane at 0.1 rad). Here the chicane is at
# the lattice's 0.13 rad, near full compression; three quantities measure more
# than the test's ceilings, so they get the test's rule, 1.4x the value measured
# here (2026-11-08, SIMBA main 26aed74):
#   madx/ocelot sy 3.96 and cheetah/ocelot sy 3.79: the ~25 um waist at S11;
#   elegant/ocelot sz 6.45: elegant compresses ~6% more (see above).
#                          E      dE     enx    eny    sx    sy    sz
TOLERANCES = {
    ("madx", "ocelot"):    (0.19, 0.087, 9.4,  5.6,  8.7,  5.6, 17.0),
    ("cheetah", "madx"):   (0.02, 0.01,  1.0,  2.0,  1.0,  1.0,  1.0),
    ("cheetah", "ocelot"): (0.19, 0.087, 9.5,  5.9,  8.7,  5.3, 17.0),
    ("cheetah", "elegant"): (0.16, 0.11, 11.0, 10.0, 11.0,  8.9, 20.0),
    ("elegant", "madx"):   (0.16, 0.11, 11.0,  9.6, 11.0,  9.1, 21.0),
    ("elegant", "ocelot"): (0.22, 0.026, 2.8,  6.7,  3.9,  7.8,  9.0),
}
MIN_SCREENS = 57


def docker_ready():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


parser = argparse.ArgumentParser()
parser.add_argument("--codes", nargs="+", default=["ocelot", "cheetah", "madx"] + (["elegant"] if docker_ready() else []))
codes = parser.parse_args().codes


def track(code):
    """Track the Linac line in `code`; return {screen: {"s": s, quantity: value}}."""
    directory = OUT / code
    framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(directory), clean=True, verbose=False,
                             container_runtime="docker" if code == "elegant" else None)
    framework.loadSettings("Lattices/jfel_linac_codes.def")
    framework.change_Lattice_Code("Linac", code)
    line = framework["Linac"]
    line.csr_enable = line.lsc_enable = line.wakefield_enable = False
    framework.track()
    stats = {}
    for f in sorted(directory.glob(f"*{SUFFIX}")):
        beam = rbf.load_file(str(f))
        row = {"s": float(np.atleast_1d(beam.s)[0])}
        row.update({k: float(fn(beam)) for k, (fn, _) in QUANTITIES.items()})
        stats[f.name[: -len(SUFFIX)]] = row
    return stats


def disagreement(a, b, relative):
    if not relative:
        return abs(a - b)
    return 100 * abs(a - b) / abs(0.5 * (a + b))


results = {code: track(code) for code in codes}
common = sorted(set.intersection(*(set(r) for r in results.values())), key=lambda n: results[codes[0]][n]["s"])
end = common[-1]
print(f"\n{len(common)} screens common to {', '.join(codes)}. At {end}:")
print(f"  {'':8s}" + "".join(f"{k:>9s}" for k in QUANTITIES))
for code in codes:
    print(f"  {code:8s}" + "".join(f"{results[code][end][k]:9.4g}" for k in QUANTITIES))

print("\nWorst disagreement along the line (% of the mean; dE in percentage points) / ceiling:")
print(f"  {'':18s}" + "".join(f"{k:>13s}" for k in QUANTITIES))
problems = []
for a, b in itertools.combinations(codes, 2):
    pair = (a, b) if (a, b) in TOLERANCES else (b, a)
    ceilings = TOLERANCES.get(pair)
    row = f"  {a + ' vs ' + b:18s}"
    for i, (k, (_, relative)) in enumerate(QUANTITIES.items()):
        worst = max(disagreement(results[a][n][k], results[b][n][k], relative) for n in common)
        row += f"{worst:8.3f}/{ceilings[i]:<4g}" if ceilings else f"{worst:8.3f}/-   "
        if ceilings and worst > ceilings[i]:
            problems.append(f"{a} vs {b}: {k} {worst:.3f} > {ceilings[i]}")
    print(row)

fig, axes = plt.subplots(len(QUANTITIES), 1, sharex=True, figsize=(8, 2 * len(QUANTITIES)))
for code in codes:
    s = [results[code][n]["s"] for n in common]
    for ax, k in zip(axes, QUANTITIES):
        ax.plot(s, [results[code][n][k] for n in common], marker=".", ms=3, label=code)
        ax.set_ylabel(k)
axes[0].legend(ncol=len(codes), fontsize="small")
axes[-1].set_xlabel("s [m]")
fig.tight_layout()
fig.savefig(OUT / "compare.png", dpi=120)
print(f"\nPlot: {OUT / 'compare.png'}")

# --- Check -----------------------------------------------------------------
assert len(common) >= MIN_SCREENS, f"only {len(common)} screens in common, expected >= {MIN_SCREENS}"
assert not problems, "Codes disagree by more than the ceilings:\n  " + "\n  ".join(problems)
print(f"\nOK: {', '.join(codes)} agree within the ceilings at all {len(common)} common screens")
