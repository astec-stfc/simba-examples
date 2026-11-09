"""
21 - Optimise a setting with scipy
==================================

Shows: wrapping track() in a function and giving it to an optimiser. The
optimiser (scipy.optimize.minimize_scalar) chooses quadrupole Q's k1l to
make the beam smallest in x on a screen downstream. Each call sets k1l
(modifyElement), tracks and reads the screen's beam back. Every trial is
logged to a CSV, and the best setting is saved with save_changes_file, so
it can be applied later with load_changes_file (as in 15).

Run:    python 04_scans_optimisation/21_optimise_scipy.py
Tier:   A
Needs:  SIMBA, LAURA and Ocelot.
Output: output/21_optimise_scipy/: the lattice, run/, trials.csv and
        best.yaml (the changes file).

Units: k1l in 1/m; sizes in m.

The line is 20's: a 0.1 m quad 0.5 m from the start, and a screen L = 3 m
beyond its centre, with a 100 MeV beam, beta = 10 m and alpha = 0 at the
start. For a thin lens the smallest sigma_x is at k = 1/L + S12/S11, with
S the sigma matrix at the quad (k = 1/L - alpha/beta there).

Check: the optimiser's k1l must equal the optimum of the thick-quad matrix
model (02_tracking/simple_lattice, applied to the starting beam's sigma
matrix) to 1e-4 relative. The two differ only by the second-order
chromatic terms the matrices leave out, and by the optimiser's tolerance
(xatol 1e-6); measured 3e-6, after 10 tracks. The thin-lens value is
printed: it is off by 0.6% (the quad's length). The changes file must hold
the best k1l.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
import yaml
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
from simple_lattice import make_beam, marker, quad, sigma_matrix, transfer_matrix, write_master_lattice  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
Z_QUAD, L, QUAD_LENGTH = 0.5, 3.0, 0.1
Z_SCREEN = Z_QUAD + L
BOUNDS = (0.0, 1.0)

files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "SCREEN"},
    }
}
elements = [marker("START", 0.0), quad("Q", Z_QUAD, 0.0, QUAD_LENGTH), marker("SCREEN", Z_SCREEN)]
settings = write_master_lattice(LATTICE, elements, files, "optimise.def")
start = make_beam(LATTICE / "START.openpmd.hdf5", beta=(10.0, 10.0), alpha=(0.0, 0.0))

framework = fw.Framework(master_lattice=str(LATTICE), directory=str(OUT / "run"), clean=True, verbose=False)
framework.loadSettings(settings)
trials = []


def sigma_x_at_screen(k1l):
    """The objective: set Q, track, return sigma_x on the screen."""
    framework.modifyElement("Q", "k1l", float(k1l))
    framework.track()
    size = float(np.std(rbf.load_file(str(OUT / "run" / "SCREEN.openpmd.hdf5")).x.val))
    trials.append((float(k1l), size))
    return size


result = minimize_scalar(sigma_x_at_screen, bounds=BOUNDS, method="bounded", options={"xatol": 1e-6})
best = float(result.x)
with open(OUT / "trials.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k1l", "sigma_x"])
    writer.writerows(trials)

# The same optimum from the matrix model
s0 = sigma_matrix(start, "x")


def model_size2(k1l):
    m = transfer_matrix([quad("Q", Z_QUAD, k1l, QUAD_LENGTH)], "x", 0.0, Z_SCREEN)
    return (m @ s0 @ m.T)[0, 0]


model = float(minimize_scalar(model_size2, bounds=BOUNDS, method="bounded", options={"xatol": 1e-9}).x)
drift = np.array([[1.0, Z_QUAD], [0.0, 1.0]])
s_quad = drift @ s0 @ drift.T
thin = 1 / L + s_quad[0, 1] / s_quad[0, 0]

# Leave the best setting in place, then save it
framework.modifyElement("Q", "k1l", best)
changes_file = OUT / "best.yaml"
framework.save_changes_file(filename=str(changes_file))
saved = yaml.safe_load(changes_file.read_text())["Q"]

print(f"optimiser:  k1l = {best:.6f} 1/m after {len(trials)} tracks, sigma_x = {result.fun * 1e6:.3f} um")
print(f"matrices:   k1l = {model:.6f} 1/m   (relative difference {best / model - 1:.1e})")
print(f"thin lens:  k1l = {thin:.6f} 1/m   (relative difference {thin / model - 1:.1e})")
print(f"\n{changes_file.name}:\n" + yaml.safe_dump({"Q": saved}))

# --- Check -----------------------------------------------------------------
assert result.success, result.message
assert abs(best / model - 1) < 1e-4, f"optimiser {best}, matrix model {model}"
saved_k1l = saved["magnetic"]["multipoles"]["K1L"]["normal"]["new"]
assert np.isclose(saved_k1l, best, rtol=1e-12), f"changes file has {saved_k1l}, best is {best}"
print("OK: the optimiser finds the matrix model's optimum, and the changes file holds it")
