"""
20 - Measure the emittance with a quadrupole scan
=================================================

Shows: a quad scan, the standard way to measure emittance. Vary one
quadrupole's strength (modifyElement), record the beam size on a screen
downstream after each track(), and fit. The fit gives the beam's sigma
matrix at the quad, and so its emittance (and Twiss parameters).

Run:    python 04_scans_optimisation/20_quad_scan_emittance.py
Tier:   A
Needs:  SIMBA, LAURA and Ocelot.
Output: output/20_quad_scan_emittance/: the lattice, run/ (the last scan
        point) and quad_scan.png.

Units: k1l in 1/m; sizes in m; emittance (geometric) in m rad.

The line: a 0.1 m quadrupole Q centred 0.5 m from the start, then a screen
L = 3 m beyond its centre. The beam: 100 MeV, beta = 10 m and alpha = 0 at
the start. One scan gives both planes: Q focuses in x (sigma_x passes
through a minimum) and defocuses in y.

Two fits:
- The textbook one treats Q as a thin lens, M = [[1 - k L, L], [-k, 1]]
  with k = k1l. Then sigma^2 = A k^2 + B k + C, a parabola, and the sigma
  matrix at the quad is S11 = A / L^2, S12 = -(B + 2 L S11) / (2 L^2),
  S22 = (C - S11 - 2 L S12) / L^2, with emittance sqrt(S11 S22 - S12^2).
- The exact one uses the thick-quad matrices from the start to the screen
  (02_tracking/simple_lattice.transfer_matrix). sigma^2 is linear in the
  three elements of the starting sigma matrix, so a linear least-squares
  fit gives them.

Check: the exact fit's emittance must equal the starting beam's own
(sqrt(det Sigma0), from its particles) to 1e-3 in both planes. Tracking has
no noise here, and the second-order chromatic terms the matrices leave out
are ~1e-5 (measured 5e-6). The thin-lens fit must be within 5% in x, where
the scan passes through the waist: its error comes from the quad's length
(measured -1.1%). In y it is only printed: sigma_y grows over the whole
scan, the parabola's minimum is far outside it, and the fit is
ill-conditioned. The same ~1% model error then becomes +43% (fed synthetic
thin-lens data instead, both planes come back exact). The lesson for a real
scan: span the waist. A real measurement also adds the screen's noise and
resolution, which this leaves out.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
from simple_lattice import make_beam, marker, quad, sigma_matrix, transfer_matrix, write_master_lattice  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
Z_QUAD, L, QUAD_LENGTH = 0.5, 3.0, 0.1
Z_SCREEN = Z_QUAD + L

files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "SCREEN"},
    }
}
elements = [marker("START", 0.0), quad("Q", Z_QUAD, 0.0, QUAD_LENGTH), marker("SCREEN", Z_SCREEN)]
settings = write_master_lattice(LATTICE, elements, files, "scan.def")
start = make_beam(LATTICE / "START.openpmd.hdf5", beta=(10.0, 10.0), alpha=(0.0, 0.0))

framework = fw.Framework(master_lattice=str(LATTICE), directory=str(OUT / "run"), clean=True, verbose=False)
framework.loadSettings(settings)

strengths = np.linspace(0.1, 0.6, 11)
size2 = {"x": [], "y": []}
for k1l in strengths:
    framework.modifyElement("Q", "k1l", float(k1l))
    framework.track()
    screen = rbf.load_file(str(OUT / "run" / "SCREEN.openpmd.hdf5"))
    for plane in ("x", "y"):
        size2[plane].append(np.var(getattr(screen, plane).val))


def thin_lens_fit(k, s2, sign):
    """Emittance from the parabola sigma^2(k); sign -1 for y, where the quad's k is -k1l."""
    a, b, c = np.polyfit(sign * k, s2, 2)
    s11 = a / L ** 2
    s12 = -(b + 2 * L * s11) / (2 * L ** 2)
    s22 = (c - s11 - 2 * L * s12) / L ** 2
    return np.sqrt(s11 * s22 - s12 ** 2)


def exact_fit(k, s2, plane):
    """Sigma matrix at the start from the thick-quad matrices; returns its emittance."""
    rows = []
    for k1l in k:
        m = transfer_matrix([quad("Q", Z_QUAD, k1l, QUAD_LENGTH)], plane, 0.0, Z_SCREEN)
        rows.append([m[0, 0] ** 2, 2 * m[0, 0] * m[0, 1], m[0, 1] ** 2])
    s11, s12, s22 = np.linalg.lstsq(np.array(rows), s2, rcond=None)[0]
    return np.sqrt(s11 * s22 - s12 ** 2)


print(f"{'plane':5s} {'beam eps [nm]':>13s} {'exact fit':>10s} {'rel diff':>9s} {'thin lens':>10s} {'rel diff':>9s}")
problems = []
for plane, sign in (("x", 1), ("y", -1)):
    s2 = np.array(size2[plane])
    eps = np.sqrt(np.linalg.det(sigma_matrix(start, plane)))
    exact, thin = exact_fit(strengths, s2, plane), thin_lens_fit(strengths, s2, sign)
    print(f"{plane:5s} {eps * 1e9:13.4f} {exact * 1e9:10.4f} {exact / eps - 1:9.1e} {thin * 1e9:10.4f} {thin / eps - 1:9.1e}")
    if abs(exact / eps - 1) > 1e-3:
        problems.append(f"{plane}: exact fit {exact:.4e}, beam {eps:.4e}")
    if plane == "x" and abs(thin / eps - 1) > 0.05:
        problems.append(f"{plane}: thin-lens fit {thin:.4e}, beam {eps:.4e}")
print("(y's thin-lens fit is ill-conditioned: the scan does not pass through a y waist)")

for plane in ("x", "y"):
    plt.plot(strengths, np.sqrt(size2[plane]) * 1e6, "o-", label=f"sigma_{plane}")
plt.xlabel("Q k1l [1/m]")
plt.ylabel("beam size at the screen [um]")
plt.legend()
plt.savefig(OUT / "quad_scan.png", dpi=120, bbox_inches="tight")
print(f"\nPlot: {OUT / 'quad_scan.png'}")

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print("\nOK: the quad scan gives back the beam's emittance in both planes")
