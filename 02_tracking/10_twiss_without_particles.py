"""
10 - Periodic optics of a FODO cell
===================================

Shows: asking a code for the matched (periodic) optics of a line rather than
tracking a beam's own Twiss parameters through it. Set
`tracking: {periodic: true}` in the line's `files:` entry. After track(),
`framework["LINE"].ring_parameters()` gives the tune and the periodic Twiss
parameters at the start of the line, from the code's one-turn (here
one-cell) map. The Twiss parameters along the line are in the run's Twiss
summary, read with load_directory.

Ocelot, MAD-X, Xsuite, elegant, Bmad and ImpactX can do this; the other codes
warn and propagate the incoming beam's Twiss instead. A beam file is still
needed: here it only sets the reference momentum.

Run:    python 02_tracking/10_twiss_without_particles.py
Tier:   A
Needs:  SIMBA, LAURA and Ocelot.
Output: output/10_twiss_without_particles/twiss.png, the periodic beta
        functions along the cell.

Units: beta in m; tune in turns (1 = 360 deg of phase advance); k1l in 1/m.

The cell is 09's: 4 m long, starting and ending in the middle of a QF (two
0.1 m halves with k1l = 0.3 /m), a 0.2 m QD (k1l = -0.6 /m) in the middle.

Known noise on SIMBA main: reading Ocelot's optics after a periodic run
prints RuntimeWarnings ("invalid value encountered in divide") from
simba/Modules/Twiss/ocelot.py, because the beam-based columns are zero there
(ISSUES.md #24). They don't affect the lattice Twiss used here. SIMBA
also gives a NotClosedWarning ("its geometry does not close"): it checks
whether a periodic line closes on itself like a ring. A straight FODO cell
never does, and its periodic optics are still the matched ones.

Check: the cos(mu) = Tr(M)/2 and beta = M12/sin(mu) of the thick-lens cell
matrix M (simple_lattice.transfer_matrix) must give Ocelot's tune and
periodic beta to 1e-6, and every point of Ocelot's beta along the cell must
match the periodic Twiss carried through the matrices to 1e-6. The script
also prints the thin-lens FODO formulas,
sin(mu/2) = L/(4f), beta_max,min = L(1 +- sin(mu/2))/sin(mu), for
comparison. They are up to 7% off here (beta_y at a QF), because the
quadrupoles have length.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simba.Framework as fw
from simple_lattice import fodo, make_beam, transfer_matrix, write_master_lattice

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
RUN = OUT / "run"
CELL, K1L = 4.0, 0.6

elements = fodo(cells=1, length=CELL, k1l=K1L)
files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "tracking": {"periodic": True},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
settings = write_master_lattice(LATTICE, elements, files, "cell.def")
make_beam(LATTICE / "START.openpmd.hdf5", particles=1000)

framework = fw.Framework(master_lattice=str(LATTICE), directory=str(RUN), clean=True, verbose=False)
framework.loadSettings(settings)
framework.track()
ring = framework["LINE"].ring_parameters()
twiss = fw.load_directory(str(RUN), master_lattice=str(LATTICE)).twiss

# Thin lens: each quad a kick 1/f = |k1l|, the cell length L.
sin_half = CELL * K1L / 4
mu_thin = 2 * np.arcsin(sin_half)
thin = {"beta_max": CELL * (1 + sin_half) / np.sin(mu_thin), "beta_min": CELL * (1 - sin_half) / np.sin(mu_thin)}

print(f"{'':16s} {'Ocelot':>10s} {'matrix':>10s} {'thin lens':>10s}")
problems = []
for plane in ("x", "y"):
    m = transfer_matrix(elements, plane, 0.0, CELL)
    mu = np.arccos(np.trace(m) / 2)
    beta0 = m[0, 1] / np.sin(mu)
    alpha0 = (m[0, 0] - m[1, 1]) / (2 * np.sin(mu))
    beta_thin = thin["beta_max"] if plane == "x" else thin["beta_min"]  # the cell starts at a QF
    print(f"tune_{plane:11s} {ring[f'tune_{plane}']:10.6f} {mu / (2 * np.pi):10.6f} {mu_thin / (2 * np.pi):10.6f}")
    print(f"beta_{plane} at START {ring[f'beta_{plane}']:10.6f} {beta0:10.6f} {beta_thin:10.6f}")
    if abs(ring[f"tune_{plane}"] / (mu / (2 * np.pi)) - 1) > 1e-6:
        problems.append(f"tune_{plane}: Ocelot {ring[f'tune_{plane}']}, matrix {mu / (2 * np.pi)}")
    if abs(ring[f"beta_{plane}"] / beta0 - 1) > 1e-6:
        problems.append(f"beta_{plane}: Ocelot {ring[f'beta_{plane}']}, matrix {beta0}")

    # Carry the periodic Twiss along the cell: B(s) = M(0->s) B0 M(0->s)^T, B = [[beta, -alpha], [-alpha, gamma]].
    b0 = np.array([[beta0, -alpha0], [-alpha0, (1 + alpha0 ** 2) / beta0]])
    z = np.asarray(twiss.z.val)
    beta_code = np.asarray(getattr(twiss, f"beta_{plane}").val)
    beta_matrix = []
    for zi in z:
        # Ocelot gives the optics at element ends, so no point falls inside a quad.
        mi = transfer_matrix(elements, plane, 0.0, zi)
        beta_matrix.append((mi @ b0 @ mi.T)[0, 0])
    worst = np.max(np.abs(beta_code / np.array(beta_matrix) - 1))
    if worst > 1e-6:
        problems.append(f"beta_{plane} along the cell differs from the matrices by {worst:.1e}")
    plt.plot(z, beta_code, marker=".", label=f"beta_{plane} (Ocelot)")

print(f"\nbeta along the cell ({len(z)} points, at element ends): "
      f"beta_x {np.min(twiss.beta_x.val):.3f}..{np.max(twiss.beta_x.val):.3f} m, "
      f"beta_y {np.min(twiss.beta_y.val):.3f}..{np.max(twiss.beta_y.val):.3f} m")
plt.xlabel("s [m]")
plt.ylabel("periodic beta [m]")
plt.legend()
plt.savefig(OUT / "twiss.png", dpi=120, bbox_inches="tight")
print(f"Plot: {OUT / 'twiss.png'}")

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print("\nOK: Ocelot's periodic tune and beta match the thick-lens cell matrix")
