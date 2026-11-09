"""
08 - Track a beam through a drift
=================================

Shows: the smallest complete tracking run. A 2 m line with only two
markers (SIMBA fills the gap with a drift), a converging beam, one line in
the .def run by Ocelot, and the beam read back at the end.

The steps are the same for any machine:
  1. a master lattice folder: the LAURA YAML plus a .def (here written by
     simple_lattice.py; for JFEL they are in laura-lattices);
  2. Framework(master_lattice=..., directory=...) and loadSettings(".def");
  3. framework.track();
  4. read the beam files SIMBA wrote in `directory`, one per screen
     (marker) plus one at the end of each line, named after the element.

Run:    python 02_tracking/08_track_drift.py
Tier:   A
Needs:  SIMBA, LAURA and Ocelot.
Output: output/08_track_drift/lattice/ (the master lattice and START beam)
        and output/08_track_drift/run/ (Ocelot's input deck LINE.py, the
        beam END.openpmd.hdf5, and SIMBA's summaries).

Units: x in m; x' = px/pz (rad); emittance in m rad; momentum in eV/c.

The starting beam converges (alpha_x = alpha_y = 2 at beta = 5 m), so it
passes through a waist 2 m on, at the END marker, with beta* = 1/gamma = 1 m.

Check: a drift of length L maps x to x + L x', so the rms size at the end
is sigma^2(L) = <x^2> + 2L<x x'> + L^2 <x'^2>, with the moments taken from
the starting beam. The tracked rms sizes must match this to 1e-6
relative. Ocelot's drift is exact to second order, and the beam's
energy spread (1e-4) changes nothing at that level. N, charge, emittance
and mean momentum must be unchanged, to the same 1e-6.
"""

from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from simple_lattice import make_beam, marker, sigma_matrix, write_master_lattice

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
L = 2.0  # m

# The .def: one line, run by Ocelot, from START to END. Its beam is the file
# START.openpmd.hdf5 in the master lattice folder.
files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
settings = write_master_lattice(LATTICE, [marker("START", 0.0), marker("END", L)], files, "drift.def")
start = make_beam(LATTICE / "START.openpmd.hdf5", beta=(5.0, 5.0), alpha=(2.0, 2.0))

framework = fw.Framework(master_lattice=str(LATTICE), directory=str(OUT / "run"), clean=True, verbose=False)
framework.loadSettings(settings)
print(f"Line LINE: {framework['LINE'].code}, elements {list(framework['LINE'].elements)}")
framework.track()
end = rbf.load_file(str(OUT / "run" / "END.openpmd.hdf5"))

print(f"\n{'':12s} {'START':>12s} {'END':>12s} {'expected':>12s}")
problems = []
for plane in ("x", "y"):
    s = sigma_matrix(start, plane)
    expected = np.sqrt(s[0, 0] + 2 * L * s[0, 1] + L ** 2 * s[1, 1])
    got = np.std(getattr(end, plane).val)
    print(f"sigma_{plane} [um] {np.sqrt(s[0, 0]) * 1e6:12.3f} {got * 1e6:12.3f} {expected * 1e6:12.3f}")
    if abs(got / expected - 1) > 1e-6:
        problems.append(f"sigma_{plane}: {got:.6e} m, expected {expected:.6e} m")

# The same in Twiss terms: beta(L) = beta0 - 2 alpha0 L + gamma0 L^2.
P0, P1 = start.Particles.twiss, end.Particles.twiss
beta_expected = P0.beta_x.val - 2 * P0.alpha_x.val * L + (1 + P0.alpha_x.val ** 2) / P0.beta_x.val * L ** 2
print(f"beta_x [m]   {P0.beta_x.val:12.4f} {P1.beta_x.val:12.4f} {beta_expected:12.4f}"
      f"   (waist: alpha_x {P1.alpha_x.val:+.4f})")

quantities = {
    "N": lambda b: len(b.x),
    "charge [pC]": lambda b: b.Q * 1e12,
    "enx [um]": lambda b: b.Particles.emittance.enx.val * 1e6,
    "eny [um]": lambda b: b.Particles.emittance.eny.val * 1e6,
    "<cp> [MeV/c]": lambda b: np.mean(b.cp.val) / 1e6,
}
for name, fn in quantities.items():
    a, b = fn(start), fn(end)
    print(f"{name:12s} {a:12.6g} {b:12.6g}")
    if abs(b / a - 1) > 1e-6:
        problems.append(f"{name} changed: {a} -> {b}")

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print("\nOK: rms sizes follow the drift's sigma matrix; N, charge, emittance and momentum unchanged")
