"""
09 - Track through a FODO channel
=================================

Shows: tracking a beam through three FODO cells in Ocelot, with a screen
(marker) halfway along every drift. It reads the beam at the screens, loads
the whole run back with `load_directory`, and plots the beam size and
emittance along s.

Run:    python 02_tracking/09_track_fodo.py
Tier:   A
Needs:  SIMBA, LAURA and Ocelot.
Output: output/09_track_fodo/run/: one beam file per screen
        (SCR0A.openpmd.hdf5, ...) and END.openpmd.hdf5;
        output/09_track_fodo/envelope.png.

Units: x in m; x' = px/pz (rad); k1l in 1/m; emittance in m rad.

The lattice: 4 m cells, 0.2 m quadrupoles with k1l = +-0.6 /m (a phase
advance of 71 deg per cell). The beam starts with beta = 5 m and alpha = 0
in both planes. That isn't the cell's matched (periodic) beta, so the beam
size beats from cell to cell instead of repeating (10 shows the matched
beam).

load_directory(directory) reads a finished run back (the Twiss and beam
summaries, and with beams=True the beam at every screen), so you can look at
a run without tracking it again. On SIMBA main, give it the master lattice
too: without it, reading the run's settings.def fails when the .def names
its lattice files relative to the master lattice, as every laura-lattices
.def does (ISSUES.md #22). On main it also warns that `simcodes` and
`container_runtime` aren't set and prints "SimCodes not available". Both are
harmless here, because loading a run doesn't run a code (ISSUES.md #20).

Check: at every screen, the tracked beam's sigma matrix must match
M Sigma0 M^T, where Sigma0 is the starting beam's and M the thick-lens
transfer matrix from the start (simple_lattice.transfer_matrix), to 1e-4
relative. Ocelot also includes the second-order (chromatic) terms that the
matrix leaves out; they give a measured 2e-5, so the limit has 5x headroom.
The emittance must stay the same to 1e-4.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from simple_lattice import fodo, make_beam, sigma_matrix, transfer_matrix, write_master_lattice

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
RUN = OUT / "run"

elements = fodo(cells=3)
files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
settings = write_master_lattice(LATTICE, elements, files, "fodo.def")
start = make_beam(LATTICE / "START.openpmd.hdf5", beta=(5.0, 5.0), alpha=(0.0, 0.0))

framework = fw.Framework(master_lattice=str(LATTICE), directory=str(RUN), clean=True, verbose=False)
framework.loadSettings(settings)
framework.track()

screens = [e for e in elements if e.hardware_class == "Marker" and e.name != "START"]
print(f"{'screen':8s} {'s [m]':>6s} {'sigma_x [um]':>13s} {'expected':>9s} {'sigma_y [um]':>13s} "
      f"{'expected':>9s} {'enx [um]':>9s}")
worst, problems = 0.0, []
for screen in screens:
    z = screen.physical.middle.z
    beam = rbf.load_file(str(RUN / f"{screen.name}.openpmd.hdf5"))
    row = f"{screen.name:8s} {z:6.2f}"
    for plane in ("x", "y"):
        m = transfer_matrix(elements, plane, 0.0, z)
        expected = m @ sigma_matrix(start, plane) @ m.T
        got = sigma_matrix(beam, plane)
        # each element of the matrix, on the scale of the diagonal (an off-diagonal one can be ~0)
        scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        worst = max(worst, np.max(np.abs(got - expected) / scale))
        row += f" {np.sqrt(got[0, 0]) * 1e6:13.2f} {np.sqrt(expected[0, 0]) * 1e6:9.2f}"
    enx = beam.Particles.emittance.enx.val
    row += f" {enx * 1e6:9.5f}"
    print(row)
    if abs(enx / start.Particles.emittance.enx.val - 1) > 1e-4:
        problems.append(f"{screen.name}: enx {enx:.6e}, started at {start.Particles.emittance.enx.val:.6e}")
print(f"\nWorst difference from M Sigma0 M^T: {worst:.1e} (relative)")

# The whole run, read back from the directory. The plot's lines come from
# Ocelot's own optics output (Twiss_Summary.hdf5), one point per element.
run = fw.load_directory(str(RUN), master_lattice=str(LATTICE), beams=True)
print(f"load_directory: {len(run.twiss.z)} points along s, {len(run.beams.beams)} beam files")
run.plot(xkey="z", ykeys=["sigma_x", "sigma_y"], ykeys2=["enx", "eny"])
plt.savefig(OUT / "envelope.png", dpi=120, bbox_inches="tight")
print(f"Plot: {OUT / 'envelope.png'}")

# --- Check -----------------------------------------------------------------
assert worst < 1e-4, f"tracked sigma matrices differ from M Sigma0 M^T by {worst:.1e}"
assert not problems, "\n".join(problems)
print("\nOK: the beam at every screen follows the transfer matrices; emittance unchanged")
