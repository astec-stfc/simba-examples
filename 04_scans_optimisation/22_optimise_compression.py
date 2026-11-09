"""
22 - Optimise the JFEL bunch compressor for a target peak current
=================================================================

Shows: optimising a real lattice. JFEL's L4H harmonic cavity sets the
energy chirp the chicane compresses. A root finder (scipy brentq) adjusts
its phase until the peak current after the chicane is 1000 A. It then
re-runs the answer in other codes, which shows the JFEL README's warning
that the model must include what shapes the compression: here, the
structure wakes.

Run:    python 04_scans_optimisation/22_optimise_compression.py
Tier:   A (Ocelot, Cheetah); B for the elegant check (docker).
Needs:  SIMBA, LAURA, laura-lattices (JFEL), Ocelot and Cheetah; docker and
        ghcr.io/astec-stfc/simcodes-docker:master for elegant (skipped
        otherwise).
Output: output/22_optimise_compression/: one directory per code.

Units: phase in degrees (SIMBA's convention: -180 is the decelerating
crest); peak current in A; sigma_t in fs.

The line is JFEL's Lattices/jfel_compression.def ("Compressor", from the
L4H cavity through the chicane), at the nominal chicane angle (0.13 rad)
and amplitude, with CSR off (it doubles the run time and changes little
here) and the structure wakes on. The figures of merit are as in JFEL's
Scripts/compression.py: the charge-weighted sigma_t, and the peak of a
200-bin current histogram smoothed over 3 bins. "Spikiness" is the peak
over a flat-top bunch's current, Q / (sqrt(12) sigma_t).

At the nominal phase (-184) the bunch is not compressed enough (~480 A);
at -180 it is past the edge of full compression (~1560 A). brentq brackets
[-184, -180] and stops at 0.01 deg.

Checks:
- Ocelot (wakes on) reaches 1000 A within 1%. The 0.01 deg tolerance is
  ~4 A at the slope here (~400 A/deg); measured 1001 A, in 7 tracks.
- At the same phase, Cheetah (which has no structure wakes) agrees with
  Ocelot run without wakes to 5% (measured 2%): with the same physics the
  codes agree.
- The wakes change the answer: Ocelot's peak current is >1.5x higher with
  them than without (measured 2.05x). An optimisation in a code without
  wakes (Cheetah, MAD-X) would land somewhere else entirely.
- elegant (docker) shows the same: >1.5x with the wakes (measured 2.8x).
  Its current with the wakes is printed, not checked: measured 1547 A
  against Ocelot's 999 A. The README quotes ~40% for the same comparison
  with CSR on. The optimum sits on the edge of full compression, where
  +-0.2 deg moves the current by ~8% (printed below), so small differences
  between the codes' RF and wake models show up large.

The README's optimisation also varies the chicane angle and the amplitude
(optimise_compression.py --target 1000, 1010 A). One target needs one knob,
so this keeps the other two at nominal. The README's sigma_t values were
made with an older SIMBA, and today's runs differ (README "~190 fs, 450 A"
at nominal, measured 182 fs, 480 A); the checks above don't use them.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import JFEL
from scipy.optimize import brentq

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
JFEL_DIR = os.path.dirname(JFEL.__file__)
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"
L4H, END = "JFEL-L4H-LIN-CAV-01", "JFEL-S06-SIM-MARK-01"
TARGET = 1000.0


def docker_ready():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


def compressor(code, tag, wakes=True, **kwargs):
    """JFEL's compression line in `code`, with CSR off."""
    f = fw.Framework(master_lattice=JFEL_DIR, directory=str(OUT / tag), clean=True, verbose=False, **kwargs)
    f.loadSettings("Lattices/jfel_compression.def")
    f.change_Lattice_Code("Compressor", code)
    f["Compressor"].csr_enable = False
    f["Compressor"].wakefield_enable = wakes
    return f


def compress(f, tag, phase):
    """Track at an L4H phase; return the peak current [A], sigma_t [fs] and spikiness."""
    f.modifyElement(L4H, "phase", float(phase))
    f.track(files=["Compressor"], check_lattice=False)
    beam = rbf.load_file(str(OUT / tag / f"{END}.openpmd.hdf5"))
    q, t = np.abs(beam.Particles.charge.val), beam.t.val
    sigma_t = np.sqrt(np.cov(t, aweights=q))
    counts, edges = np.histogram(t, bins=200, weights=q)
    peak = np.convolve(counts / (edges[1] - edges[0]), np.ones(3) / 3, mode="same").max()
    spiky = peak / (q.sum() / (np.sqrt(12) * sigma_t))
    print(f"  {tag:16s} phase {phase:9.3f}: I = {peak:6.0f} A, sigma_t = {sigma_t * 1e15:5.1f} fs, spikiness {spiky:.2f}")
    return peak


print("Ocelot, wakes on: solving for the L4H phase")
ocelot = compressor("ocelot", "ocelot")
phase = brentq(lambda p: compress(ocelot, "ocelot", p) - TARGET, -184.0, -180.0, xtol=0.01)
current = compress(ocelot, "ocelot", phase)
print(f"solution: L4H phase {phase:.3f} deg\n\nThe edge of compression, +-0.2 deg:")
for p in (phase - 0.2, phase + 0.2):
    compress(ocelot, "ocelot", p)

print("\nThe same phase without structure wakes:")
no_wakes = compress(compressor("ocelot", "ocelot_no_wakes", wakes=False), "ocelot_no_wakes", phase)
cheetah = compress(compressor("cheetah", "cheetah"), "cheetah", phase)

elegant = None
if docker_ready():
    print("\nelegant (docker):")
    elegant = {w: compress(compressor("elegant", f"elegant_{t}", wakes=w, container_runtime="docker"), f"elegant_{t}", phase)
               for w, t in ((True, "wakes"), (False, "no_wakes"))}
    print(f"elegant with wakes / Ocelot with wakes = {elegant[True] / current:.2f} (not checked; see the docstring)")
else:
    print(f"\nelegant check skipped: docker or {IMAGE} not available")

# --- Checks ----------------------------------------------------------------
print()
assert abs(current / TARGET - 1) < 0.01, f"Ocelot reached {current:.0f} A, target {TARGET:.0f} A"
assert abs(cheetah / no_wakes - 1) < 0.05, f"Cheetah {cheetah:.0f} A, Ocelot without wakes {no_wakes:.0f} A"
assert current / no_wakes > 1.5, f"wakes changed Ocelot's current only {current / no_wakes:.2f}x"
if elegant:
    assert elegant[True] / elegant[False] > 1.5, f"wakes changed elegant's current only {elegant[True] / elegant[False]:.2f}x"
print(f"OK: {current:.0f} A at L4H {phase:.2f} deg; without wakes the codes agree "
      f"(Cheetah/Ocelot {cheetah / no_wakes:.3f}), and the wakes multiply the current by {current / no_wakes:.2f}")
