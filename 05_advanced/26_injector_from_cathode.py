"""
26 - Track the JFEL injector from the cathode in ASTRA
======================================================

Shows: a start-to-end run's first stage, from the cathode. SIMBA generates
the laser-shaped bunch at the cathode (the .def's `generator:` block, with
ASTRA's generator and JFEL's jfel.yaml defaults), then ASTRA tracks it
through the HRRG gun and the first linac (L01) to ~35 MeV, with 2D space
charge and the cathode's mirror charge. This is how JFEL's saved injector
beam, Data_Files/jfel_injector_S02.openpmd.hdf5, the input to 13, 14, 22,
23 and 25, was made (JFEL's Generators/make_injector_beam.py).

    framework = fw.Framework(master_lattice=JFEL + "/", generator_defaults="/jfel.yaml", ...)
    framework.loadSettings("Lattices/jfel_combined.def")
    framework["generator"].number_of_particles = 4096
    framework.setSeedValue(1234)
    framework.track(endfile="injector400")   # run up to and including this line

Run:    python 05_advanced/26_injector_from_cathode.py [--particles 4096]
Tier:   B
Needs:  SIMBA, LAURA, laura-lattices (JFEL), docker and
        ghcr.io/astec-stfc/simcodes-docker:master (ASTRA).
Output: output/26_injector_from_cathode/: ASTRA's files and the beam at
        every screen.

Units: energy in MeV; charge in pC; emittance (normalised) in um; sigma_t
in ps; sigma_x in mm.

Checks, against the saved beam (32768 particles, seed 1234; the README:
"250 pC, 35 MeV, eps_n 0.81 um, 3 ps flat-top"):
- The energy to 0.1%, the charge to 1e-9, sigma_t to 2% and sigma_x to 3%
  (measured at 4096: 5e-5, exact, 0.1%, 0.3%; sampling noise ~1/sqrt(N)
  is 1.6% for a size).
- eps_n to 15% with fewer than 16384 particles, 2% with more. Space charge
  with few macroparticles adds numerical emittance growth, more than the
  sampling noise: measured +10% at 4096 (26 s), +0.6% at 16384 (2 min).
- With 32768 particles (5 min), the same seed gives back the saved beam
  exactly (measured 1e-13): the run is reproducible.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import JFEL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
JFEL_DIR = os.path.dirname(JFEL.__file__)
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"
SCREEN = "JFEL-S02-SIM-APER-01"
SAVED_PARTICLES, SEED = 32768, 1234

parser = argparse.ArgumentParser()
parser.add_argument("--particles", type=int, default=4096)
args = parser.parse_args()


def summary(beam):
    q = np.abs(beam.Particles.charge.val)
    return {"E [MeV]": np.mean(np.hypot(beam.cp.val, 0.51099895e6)) / 1e6, "Q [pC]": q.sum() * 1e12,
            "eps_nx [um]": beam.Particles.emittance.enx.val * 1e6, "eps_ny [um]": beam.Particles.emittance.eny.val * 1e6,
            "sigma_t [ps]": np.std(beam.t.val) * 1e12, "sigma_x [mm]": np.std(beam.x.val) * 1e3}


if not (shutil.which("docker") and subprocess.run(["docker", "image", "inspect", IMAGE],
                                                  capture_output=True).returncode == 0):
    print(f"SKIP: needs docker and {IMAGE} for ASTRA")
else:  # not sys.exit(0): in a notebook that is an error and stops "Run All"
    framework = fw.Framework(master_lattice=JFEL_DIR + "/", generator_defaults="/jfel.yaml", directory=str(OUT),
                             clean=True, verbose=False, container_runtime="docker")
    framework.loadSettings("Lattices/jfel_combined.def")
    framework["generator"].number_of_particles = args.particles
    framework.setSeedValue(SEED)
    framework.track(endfile="injector400")

    run = summary(rbf.load_file(str(OUT / f"{SCREEN}.openpmd.hdf5")))
    saved = summary(rbf.load_file(os.path.join(JFEL_DIR, "Data_Files", "jfel_injector_S02.openpmd.hdf5")))
    print(f"\nAt {SCREEN}:")
    print(f"{'':14s} {f'this run ({args.particles})':>18s} {f'saved ({SAVED_PARTICLES})':>15s} {'rel diff':>9s}")
    for key in run:
        print(f"{key:14s} {run[key]:18.4f} {saved[key]:15.4f} {run[key] / saved[key] - 1:9.1e}")

    # --- Checks ------------------------------------------------------------
    eps_tol = 0.15 if args.particles < 16384 else 0.02
    limits = {"E [MeV]": 1e-3, "Q [pC]": 1e-9, "sigma_t [ps]": 0.02, "sigma_x [mm]": 0.03,
              "eps_nx [um]": eps_tol, "eps_ny [um]": eps_tol}
    if args.particles == SAVED_PARTICLES:
        limits = dict.fromkeys(limits, 1e-9)
    bad = [f"{k}: {run[k]:.4f} against {saved[k]:.4f}" for k, tol in limits.items() if abs(run[k] / saved[k] - 1) > tol]
    assert not bad, "\n".join(bad)
    print(f"\nOK: {run['E [MeV]']:.1f} MeV, {run['Q [pC]']:.0f} pC, eps_n {run['eps_nx [um]']:.2f} um from the cathode"
          + (" (the saved beam, exactly)" if args.particles == SAVED_PARTICLES else ""))
