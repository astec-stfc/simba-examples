"""
24 - Scan two parameters in parallel
====================================

Shows: a grid scan run over several processes, one directory per run, with
the results gathered into a CSV. It scans JFEL's chicane angle against the
L4H cavity phase and maps the bunch length after the chicane: the
"compression valley" of the JFEL README.

Run:    python 04_scans_optimisation/24_parameter_scan_parallel.py
Tier:   A
Needs:  SIMBA, LAURA, laura-lattices (JFEL) and Ocelot.
Output: output/24_parameter_scan_parallel/: one directory per run
        (angle_phase/), scan.csv and valley.png.

Units: angle in rad; phase in degrees (SIMBA's convention); sigma_t in fs.

How it runs in parallel:
- concurrent.futures.ProcessPoolExecutor, one process per core. Each
  process builds one Framework when it starts (the pool's initializer),
  and every run then only changes settings, so the lattice is loaded once
  per process, not per run.
- framework.setSubDirectory(path) before each track() gives each run its
  own directory, so runs don't overwrite each other's files.
- Each process uses one thread (OMP_NUM_THREADS=1 and friends, set before
  numpy is imported). Ocelot is multithreaded through numpy, and 8
  processes x 8 threads on 8 cores runs twice as slowly as 8 x 1.
- sample_interval = 4 tracks every 4th particle, with the same total
  charge: 4x faster, and sigma_t changes by ~1 fs.

The line is JFEL's Lattices/jfel_compression.def, CSR off, wakes on (as
22). Each run calls set_angle on the chicane group (which also sets the
dipole lengths; see ISSUES.md #27) and sets the L4H phase.

Check: the valley. Along each angle's row, sigma_t falls linearly to a
rounded floor at full compression and rises linearly after it. Straight
lines fitted to the two arms (points above twice the floor) meet at the
phase of full compression, phi_c. It must rise steadily with the angle: a
bigger angle gives a bigger R56, so full compression needs a different
chirp from L4H (measured ~0.9 deg per mrad). And it must be within 5 deg
(one grid step) of the README's "L4H phase at full compression", read off
the same grid by its minimum. Measured 2.1-2.9 deg, every one above the
README's grid value.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import csv  # noqa: E402
import itertools  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
L4H, END = "JFEL-L4H-LIN-CAV-01", "JFEL-S06-SIM-MARK-01"
ANGLES = [0.120, 0.125, 0.130, 0.135, 0.140]
PHASES = np.arange(-200.0, -154.0, 5.0)
README_VALLEY = {0.120: -190, 0.125: -185, 0.130: -180, 0.135: -175, 0.140: -170}
framework = None  # one per worker process


def start_worker():
    global framework
    import simba.Framework as fw
    from laura_lattices import JFEL

    framework = fw.Framework(master_lattice=os.path.dirname(JFEL.__file__),
                             directory=str(OUT / f"worker_{os.getpid()}"), clean=True, verbose=False)
    framework.loadSettings("Lattices/jfel_compression.def")
    framework.change_Lattice_Code("Compressor", "ocelot")
    framework["Compressor"].csr_enable = False
    framework["Compressor"].sample_interval = 4


def run(point):
    """One scan point, in a worker: set, track into its own directory, return sigma_t [fs]."""
    import simba.Modules.Beams as rbf

    angle, phase = point
    directory = OUT / f"{angle:.3f}_{phase:+.0f}"
    framework.setSubDirectory(str(directory))
    framework["bunch_compressor"].set_angle(angle)
    framework.modifyElement(L4H, "phase", float(phase))
    framework.track(files=["Compressor"], check_lattice=False)
    beam = rbf.load_file(str(directory / f"{END}.openpmd.hdf5"))
    sigma_t = np.sqrt(np.cov(beam.t.val, aweights=np.abs(beam.Particles.charge.val)))
    return angle, float(phase), sigma_t * 1e15, str(directory.relative_to(OUT))


def full_compression_phase(sigma_t):
    """Where straight lines through the two arms of a row meet."""
    low = np.argmin(sigma_t)
    arm = sigma_t > 2 * sigma_t[low]
    left, right = arm & (np.arange(len(PHASES)) < low), arm & (np.arange(len(PHASES)) > low)
    assert left.sum() >= 2 and right.sum() >= 2, f"not enough points on the arms: {sigma_t.round()}"
    (m1, c1), (m2, c2) = np.polyfit(PHASES[left], sigma_t[left], 1), np.polyfit(PHASES[right], sigma_t[right], 1)
    return (c2 - c1) / (m1 - m2)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    points = list(itertools.product(ANGLES, PHASES))
    workers = os.cpu_count()
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=start_worker) as pool:
        results = list(pool.map(run, points))
    print(f"{len(points)} runs on {workers} processes in {time.time() - t0:.0f} s")

    with open(OUT / "scan.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_rad", "l4h_phase_deg", "sigma_t_fs", "directory"])
        writer.writerows(results)
    grid = np.array([r[2] for r in results]).reshape(len(ANGLES), len(PHASES))

    print(f"\nsigma_t [fs]; phases {PHASES[0]:.0f} to {PHASES[-1]:.0f} deg in steps of 5")
    valley = {}
    for angle, row in zip(ANGLES, grid):
        valley[angle] = full_compression_phase(row)
        print(f"  {angle:.3f} rad: " + " ".join(f"{s:4.0f}" for s in row)
              + f"   phi_c {valley[angle]:7.2f} (README {README_VALLEY[angle]})")

    plt.pcolormesh(PHASES, ANGLES, grid, shading="nearest", cmap="viridis")
    plt.colorbar(label="sigma_t after the chicane [fs]")
    plt.plot(list(valley.values()), ANGLES, "w.-", label="full compression (arm fits)")
    plt.plot(list(README_VALLEY.values()), ANGLES, "rx", label="JFEL README")
    plt.xlabel("L4H phase [deg]")
    plt.ylabel("chicane angle [rad]")
    plt.legend()
    plt.savefig(OUT / "valley.png", dpi=120, bbox_inches="tight")
    print(f"\nCSV: {OUT / 'scan.csv'}\nPlot: {OUT / 'valley.png'}")

    # --- Check -------------------------------------------------------------
    phi = np.array([valley[a] for a in ANGLES])
    assert np.all(np.diff(phi) > 0), f"full-compression phase does not rise with angle: {phi.round(2)}"
    worst = max(abs(valley[a] - README_VALLEY[a]) for a in ANGLES)
    assert worst <= 5.0, f"valley {worst:.1f} deg from the README's"
    print(f"OK: the valley rises with the angle and is within {worst:.1f} deg of the README's")
