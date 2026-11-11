"""
30 - A start-to-end FEL: elegant into Genesis 4
===============================================

Shows: a Genesis 4 line after a tracking line. JFEL's Lattices/jfel_fel.def
takes the saved injector beam through the linac in elegant (structure wakes
and CSR on) to the undulator entrance, then hands that beam to Genesis 4,
steady-state (FEL_ST, a 1 kW seed, seconds) or time-dependent (FEL_TD, SASE,
~4 min; not run here). Genesis' settings are in each line's `fel:` block.
SIMBA writes Genesis' input from the beam and the LAURA undulator; Genesis'
own output is FEL_ST.out.h5 in the run directory.

Run:    python 05_advanced/30_fel_genesis.py
Tier:   B
Needs:  SIMBA, LAURA, laura-lattices (JFEL), h5py; docker and
        ghcr.io/astec-stfc/simcodes-docker:master (elegant and Genesis).
        Prints SKIP and exits 0 without them.
Output: output/30_fel_genesis/ (FEL_ST.out.h5 is Genesis' output).

Units: wavelength in um; lengths in m; power in W; current in A.

The undulator (JFEL-FEL-MAG-WIGG-01) is helical, K = 8.52, period 82 mm,
ten segments. The Ming Xie gain length is computed here from the beam at
the undulator entrance: peak current, mean gamma, rms spread, the
geometric-mean size and emittance (Xie, NIM A 445 (2000) 59; the fit and
its 19 coefficients as in laura-lattices' JFEL/Scripts/run_fel.py).

Checks:
- Wavelength: Genesis' reference wavelength equals lambda_u (1 + K^2) /
  (2 gamma^2) with the beam's mean gamma, to 1e-3 (measured 8e-7: Genesis
  takes its reference from the same gamma, so this checks the hand-over).
- Gain length, fitted to Genesis' power between 1e-4 and 1e-2 of its
  maximum: longer than the 1D gain length (0.21 m; 3D effects only add to
  it), and 1x to 2x Ming Xie's (measured 0.757 against 0.498 m, 1.52x).
  Genesis is longer because Ming Xie treats the beam as round with
  averaged parameters, and after CSR eps_nx is ~5x eps_ny (JFEL README).
- The JFEL README's steady-state row: lambda 2.380 um to 1e-3, L_g 0.76 m
  and Ming Xie 0.50 m to 2%, 1.7 GW at the exit to 5%. The run repeats
  exactly, so these are the README's two-digit rounding (up to 3% for
  1.7) plus version drift. Measured: 0.4%, 0.4%, 1.5%.
"""

import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import JFEL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
JFEL_DIR = os.path.dirname(JFEL.__file__)
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"
ENTRANCE = "JFEL-FEL-SIM-MARK-01"
MC2, IA = 0.51099895e6, 17045.0  # eV; Alfven current [A]
XIE = [0.45, 0.57, 0.55, 1.6, 3, 2, 0.35, 2.9, 2.4, 51, 0.95, 3, 5.4, 0.7, 1.9, 1140, 2.2, 2.9, 3.2]
README = {"lambda": 2.380e-6, "Lg": 0.76, "Lg_xie": 0.50, "P_end": 1.7e9}  # JFEL README, steady-state


def docker_ready():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


def ming_xie(beam, period, K):
    """Ming Xie's 3D gain length for `beam` in a helical undulator; also rho, the 1D gain length, current."""
    q, t = np.abs(np.asarray(beam.Particles.charge.val)), np.asarray(beam.t.val)
    counts, edges = np.histogram(t, bins=max(int(np.sqrt(len(t))), 20), weights=q)
    current = np.convolve(counts / np.diff(edges), np.ones(3) / 3, "same").max()
    gamma = np.mean(beam.cp.val) / MC2
    spread = np.std(beam.cp.val) / np.mean(beam.cp.val)
    sigma = np.sqrt(np.std(beam.x.val) * np.std(beam.y.val))
    eps = np.sqrt(beam.Particles.emittance.enx.val * beam.Particles.emittance.eny.val) / gamma
    ku, lam = 2 * np.pi / period, period * (1 + K ** 2) / (2 * gamma ** 2)
    rho = (current / IA * 2 * K ** 2 / (16 * gamma ** 3 * sigma ** 2 * ku ** 2)) ** (1 / 3)  # helical: [JJ] = 1
    lg1d = period / (4 * np.pi * np.sqrt(3) * rho)
    d, e, g = lg1d * lam / (4 * np.pi * sigma ** 2), lg1d * 4 * np.pi * eps / (sigma ** 2 / eps * lam), 4 * np.pi * lg1d * spread / period
    a = XIE
    eta = (a[0] * d ** a[1] + a[2] * e ** a[3] + a[4] * g ** a[5] + a[6] * e ** a[7] * g ** a[8]
           + a[9] * d ** a[10] * g ** a[11] + a[12] * d ** a[13] * e ** a[14] + a[15] * d ** a[16] * e ** a[17] * g ** a[18])
    return {"Lg": lg1d * (1 + eta), "Lg1d": lg1d, "rho": rho, "current": current, "gamma": gamma}


if not docker_ready():
    print(f"SKIP: needs docker and {IMAGE} (elegant and Genesis)")
else:  # not sys.exit(0): in a notebook that is an error and stops "Run All"
    f = fw.Framework(master_lattice=JFEL_DIR + "/", directory=str(OUT), clean=True, verbose=False,
                     container_runtime="docker")
    f.loadSettings("Lattices/jfel_fel.def")
    f.track(files=["Linac", "FEL_ST"], check_lattice=False)

    undulator = f.getElement("JFEL-FEL-MAG-WIGG-01").magnetic
    period, K = float(undulator.period), float(undulator.strength)
    assert undulator.helical, "the formulas here are for a helical undulator"
    xie = ming_xie(rbf.load_file(str(OUT / f"{ENTRANCE}.openpmd.hdf5")), period, K)
    lam_formula = period * (1 + K ** 2) / (2 * xie["gamma"] ** 2)

    with h5py.File(OUT / "FEL_ST.out.h5") as h:
        z, power = h["Lattice/zplot"][()], h["Field/power"][()][:, 0]
        lam_genesis = float(h["Global/lambdaref"][()][0])
    peak = int(np.argmax(power))
    exponential = (power > 1e-4 * power[peak]) & (power < 1e-2 * power[peak]) & (np.arange(len(z)) < peak)
    lg = 1 / np.polyfit(z[exponential], np.log(power[exponential]), 1)[0]

    print(f"\nbeam at the undulator: I_peak {xie['current']:.0f} A, gamma {xie['gamma']:.1f}, rho {xie['rho']:.4f}")
    print(f"wavelength: Genesis {lam_genesis * 1e6:.4f} um, formula {lam_formula * 1e6:.4f} um "
          f"({lam_genesis / lam_formula - 1:+.1e}); K {K}, period {period} m")
    print(f"gain length: Genesis {lg:.3f} m, Ming Xie {xie['Lg']:.3f} m, 1D {xie['Lg1d']:.3f} m "
          f"(Genesis / Ming Xie = {lg / xie['Lg']:.2f})")
    print(f"power: {power[peak]:.3e} W at most (z = {z[peak]:.1f} m), {power[-1]:.3e} W at the exit")
    print(f"README: lambda {README['lambda'] * 1e6} um, L_g {README['Lg']} m, Ming Xie {README['Lg_xie']} m, "
          f"{README['P_end']:.1e} W at the exit")

    # --- Checks ------------------------------------------------------------
    problems = []
    if abs(lam_genesis / lam_formula - 1) > 1e-3:
        problems.append(f"wavelength {lam_genesis:.5e}, formula {lam_formula:.5e}")
    if lg < xie["Lg1d"]:
        problems.append(f"gain length {lg:.3f} m shorter than the 1D one {xie['Lg1d']:.3f} m")
    if not 1 <= lg / xie["Lg"] <= 2:
        problems.append(f"gain length {lg:.3f} m, {lg / xie['Lg']:.2f}x Ming Xie's")
    for got, key, limit in ((lam_genesis, "lambda", 1e-3), (lg, "Lg", 0.02), (xie["Lg"], "Lg_xie", 0.02),
                            (power[-1], "P_end", 0.05)):
        if abs(got / README[key] - 1) > limit:
            problems.append(f"{key}: {got:.4g}, README {README[key]}")
    assert not problems, "\n".join(problems)
    print("\nOK: the resonant wavelength, a gain length between 1x and 2x Ming Xie's, and the README's row")
