"""
29 - A plasma-wakefield stage in Wake-T
=======================================

Shows: a plasma stage as one more SIMBA line. A LAURA `Plasma` element
holds the density profile and the simulation box; with no laser the
incoming beam drives the wake (PWFA). The stage runs in Wake-T
(quasi-static, seconds on a CPU) or, with one call, in FBPIC (full PIC,
GPU; not run here):

    f.change_Lattice_Code("PWFA", "fbpic")

The next line (Ocelot) carries the beam on through a 0.5 m drift to a
screen.

Run:    python 05_advanced/29_plasma_wake_t.py
Tier:   A
Needs:  SIMBA, LAURA, laura-lattices (PLASMA), Wake-T and Ocelot (all in
        simba-accelerator[simcodes]).
Output: output/29_plasma_wake_t/<lattice>/.

Units: gamma (Lorentz factor); spread relative, rms; emittance normalised,
in um, against u_x = p_x / mc as Wake-T reports it; beta in mm; energy in mJ.

Two of laura-lattices' PLASMA lattices, each a 500 pC, 1 GeV driver and a
50 pC, 100 MeV witness 80 um behind it, through 1 cm of plasma at 1e23 m^-3:
- pwfa_single: the plateau alone;
- pwfa_ramps: the same plateau between 2 mm matching ramps (density n0/25
  at the ends), so beta grows as n^-1/2 on the way out.
The driver and witness are told apart by energy (500 MeV), as in
PLASMA/Scripts/run_example.py.

Checks:
- The driver loses energy and the witness gains it: in pwfa_single the
  driver's gamma falls from 2000 to 1702 and the witness's rises from 200
  to 637. The witness gains less energy than the driver loses (measured
  15%, 14% with the ramps; the rest stays in the wake).
- Each stage reproduces the PLASMA README's witness table (gamma, spread,
  eps_nx and beta at the exit, eps_nx at the screen) to 1%. Same input and
  codes, so this is rounding plus version drift. Measured: 0.3% or less
  (pwfa_single's screen eps_nx is 1.994 against 2.000).
- The ramps do what they are for: beta at the exit grows more than 4x
  (designed 5x, measured 5.1x), and the emittance growth in the drift to
  the screen is smaller (measured 1.83 against 1.99 um).
"""

import os
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import PLASMA

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
PLASMA_DIR = os.path.dirname(PLASMA.__file__)
MC2, E = 0.51099895e6, 1.602176634e-19  # eV; C
SPLIT = 500e6  # eV/c: the driver above, the witness below
LATTICES = {"pwfa_single": "PWFA", "pwfa_ramps": "PWFAR"}  # lattice: its plasma line
README = {  # PLASMA README, pwfa_ramps table: witness gamma, spread, eps_nx at exit, beta at exit [mm], eps_nx at screen
    "pwfa_single": (636.7, 0.0071, 0.987, 0.59, 2.000),
    "pwfa_ramps": (648.4, 0.0079, 0.991, 3.01, 1.829),
}


def bunch(beam, witness):
    """gamma, spread, eps_nx [um], beta [mm], energy [mJ] of the driver or the witness."""
    sel = (beam.cp.val <= SPLIT) if witness else (beam.cp.val > SPLIT)
    q = np.abs(np.asarray(beam.Particles.charge.val))[sel]
    gamma = np.hypot(beam.cp.val[sel] / MC2, 1)
    mean = np.average(gamma, weights=q)
    cov = np.cov(np.vstack([beam.x.val[sel], beam.cpx.val[sel] / MC2]), aweights=q)
    eps = np.sqrt(np.linalg.det(cov))
    return {"gamma": mean, "spread": np.sqrt(np.cov(gamma, aweights=q)) / mean, "eps_nx": eps * 1e6,
            "beta": cov[0, 0] / (eps / mean) * 1e3, "energy": np.sum(q * gamma) * MC2 * 1e3}  # q/e * gamma mc^2 [eV] * e


def run(lattice):
    """The stage and its drift; driver and witness at the entrance, the plasma's exit and the screen."""
    directory = OUT / lattice
    f = fw.Framework(master_lattice=PLASMA_DIR + "/", directory=str(directory), clean=True, verbose=False)
    f.loadSettings(f"Lattices/{lattice}.def")
    f.track()
    plasma, drift = f[LATTICES[lattice]], f[LATTICES[lattice] + "_DIAG"]
    beams = {"entrance": rbf.load_file(f"{PLASMA_DIR}/Data_Files/{lattice.replace('_single', '')}_driver_witness.openpmd.hdf5"),
             "exit": rbf.load_file(str(directory / f"{plasma.end}.openpmd.hdf5")),
             "screen": rbf.load_file(str(directory / f"{drift.end}.openpmd.hdf5"))}
    return {(where, who): bunch(b, who == "witness") for where, b in beams.items() for who in ("driver", "witness")}


results = {lattice: run(lattice) for lattice in LATTICES}

print(f"{'lattice':12s} {'where':9s} {'bunch':8s} {'gamma':>8s} {'spread':>8s} {'eps_nx':>8s} {'beta':>8s} {'E [mJ]':>8s}")
for lattice, r in results.items():
    for (where, who), b in r.items():
        print(f"{lattice:12s} {where:9s} {who:8s} {b['gamma']:8.1f} {b['spread']:8.2%} {b['eps_nx']:8.3f} "
              f"{b['beta']:8.2f} {b['energy']:8.3f}")
    lost = r[("entrance", "driver")]["energy"] - r[("exit", "driver")]["energy"]
    gained = r[("exit", "witness")]["energy"] - r[("entrance", "witness")]["energy"]
    print(f"{lattice:12s} driver lost {lost:.2f} mJ, witness gained {gained:.2f} mJ ({gained / lost:.0%})\n")

# --- Checks ----------------------------------------------------------------
problems = []
for lattice, r in results.items():
    lost = r[("entrance", "driver")]["energy"] - r[("exit", "driver")]["energy"]
    gained = r[("exit", "witness")]["energy"] - r[("entrance", "witness")]["energy"]
    if not 0 < gained < lost:
        problems.append(f"{lattice}: driver lost {lost:.3f} mJ, witness gained {gained:.3f} mJ")
    w, s = r[("exit", "witness")], r[("screen", "witness")]
    got = (w["gamma"], w["spread"], w["eps_nx"], w["beta"], s["eps_nx"])
    if max(abs(g / ref - 1) for g, ref in zip(got, README[lattice])) > 0.01:
        problems.append(f"{lattice}: {np.round(got, 4)} against the README's {README[lattice]}")
single, ramps = results["pwfa_single"], results["pwfa_ramps"]
if ramps[("exit", "witness")]["beta"] / single[("exit", "witness")]["beta"] < 4:
    problems.append("the ramps grew beta at the exit less than 4x")
if ramps[("screen", "witness")]["eps_nx"] >= single[("screen", "witness")]["eps_nx"]:
    problems.append("the ramps did not reduce the emittance growth in the drift")
assert not problems, "\n".join(problems)
print("OK: the driver gives energy to the witness, the README's witness table is reproduced, and the ramps "
      "grow beta and cut the drift's emittance growth")
