"""
25 - Collective effects: wakes, CSR, LSC
========================================

Shows: SIMBA's switches for collective effects, one at a time. Every line
has `csr_enable`, `lsc_enable` and `wakefield_enable`; SIMBA passes them to
each code in that code's own terms. A code that doesn't model an effect
tracks without it (no error), so check which codes model what:

    | Effect                   | Switch           | Modelled by (JFEL README)  |
    | structure wakes          | wakefield_enable | elegant, Ocelot            |
    | CSR in bends             | csr_enable       | elegant, Ocelot, ImpactX, Bmad |
    | longitudinal space charge| lsc_enable       | elegant, Ocelot            |

(3D space charge is set differently, with the line's
`charge: {space_charge_mode: 3d}`; it takes ~90 s in Ocelot here and is
left out.)

Run:    python 05_advanced/25_collective_effects.py
Tier:   A (Ocelot); B for elegant (docker).
Needs:  SIMBA, LAURA, laura-lattices (JFEL) and Ocelot; docker and
        ghcr.io/astec-stfc/simcodes-docker:master for elegant (skipped
        otherwise).
Output: output/25_collective_effects/<code>_<effect>/.

Units: emittance (normalised) and sizes in um; energy spread in %.

The line is JFEL's Lattices/jfel_linac_codes.def (as 13 and 14): the
250 pC injector beam from S02 to the undulator entrance, through the
chicane at ~120 MeV. Each run turns on one effect (or none) and the beam
is compared at the undulator entrance.

Checks, in each code:
- Wakes (the S- and X-band structures) reduce the chirp into the chicane:
  sigma_z falls by more than 30% (measured 37% in Ocelot, 42% in elegant).
- CSR in the chicane's bends: eps_nx grows more than 2x (measured 5.3x
  Ocelot, 3.1x elegant), and eps_ny, in the non-bending plane, changes
  less than 2% (measured < 0.2%).
- Each run reproduces the JFEL README's "Collective effects" table (eps_nx,
  sigma_x, sigma_z) to 2%, except Ocelot's CSR row. Same input, same code,
  so the match is to the README's rounding; 2% covers version drift.
  Ocelot's CSR row is printed only. Since SIMBA PR #67, Ocelot's CSR runs
  from the first bend to the next cavity, instead of over the whole line
  when the line had a `csr:` block, and eps_nx comes out 4.04 against the
  README's 3.31 (ISSUES.md #29). Ocelot was already the outlier: elegant
  gives 2.33, Bmad 2.00 and ImpactX 1.68 (README).
"""

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
ENTRANCE = "JFEL-FEL-SIM-MARK-01"
EFFECTS = {
    "none": dict(csr_enable=False, lsc_enable=False, wakefield_enable=False),
    "wakes": dict(csr_enable=False, lsc_enable=False, wakefield_enable=True),
    "csr": dict(csr_enable=True, lsc_enable=False, wakefield_enable=False),
    "lsc": dict(csr_enable=False, lsc_enable=True, wakefield_enable=False),
}
README = {  # JFEL README, "Collective effects": eps_nx, sigma_x, sigma_z [um]
    ("elegant", "wakes"): (0.762, 49.5, 48.2), ("ocelot", "wakes"): (0.759, 48.6, 55.3),
    ("elegant", "csr"): (2.33, 109.3, 81.5), ("ocelot", "csr"): (3.31, 154.8, 87.2),
    ("elegant", "lsc"): (0.798, 53.0, 80.4), ("ocelot", "lsc"): (0.871, 56.4, 82.9),
}
STALE = {("ocelot", "csr")}  # ISSUES.md #29


def docker_ready():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


def track(code, effect):
    """JFEL's linac in `code` with one effect on; the beam's summary at the undulator entrance."""
    directory = OUT / f"{code}_{effect}"
    f = fw.Framework(master_lattice=JFEL_DIR, directory=str(directory), clean=True, verbose=False,
                     container_runtime="docker" if code == "elegant" else None)
    f.loadSettings("Lattices/jfel_linac_codes.def")
    f.change_Lattice_Code("Linac", code)
    for switch, value in EFFECTS[effect].items():
        setattr(f["Linac"], switch, value)
    f.track(files=["Linac"], check_lattice=False)
    beam = rbf.load_file(str(directory / f"{ENTRANCE}.openpmd.hdf5"))
    cp = beam.cp.val
    return {"eps_nx": beam.Particles.emittance.enx.val * 1e6, "eps_ny": beam.Particles.emittance.eny.val * 1e6,
            "sigma_x": np.std(beam.x.val) * 1e6, "sigma_z": np.std(beam.z.val) * 1e6,
            "dE": np.std(cp) / np.mean(cp) * 100}


codes = ["ocelot"] + (["elegant"] if docker_ready() else [])
if "elegant" not in codes:
    print(f"elegant skipped: docker or {IMAGE} not available\n")
results = {(code, effect): track(code, effect) for code in codes for effect in EFFECTS}

print(f"{'code':8s} {'effect':6s} {'eps_nx':>7s} {'eps_ny':>7s} {'sigma_x':>8s} {'sigma_z':>8s} {'dE [%]':>7s}   README (eps_nx, sigma_x, sigma_z)")
for (code, effect), r in results.items():
    ref = README.get((code, effect))
    note = "" if ref is None else f"   {ref}" + ("  stale, ISSUES.md #29" if (code, effect) in STALE else "")
    print(f"{code:8s} {effect:6s} {r['eps_nx']:7.3f} {r['eps_ny']:7.3f} {r['sigma_x']:8.1f} {r['sigma_z']:8.1f} {r['dE']:7.3f}{note}")

# --- Checks ----------------------------------------------------------------
problems = []
for code in codes:
    none, wakes, csr = (results[(code, e)] for e in ("none", "wakes", "csr"))
    if wakes["sigma_z"] / none["sigma_z"] > 0.7:
        problems.append(f"{code}: wakes changed sigma_z only {wakes['sigma_z'] / none['sigma_z']:.2f}x")
    if csr["eps_nx"] / none["eps_nx"] < 2:
        problems.append(f"{code}: CSR grew eps_nx only {csr['eps_nx'] / none['eps_nx']:.2f}x")
    if abs(csr["eps_ny"] / none["eps_ny"] - 1) > 0.02:
        problems.append(f"{code}: CSR changed eps_ny by {csr['eps_ny'] / none['eps_ny'] - 1:+.1%}")
for key, ref in README.items():
    if key in results and key not in STALE:
        got = [results[key][q] for q in ("eps_nx", "sigma_x", "sigma_z")]
        worst = max(abs(g / r - 1) for g, r in zip(got, ref))
        if worst > 0.02:
            problems.append(f"{key}: {np.round(got, 3)} against the README's {ref}")
assert not problems, "\n".join(problems)
print(f"\nOK ({', '.join(codes)}): wakes shorten the bunch, CSR grows eps_nx only, and the README's table is reproduced")
