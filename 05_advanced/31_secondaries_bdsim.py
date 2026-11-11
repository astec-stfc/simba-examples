"""
31 - Secondaries in BDSIM: a target, a converter, a bent crystal
================================================================

Shows: beam-matter lines in BDSIM (Geant4) from LAURA `Target` and
`Crystal` elements. A line's `tracking:` block picks the Geant4
`physics_list`, the `secondaries` to keep and extra `samplers`; the
secondaries come out as their own species in the same openPMD file,
`beam.others[species]`, next to the primaries. A next line can take one
species on to another code (`input: {species: positron}`).

Run:    python 05_advanced/31_secondaries_bdsim.py
Tier:   C
Needs:  SIMBA, LAURA, laura-lattices (SECONDARY), Xsuite; BDSIM on PATH
        with ROOT's Python bindings and pybdsim (source ROOT's thisroot.sh
        and BDSIM's bdsim.sh first). Prints SKIP and exits 0 without them.
Output: output/31_secondaries_bdsim/<lattice>/.

Units: yields per incoming electron; mean energies in MeV; angles in mrad.

laura-lattices' three SECONDARY lines, 2000 macroparticles each (~1 min in
all):
- brems_target: 100 MeV e- through 1 mm of tungsten (0.29 X0), sampled at
  the target's exit;
- positron_conversion: 1 GeV e- shower in 3.5 mm of tungsten (1 X0); the
  positrons alone go on to a screen in Xsuite;
- crystal_channelling: 855 MeV e- on 30 um of bent Si(220), 0.9 mrad.

Geant4's seed changes every run, so each check allows for the statistics
of 2000 macroparticles.

Checks (measured values from one run, against SIMBA and LAURA main):
- brems_target, at the target's exit: the electrons' mean energy is
  E0 exp(-t/X0), 75.2 MeV, to 3% (the formula is the mean radiation loss;
  measured 75.3). Electrons per electron 0.989 to 0.01 (README; a binomial
  sigma is 0.002; measured 0.987). Photons per electron 2.03 and positrons
  0.038, the README's SIMBA row, to 5 sigma of a Poisson count over 2000
  (+-0.16, +-0.022; measured 2.08, 0.033).
- positron_conversion, at the converter's exit: positrons per electron in
  the README's range over runs, 0.84-0.87, widened by its quoted seed
  spread of +-0.02 (0.82-0.89; measured 0.834); photons 11.7-11.9 the same
  way, 0.2 for the spread (11.5-12.1; measured 11.89). The Xsuite screen
  gets the positrons, and the photons carried through untracked.
- crystal_channelling: the fraction within 0.15 mrad of the 0.9 mrad bend
  (channelled through the whole crystal) is 0.085 to 5 sigma, as the
  yields (+-0.035). 0.085 and sigma = 0.0077 are the mean and spread of
  15 runs, 0.066-0.097 (the binomial sigma for 2000 is 0.0062; the 0.066
  run, 2.4 sigma low, failed an earlier 3-sigma band). The README has
  0.093 for SIMBA and 0.082 for its hand-written gmad, one run each.
  Amorphous scattering alone gives 0.004.
"""

import importlib.util
import os
import shutil
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import SECONDARY

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
SECONDARY_DIR = os.path.dirname(SECONDARY.__file__)
E = 1.602176634e-19
X0_W = 3.504e-3  # tungsten radiation length [m] (PDG)


def species(path):
    """{species: (real particles, <E> [MeV], x' [mrad] array)} in a SIMBA beam file."""
    beam = rbf.load_file(str(path))
    out = {}
    for name, b in {beam.species: beam, **beam.others}.items():
        n = np.broadcast_to(np.asarray(b.Particles.nmacro.val, dtype=float), (len(b.x),))
        out[name] = (n.sum(), np.average(np.asarray(b.energy.val) / 1e6, weights=n), np.asarray(b.xp.val) * 1e3)
    return out


def run(lattice):
    """Track `lattice`; the species at each sampler and line end, per incoming electron."""
    directory = OUT / lattice
    f = fw.Framework(master_lattice=SECONDARY_DIR + "/", directory=str(directory), clean=True, verbose=False)
    f.loadSettings(f"Lattices/{lattice}.def")
    f.track()
    lines = [line for name, line in f.latticeObjects.items() if name != "generator"]
    stem = lines[0].file_block["input"]["particle_definition"].replace("$master_lattice$", SECONDARY_DIR + "/")
    electrons = abs(float(rbf.load_file(stem + ".openpmd.hdf5").Particles.total_charge)) / E
    points = {}
    for line in lines:
        for name in list((line.file_block.get("tracking") or {}).get("samplers") or []) + [line.end]:
            points[name] = {s: (n / electrons, e, xp) for s, (n, e, xp) in species(directory / f"{name}.openpmd.hdf5").items()}
    return electrons, points


if not shutil.which("bdsim") or importlib.util.find_spec("pybdsim") is None or importlib.util.find_spec("ROOT") is None:
    print("SKIP: needs bdsim on PATH, ROOT's Python bindings and pybdsim (source thisroot.sh and bdsim.sh)")
else:  # not sys.exit(0): in a notebook that is an error and stops "Run All"
    results = {lattice: run(lattice) for lattice in ("brems_target", "positron_conversion", "crystal_channelling")}
    for lattice, (electrons, points) in results.items():
        print(f"\n{lattice} (incoming: {electrons:.3g} electrons, as 2000 macroparticles)")
        for point, by_species in points.items():
            for s, (n, e, xp) in by_species.items():
                print(f"  {point:22s} {s:9s} {n:7.3f} per e-   <E> {e:8.2f} MeV   x' median {np.median(xp):7.3f} mrad")

    brems = results["brems_target"][1]["SEC-BREMS-TGT-01"]
    conv_points = results["positron_conversion"][1]
    conversion, screen = conv_points["SEC-CONV-SIM-END-01"], conv_points["SEC-CONV-DIA-SCR-01"]
    crystal_xp = results["crystal_channelling"][1]["SEC-CRY-DIA-SCR-01"]["electron"][2]
    channelled = np.mean(np.abs(crystal_xp - 0.9) < 0.15)
    expected_energy = 100.0 * np.exp(-1e-3 / X0_W)
    print(f"\nbrems: electrons' <E> {brems['electron'][1]:.2f} MeV, E0 exp(-t/X0) {expected_energy:.2f} MeV")
    print(f"crystal: channelled fraction {channelled:.3f} (mean of 15 runs 0.085; README 0.093)")

    # --- Checks ------------------------------------------------------------
    problems = []
    checks = [  # (what, got, low, high)
        ("brems e- <E> [MeV]", brems["electron"][1], expected_energy * 0.97, expected_energy * 1.03),
        ("brems e- per e-", brems["electron"][0], 0.979, 0.999),
        ("brems photons per e-", brems["photon"][0], 2.03 - 0.16, 2.03 + 0.16),
        ("brems e+ per e-", brems.get("positron", (0,))[0], 0.038 - 0.022, 0.038 + 0.022),
        ("conversion e+ per e-", conversion["positron"][0], 0.82, 0.89),
        ("conversion photons per e-", conversion["photon"][0], 11.5, 12.1),
        ("crystal channelled", channelled, 0.085 - 0.035, 0.085 + 0.035),
    ]
    for what, got, low, high in checks:
        if not low <= got <= high:
            problems.append(f"{what}: {got:.4f}, expected {low:.4f} to {high:.4f}")
    tracked = [s for s in screen if s != "photon"]  # photons are carried through untracked
    if tracked != ["positron"]:
        problems.append(f"the Xsuite screen has {tracked}, expected the positrons alone")
    assert not problems, "\n".join(problems)
    print("\nOK: the radiation loss, the yields and the channelled fraction are as expected")
