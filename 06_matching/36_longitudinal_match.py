"""
36 - Set the linac RF for a chirp, and linearise it with the X-band
==================================================================

Shows: `longitudinal_match`, after Zagorodnov and Dohlus (PRST-AB 14, 014403,
2011). From the line's input beam, it sets a main RF system and a harmonic of
it so the beam leaves with a given energy gain and a given chirp, curvature
and skewness of its relative energy along the bunch:

    longitudinal_match:
      cavities: linac            # a group: JFEL's S-band L02 and L03
      harmonics: [JFEL-L4H-LIN-CAV-01]   # the X-band, 4x the frequency
      chirp: -15.0               # [1/m], the coefficient of s in (E - E0)/E0
      curvature: 0.0             # [1/m^2]: linearised
      skewness: 0.0              # [1/m^3]
      energy_gain: ...           # [eV] over the line; by default what its cavities give now

with s = c(<t> - t), the head at positive s. The equations are linear in
V cos(phi) and V sin(phi) of each system; each system's voltage is shared
between its cavities as they share it now, and all its cavities take one
phase. Travelling-wave cavities (all three here) are set by their gradient,
through LAURA's voltage factor.

L02 and L03 are linked as JFEL's operators might link them: the .def
(jfel.py) sets both phases to a functional definition, `linac_phase`. The
match sets that definition, not the cavities, so they stay linked.

The method's model is a rigid, ultra-relativistic beam and the RF alone.
JFEL enters L02 at 35 MeV, where the bunch slips ~1.5 deg against the wave
over a 4 m structure, and the structure wakes take energy from the tail: so
one analytic pass is close, not exact. The script then closes the loop as
the method is used in practice: track, move the targets by what was missed,
match again. Three passes bring the tracked beam onto the targets without
wakes; with them, which change as the RF reshapes the bunch, four. Tracking is in Xsuite, which applies the RF to each
slice of the cavities exactly (Cheetah's and Ocelot's cavity maps are second
order, so they give the curvature approximately and the skewness not at all).

Run:    python 06_matching/36_longitudinal_match.py
Tier:   A
Needs:  SIMBA, LAURA, laura-lattices (JFEL) and Xsuite.
Output: output/36_longitudinal_match/: nominal/, rf/ and wakes/.

Units: energy in MeV; chirp 1/m, curvature 1/m^2, skewness 1/m^3 (of (E-E0)/E0 in s).

Checks, after the passes, without wakes and with them:
- The tracked chirp is the target to 0.2%, the curvature and skewness are
  below 1% of what the nominal RF leaves (109 1/m^2, 6.3e5 1/m^3 at L4H's
  -184 deg), and the energy at the bunch centre is the nominal one to 1e-4.
- L02 and L03 still read `linac_phase`, which the match changed, and keep
  their nominal gradient ratio (30 : 27 MV/m).
- The first, analytic pass is within 5% in chirp and has cut the skewness
  by more than ten.
"""

from pathlib import Path

import numpy as np
import simba.Framework as fw
from jfel import JFEL_DIR, L02, L03, L4H, VBC_START, compressor_def, longitudinal_shape

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
TARGET = np.array([-15.0, 0.0, 0.0])  # chirp, curvature, skewness
NAMES = ("chirp", "curvature", "skewness")
PASSES = 4


def framework_for(name, wakes):
    framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(OUT / name), clean=True, verbose=False)
    framework.loadSettings(str(compressor_def(OUT / f"{name}.def", "xsuite")))
    line = framework["Linac"]
    line.csr_enable = line.lsc_enable = False
    line.wakefield_enable = wakes
    return framework


def track(framework, name):
    framework.track(files=["Linac"])
    return longitudinal_shape(OUT / name / f"{VBC_START}.openpmd.hdf5")


# The nominal RF, without wakes: the energy to keep, and the curvature to beat
e_in = longitudinal_shape(Path(JFEL_DIR) / "Data_Files" / "jfel_injector_S02.openpmd.hdf5")[0]
e_nominal, nominal = track(framework_for("nominal", wakes=False), "nominal")
print(f"nominal: {e_nominal / 1e6:.3f} MeV, " + ", ".join(f"{n} {v:.4g}" for n, v in zip(NAMES, nominal)))

problems, first = [], {}
for name, wakes in (("rf", False), ("wakes", True)):
    framework = framework_for(name, wakes)
    asked, gain = TARGET.copy(), e_nominal - e_in
    print(f"\n{name}: {'pass':>4s} {'E [MeV]':>9s} " + "".join(f"{n:>11s}" for n in NAMES)
          + f" {'linac_phase':>11s} {'L4H phase':>9s}")
    for n in range(1, PASSES + 1):
        framework["Linac"].file_block["longitudinal_match"] = {
            "cavities": "linac", "harmonics": [L4H], "energy_gain": float(gain),
            **dict(zip(NAMES, map(float, asked))),
        }
        energy, shape = track(framework, name)
        if n == 1:
            first[name] = shape
        phase = framework.settings["functional_definitions"]["linac_phase"]
        print(f"{'':6s} {n:4d} {energy / 1e6:9.3f} " + "".join(f"{v:11.4g}" for v in shape)
              + f" {phase:11.3f} {framework.getElement(L4H).cavity.phase:9.3f}")
        asked, gain = asked + (TARGET - shape), gain + (e_nominal - energy)  # move the targets by the miss

    if abs(shape[0] / TARGET[0] - 1) > 2e-3:
        problems.append(f"{name}: chirp {shape[0]:.4f}")
    if abs(shape[1]) > 0.01 * abs(nominal[1]) or abs(shape[2]) > 0.01 * abs(nominal[2]):
        problems.append(f"{name}: curvature {shape[1]:.3f} or skewness {shape[2]:.3e}")
    if abs(energy / e_nominal - 1) > 1e-4:
        problems.append(f"{name}: energy {energy / 1e6:.3f} MeV")
    cavities = [framework.getElement(c) for c in (L02, L03)]
    if any(c.cavity.phase != "linac_phase" for c in cavities):
        problems.append(f"{name}: L02/L03 lost their linac_phase")
    ratio = cavities[0].simulation.field_amplitude / cavities[1].simulation.field_amplitude
    if abs(ratio / (30 / 27) - 1) > 1e-9:
        problems.append(f"{name}: L02:L03 gradient ratio {ratio}")

# --- Checks ----------------------------------------------------------------
if abs(first["rf"][0] / TARGET[0] - 1) > 0.05 or abs(first["rf"][2]) > 0.1 * abs(nominal[2]):
    problems.append(f"first analytic pass: {first['rf']}")
assert not problems, "\n".join(problems)
print(f"\nOK: {PASSES} passes put the beam on the chirp, linearised, at the nominal energy, with and without wakes")
