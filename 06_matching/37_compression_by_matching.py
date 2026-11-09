"""
37 - Set up JFEL's bunch compression by matching
================================================

Shows: the RF match and a chicane match working together, for a bunch
compressed by a chosen factor. A chirp h (the coefficient of s in the
relative energy, s = c(<t> - t)) through a chicane of R56 (Xsuite's
convention: zeta, delta) takes s to (1 + h R56) s, so a compression factor
C needs

    R56 = (1/C - 1) / h.

The chicane also has T566 (z gains T566 delta^2), so a beam whose energy is
linear in s leaves it curved in s. To compress linearly the RF must give the
beam the curvature that cancels it, h2 = -T566 h^2 / R56: "linearise the
compression, not the energy". The script

1. matches the VBC's angle (`bunch_compressor.angle`) for that R56, from
   the nominal beam;
2. measures the chicane's T566 by tracking through it (a fit of each
   particle's change in z against delta and delta^2);
3. matches the linac RF (L02 and L03, linked by `linac_phase`, and the
   X-band L4H) to the chirp h and the curvature h2, closing the loop by
   tracking as example 36 does;
4. tracks the chicane and checks the compression, against the same RF
   matched to curvature 0 instead.

Everything is tracked in Xsuite with wakes and CSR off, so the only effects
are the RF and the lattice, as the formulas assume.

The compression is read from the particles: each one's s after the chicane
fitted against its s before, s_after = a1 s + a2 s^2 + ... (fifth order, at
the bunch centre, like the match's coefficients). 1/a1 is the compression
and a2 what the chicane leaves curved: R56 h2 + T566 h^2, zero when the RF's
curvature cancels T566. At C = 4 that curvature barely shows in the RMS
length (it adds in quadrature to a much larger linear term); it shows in a2,
which sets how the current profile leans.

Run:    python 06_matching/37_compression_by_matching.py
Tier:   A
Needs:  SIMBA, LAURA, laura-lattices (JFEL) and Xsuite.
Output: output/37_compression_by_matching/: one run per stage.

Units: chirp 1/m, curvature 1/m^2; R56 m, T566 m; sigma_z mm; a2 1/m.

Checks:
- The matched angle gives the target R56 (the tracked fit) to 1e-3.
- The compression is C: 1/a1 to 2e-3, the RMS ratio to 2%.
- With the curvature target, a2 is under 2% of what it is with the energy
  only linearised (measured 0.1 against -17.0 1/m); and that -17.0 is the
  predicted R56 h2 + T566 h^2 to 2%.
"""

from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from jfel import JFEL_DIR, L4H, VBC_END, VBC_START, compressor_def, longitudinal_shape
from scipy.constants import c

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
C_TARGET, CHIRP = 4.0, -15.0
R56 = (1 / C_TARGET - 1) / CHIRP
PASSES = 3


def framework_for(name):
    framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(OUT / name), clean=True, verbose=False)
    framework.loadSettings(str(compressor_def(OUT / f"{name}.def", "xsuite")))
    for line in ("Linac", "BC"):
        framework[line].csr_enable = framework[line].lsc_enable = framework[line].wakefield_enable = False
    return framework


def beam(name, element):
    return rbf.load_file(str(OUT / name / f"{element}.openpmd.hdf5"))


def chicane_terms(start, end):
    """R56 and T566 of the tracked chicane: the change in z (ahead positive) fitted against delta and delta^2."""
    delta = start.cp.val / np.mean(start.cp.val) - 1
    dz = (end.z.val - np.mean(end.z.val)) - (start.z.val - np.mean(start.z.val))
    a = np.c_[delta, delta**2, start.x.val, start.cpx.val / start.cpz.val, np.ones_like(delta)]
    r56, t566 = np.linalg.lstsq(a, dz, rcond=None)[0][:2]
    return r56, t566


def sigma_z(b):
    return np.std(c * np.asarray(b.t.val))


def mapping(start, end):
    """
    a1, a2 of s_after = a1 s + a2 s^2 + ..., particle by particle, with
    s = c(<t> - t): the compression (1/a1) and what is left curved (a2), at the
    bunch centre. Fitted to fifth order, as the match's own coefficients are: a
    cubic fit would fold the bunch's fourth order (large over 0.46 mm at X-band)
    into a2.
    """
    s0 = c * (np.mean(start.t.val) - np.asarray(start.t.val))
    s1 = c * (np.mean(end.t.val) - np.asarray(end.t.val))
    coefficients = np.polyfit(s0, s1, 5)[::-1]
    return coefficients[1], coefficients[2]


# 1. The chicane's angle for R56, from the nominal beam
name = "chicane"
framework = framework_for(name)
framework.track(files=["Linac"])
framework["BC"].file_block["match"] = {"variables": ["bunch_compressor.angle"], "targets": {VBC_END: {"R56": R56}}}
framework.track(files=["BC"])
angle = framework["bunch_compressor"].angle
# 2. Its R56 and T566, by tracking
r56, t566 = chicane_terms(beam(name, VBC_START), beam(name, VBC_END))
curvature = -t566 * CHIRP**2 / r56
print(f"target R56 {R56:.6f} m: angle {angle:.6f} rad, tracked R56 {r56:.6f} m, T566 {t566:.6f} m")
print(f"curvature to cancel T566: {curvature:.2f} 1/m^2")


def compress(name, curvature):
    """Match the RF to the chirp and `curvature` (closing the loop), set the angle, track the chicane: the compression."""
    framework = framework_for(name)
    framework.track(files=["Linac"])
    e_target = longitudinal_shape(OUT / name / f"{VBC_START}.openpmd.hdf5")[0]
    e_in = longitudinal_shape(Path(JFEL_DIR) / "Data_Files" / "jfel_injector_S02.openpmd.hdf5")[0]
    target = np.array([CHIRP, curvature, 0.0])
    asked, gain = target.copy(), e_target - e_in
    for _ in range(PASSES):
        framework["Linac"].file_block["longitudinal_match"] = {
            "cavities": "linac", "harmonics": [L4H], "energy_gain": float(gain),
            "chirp": float(asked[0]), "curvature": float(asked[1]), "skewness": float(asked[2]),
        }
        framework.track(files=["Linac"])
        energy, shape = longitudinal_shape(OUT / name / f"{VBC_START}.openpmd.hdf5")
        asked, gain = asked + (target - shape), gain + (e_target - energy)
    framework["bunch_compressor"].set_angle(angle)
    framework.track(files=["BC"])
    before, after = beam(name, VBC_START), beam(name, VBC_END)
    a1, a2 = mapping(before, after)
    print(f"{name:12s} {shape[0]:8.3f} {shape[1]:10.2f} {sigma_z(before) * 1e3:9.4f} {sigma_z(after) * 1e3:9.4f} "
          f"{sigma_z(before) / sigma_z(after):7.3f} {1 / a1:7.3f} {a2:9.3f}")
    return sigma_z(before) / sigma_z(after), a1, a2, shape[1]


print(f"\n{'RF matched':12s} {'chirp':>8s} {'curvature':>10s} {'sigma_z0':>9s} {'sigma_z1':>9s} "
      f"{'C (rms)':>7s} {'1/a1':>7s} {'a2 [1/m]':>9s}")
compression, a1, a2, _ = compress("compensated", curvature)
_, _, a2_plain, h2_plain = compress("linearised", 0.0)
predicted = r56 * h2_plain + t566 * CHIRP**2
print(f"\nleft curved with curvature 0: a2 = {a2_plain:.3f} 1/m, predicted R56 h2 + T566 h^2 = {predicted:.3f} 1/m")

# --- Checks ----------------------------------------------------------------
problems = []
if abs(r56 / R56 - 1) > 1e-3:
    problems.append(f"tracked R56 {r56:.6f}")
if abs(compression / C_TARGET - 1) > 0.02 or abs(1 / a1 / C_TARGET - 1) > 2e-3:
    problems.append(f"compression {compression:.3f} (rms), {1 / a1:.3f} (mapping)")
if abs(a2) > 0.02 * abs(predicted):
    problems.append(f"compensated, still curved: a2 {a2:.3f}")
if abs(a2_plain / predicted - 1) > 0.02:
    problems.append(f"curvature 0: a2 {a2_plain:.3f}, predicted {predicted:.3f}")
assert not problems, "\n".join(problems)
print(f"OK: compressed by {compression:.2f} (target {C_TARGET}) and linearly; with the energy merely "
      f"linearised the chicane's T566 leaves a2 = {a2_plain:.1f} 1/m")
