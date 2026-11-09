"""
16 - A chicane as a group: change its angle, measure R56
========================================================

Shows: a `.def`'s `groups:` block, which ties elements together so they can
be changed as one. A `chicane` group of four dipoles has `.angle`, an
`.r56` estimate, and `set_angle(a)`, which sets all four bends and moves
every element between the dipoles onto the new path. The script runs
JFEL's bunch compressor on its own (chicane.py writes the .def) at its
design angle, 0.13 rad, and at 0.10 rad, in Ocelot, Cheetah and Xsuite.

Run:    python 03_modifying/16_groups_chicane.py
Tier:   A
Needs:  SIMBA, LAURA, laura-lattices, Ocelot, Cheetah and Xsuite.
Output: output/16_groups_chicane/: bc.def, start.openpmd.hdf5, and one run
        per code and angle (ocelot_0.13/, ...).

Units: angle in rad; R56 in m (z in m, delta = dp/p); positions in m.

R56 is measured from a chirp-free 130 MeV beam (energy spread 1e-3, no
z-energy correlation): fit each particle's change in z (positive = further
ahead) against its starting x, x' and delta (chicane.linear_coefficients).
Higher energy bends less, takes the shorter path and gets ahead, so R56 is
positive in SIMBA's convention. The exact path-length R56 of a rectangular
chicane (dipoles of arc L and radius rho = L/theta, gap D along z between
dipoles 1 and 2, and 3 and 4) is

    R56_path = 2 [2 rho (tan(theta) - theta) + D tan(theta)^2 / cos(theta)]

The codes also include the velocity term: over the line's length S a
particle with more energy is faster by delta/gamma^2, so the tracked
R56 = R56_path + S/gamma^2 (8.5e-5 m of 0.0563 m here).

The group's `.r56` property is the small-angle formula
2 theta^2 (D + 2L/3), 1.1% below the exact value at 0.13 rad.

set_angle also changes the geometry: in the middle straight the screen
moves sideways to X = 2 rho (1 - cos(theta)) + D tan(theta). On SIMBA main,
set_angle takes each dipole's length as its zero-angle (straight) length,
while JFEL and every code take it as the arc. So set_angle at the current
angle lengthens the dipoles by theta/sin(theta) - 1 (0.28% at 0.13 rad):
ISSUES.md #27. The checks below use the lattice as it is after set_angle,
so they hold either way.

Check: in every code and at both angles, the tracked R56 must equal
R56_path + S/gamma^2 to 2e-4 relative. The only noise is the second-order
term T566 delta^2 (T566 ~ -1.5 R56), which the fit sees as scatter of
~3e-5 relative for 5000 particles; the measured difference is at most 7e-6. The
screen's x must be X to 1e-6 m (rounding in the layout; measured 1e-9).
`.r56` must be within 2% of R56_path (the small-angle error).
"""

import sys
from pathlib import Path

import numpy as np
import simba.Modules.Beams as rbf
from chicane import DIPOLES, END, SCREEN, bend_geometry, linear_coefficients, load, write_def

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
from simple_lattice import M_E, make_beam  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
CODES = ["ocelot", "cheetah", "xsuite"]

start = make_beam(OUT / "start.openpmd.hdf5", beta=(10.0, 10.0), p0=130e6, sigma_pz=130e3, particles=5000)
gamma = np.mean(start.cp.val) / M_E
def_file = write_def(OUT / "bc.def", OUT / "start")

# --- The group -----------------------------------------------------------------
framework = load(def_file, OUT / "inspect", "ocelot")
group = framework["bunch_compressor"]
print(f"bunch_compressor: {type(group).__name__} of {list(group.elements)}")
print(f"angle {group.angle} rad, r56 property {group.r56:.6f} m")

# ISSUES.md #27: set_angle at the angle it already has should change nothing
arc = framework[DIPOLES[0]].magnetic.length
group.set_angle(group.angle)
if abs(framework[DIPOLES[0]].magnetic.length / arc - 1) > 1e-9:
    print(f"XFAIL #27: set_angle({group.angle}) changed the dipole arc from {arc:.6f} m "
          f"to {framework[DIPOLES[0]].magnetic.length:.6f} m")
else:
    print("XPASS #27: set_angle at the current angle keeps the dipoles; remove this check and the note in ISSUES.md")

# --- Track at two angles ----------------------------------------------------------
print(f"\n{'code':8s} {'angle':>6s} {'R56 tracked':>12s} {'expected':>10s} {'rel diff':>9s} "
      f"{'r56 prop':>9s} {'screen x':>9s} {'expected':>9s}")
problems = []
for angle in (0.13, 0.10):
    for code in CODES:
        run = OUT / f"{code}_{angle}"
        f = load(def_file, run, code, angle=None if angle == 0.13 else angle)
        f.track()
        rho, gap = bend_geometry(f, angle)
        s = f["BC"].getSValues()
        r56_path = 2 * (2 * rho * (np.tan(angle) - angle) + gap * np.tan(angle) ** 2 / np.cos(angle))
        expected = r56_path + (s[-1] - s[0]) / gamma ** 2
        end = rbf.load_file(str(run / f"{END}.openpmd.hdf5"))
        r56 = linear_coefficients(start, end.z.val - start.z.val)[2]
        x_screen = f[SCREEN].physical.middle.x
        x_expected = 2 * rho * (1 - np.cos(angle)) + gap * np.tan(angle)
        prop = f["bunch_compressor"].r56
        rel = r56 / expected - 1
        print(f"{code:8s} {angle:6.2f} {r56:12.6f} {expected:10.6f} {rel:9.1e} {prop:9.6f} "
              f"{x_screen:9.6f} {x_expected:9.6f}")
        if abs(rel) > 2e-4:
            problems.append(f"{code} at {angle}: R56 {r56:.6f}, expected {expected:.6f}")
        if abs(x_screen - x_expected) > 1e-6:
            problems.append(f"{code} at {angle}: screen x {x_screen:.6f}, expected {x_expected:.6f}")
        if abs(prop / r56_path - 1) > 0.02:
            problems.append(f"{code} at {angle}: r56 property {prop:.6f}, R56_path {r56_path:.6f}")

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print("\nOK: R56 and the chicane's geometry follow the exact formulas at both angles, in every code")
