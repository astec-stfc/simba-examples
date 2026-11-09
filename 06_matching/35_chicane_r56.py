"""
35 - Match a chicane's R56 with its angle
=========================================

Shows: a group as a match variable. JFEL's VBC chicane is a `groups:`
chicane (example 16); its `angle` is a variable of its own,

    match:
      variables: [bunch_compressor.angle]
      targets: {JFEL-VBC-SIM-MARK-01: {R56: ...}}

and the match sets it through the group: `set_angle` lays the four dipoles
and everything between them out again at every angle tried, so the model is
the chicane as it would be built. A group whose name is given with a field
it does not have (an `element_group`'s members' `k1l`, say) means each member's.

The dipoles' edge angles follow: JFEL writes them as `angle` (dipole 1's exit,
dipole 2's entrance, ...), which LAURA evaluates from each bend's angle when
it writes the lattice. An edge written as a number would stay fixed, and the
match would warn so.

The match sets the angle in the settings' `groups` too (`dipoleangle`), so a
line built later, or the saved settings, keep it.

The script runs the VBC on its own, from a chirp-free 130 MeV beam (as in
example 16), starting at its design 0.13 rad, and matches it to the R56 the
chicane has at 0.12 rad, in each code: by default in the line's own code.
A Cheetah line hands over, as Cheetah takes no matrix targets.

Run:    python 06_matching/35_chicane_r56.py
Tier:   A (elegant and Bmad run if installed, as in example 33)
Needs:  SIMBA, LAURA, laura-lattices (JFEL), Ocelot, Xsuite, MAD-X, Cheetah.
Output: output/35_chicane_r56/: bc.def, start.openpmd.hdf5, one run per code.

Units: angle in rad; R56 in m (z in m, delta = dp/p: Xsuite's zeta and delta).

The target is example 16's exact R56 at 0.12 rad: the path-length R56 of a
rectangular chicane with the arc and gap set_angle gives there, plus the
velocity term S/gamma^2.

Checks, in every code that runs:
- The matched angle is 0.12 rad to 1e-4 (R56 goes as angle^2, so this is
  the target R56 to 2e-4).
- The tracked R56 (example 16's fit) equals the target to 2e-4.
- The screen in the middle straight is where the chicane at that angle puts it.
- Dipole 1's exit edge is still `angle`, and the settings' dipoleangle is the angle.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import simba.Modules.Beams as rbf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "03_modifying"), str(ROOT / "02_tracking")]
from chicane import DIPOLES, END, SCREEN, bend_geometry, linear_coefficients, load, write_def  # noqa: E402
from simple_lattice import M_E, make_beam  # noqa: E402

OUT = ROOT / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
CODES = ("ocelot", "xsuite", "madx", "elegant", "bmad", "cheetah")
TARGET_ANGLE = 0.12
LIBTAO = os.path.join(os.environ.get("ACC_ROOT_DIR", ""), "production", "lib", "libtao.so")

start = make_beam(OUT / "start.openpmd.hdf5", beta=(10.0, 10.0), p0=130e6, sigma_pz=130e3, particles=5000)
gamma = np.mean(start.cp.val) / M_E
def_file = write_def(OUT / "bc.def", OUT / "start")


def exact_r56(framework, angle):
    """Example 16's R56 for the chicane as laid out at `angle` in `framework`."""
    rho, gap = bend_geometry(framework, angle)
    s = framework["BC"].getSValues()
    r56_path = 2 * (2 * rho * (np.tan(angle) - angle) + gap * np.tan(angle) ** 2 / np.cos(angle))
    return r56_path + (s[-1] - s[0]) / gamma**2


target = exact_r56(load(def_file, OUT / "target", "ocelot", angle=TARGET_ANGLE), TARGET_ANGLE)
print(f"target R56 {target:.6f} m (the chicane's at {TARGET_ANGLE} rad)")

rows, problems, handover = [], [], []
for code in CODES:
    if code == "bmad" and not os.path.isfile(LIBTAO):
        print("SKIP bmad: tracking needs $ACC_ROOT_DIR/production/lib/libtao.so (export ACC_ROOT_DIR)")
        continue
    run = OUT / code
    framework = load(def_file, run, code)  # at the design angle, 0.13 rad
    if code == "bmad":
        framework["BC"].libtao = LIBTAO  # as example 32
    framework["BC"].file_block["match"] = {
        "variables": ["bunch_compressor.angle"],
        "targets": {END: {"R56": target}},
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            framework.track()
    except ValueError as error:
        print(f"SKIP {code}: {error}")
        continue
    handover += [str(w.message) for w in caught if "cannot match this" in str(w.message)]
    group = framework["bunch_compressor"]
    angle = group.angle
    end = rbf.load_file(str(run / f"{END}.openpmd.hdf5"))
    r56 = linear_coefficients(start, end.z.val - start.z.val)[2]
    rho, gap = bend_geometry(framework, angle)
    x_screen = framework[SCREEN].physical.middle.x
    x_expected = 2 * rho * (1 - np.cos(angle)) + gap * np.tan(angle)
    kept = framework.settings["groups"]["bunch_compressor"].get("dipoleangle")
    edge = framework[DIPOLES[0]].magnetic.exit_edge_angle
    rows.append((code, angle, r56, x_screen, x_expected, kept, edge))
    if abs(angle / TARGET_ANGLE - 1) > 1e-4:
        problems.append(f"{code}: angle {angle}")
    if abs(r56 / target - 1) > 2e-4:
        problems.append(f"{code}: tracked R56 {r56:.6f}")
    if abs(x_screen - x_expected) > 1e-6:
        problems.append(f"{code}: screen x {x_screen:.6f}, expected {x_expected:.6f}")
    if kept is None or abs(kept - angle) > 1e-12 or edge != "angle":
        problems.append(f"{code}: dipoleangle {kept}, exit edge {edge!r}")

print(f"\n{'line':8s} {'angle':>9s} {'R56 tracked':>12s} {'rel diff':>9s} {'screen x':>9s} {'laid out':>9s} "
      f"{'dipoleangle':>11s} {'D1 exit edge':>12s}")
for code, angle, r56, x, xe, kept, edge in rows:
    print(f"{code:8s} {angle:9.6f} {r56:12.6f} {r56 / target - 1:9.1e} {x:9.6f} {xe:9.6f} {kept:11.6f} {edge!s:>12s}")
print("\n" + (handover[0] if handover else "no hand-over warning"))

# --- Checks ----------------------------------------------------------------
if not handover:
    problems.append("the Cheetah line did not hand over")
assert not problems, "\n".join(problems)
print(f"\nOK: {len(rows)} lines match the chicane's R56 through its angle; the layout, edges and settings follow")
