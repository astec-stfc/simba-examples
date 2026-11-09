"""
19 - Dispersion from a dogleg, rotation in a solenoid
=====================================================

Shows: two linear-optics effects, each measured on a tracked beam and
compared with its formula.

1. Dispersion. The first half of JFEL's chicane (dipoles 1 and 2) is a
   dogleg: it moves the beam sideways by X and leaves it parallel to the
   axis again. A particle with more energy bends less, so it is moved less.
   At the screen in the middle straight, x depends on the energy as
   x = eta delta, and x' doesn't (eta' = 0). Run in Ocelot, Cheetah and
   Xsuite, using 16's chicane .def (chicane.py).
2. A solenoid (LAURA `Solenoid`, built in Python) rotates the beam about
   its axis by S0L / (2 B rho), half the angle K L with K = B / (B rho):
   the Larmor angle. Run in Ocelot, Cheetah, Xsuite and MAD-X.

Run:    python 03_modifying/19_dispersion_and_solenoid.py
Tier:   A
Needs:  SIMBA, LAURA, laura-lattices, Ocelot, Cheetah, Xsuite and MAD-X
        (cpymad).
Output: output/19_dispersion_and_solenoid/: the chicane .def and beam,
        dogleg_<code>/, and solenoid/ (lattice and one run per code).

Units: eta in m (x per unit delta = dp/p); angles in rad; S0L (the
solenoid's integrated field) in T m; B rho in T m.

The dogleg: JFEL's dipoles are rectangular magnets, with both faces square
to the z axis. A particle of momentum p0 (1 + delta) has radius
rho (1 + delta) and crosses the first magnet's z length rho sin(theta)
while turning by theta' with sin(theta') = sin(theta) / (1 + delta). The
second magnet, the same length in z, turns it back by exactly theta', so
every particle leaves parallel to z (eta' = 0 to all orders), displaced by

    X(delta) = 2 rho (1 + delta) (1 - cos(theta')) + D tan(theta')

where D is the z gap between the two magnets. X(0) is where the screen
sits (checked in 16). The codes' x points away from the bend's centre of
curvature, so x = X(0) - X(delta) and

    eta = -dX/ddelta = 2 rho (1/cos(theta) - 1) + D tan(theta) / cos(theta)^2

eta is measured like 16's R56: fit x (and x') at the screen against the
starting x, x' and delta.

The solenoid: 0.4 m, S0L = 0.2 T m, a 100 MeV beam (B rho = 0.3336 T m),
so the rotation is 0.2998 rad. The beam is flat (enx 10 um, eny 0.1 um)
and nearly parallel (beta = 1000 m), so its tilt,
1/2 atan2(2<xy>, <x^2> - <y^2>), is the rotation. The starting beam has a
small tilt of its own from sampling (-0.0026 rad), so measure the change.

Check: in every code, eta must match the formula to 1e-4 relative (the
fit's noise from the delta^2 term; measured 5e-6), and |eta'| must be below
1e-5 (measured 3e-9). The beam's tilt must change by S0L / (2 B rho) to
1e-3 relative, with the same sign in every code (positive S0L turns the
beam towards +y here). The solenoid also focuses, which changes the shape
of a beam that isn't perfectly parallel, and so its tilt a little:
measured 4e-5 at most (Xsuite), 3e-6 in the others.
"""

import sys
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from chicane import SCREEN, bend_geometry, linear_coefficients, load, write_def
from laura.models.element import Solenoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
from simple_lattice import AREA, make_beam, marker, write_master_lattice  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
C = 299792458.0
problems = []

# --- 1. Dispersion at the middle of the chicane ------------------------------
start = make_beam(OUT / "start.openpmd.hdf5", beta=(10.0, 10.0), p0=130e6, sigma_pz=130e3, particles=5000)
def_file = write_def(OUT / "bc.def", OUT / "start")
theta = 0.13
print(f"{'code':8s} {'eta [m]':>9s} {'expected':>9s} {'rel diff':>9s} {'eta prime':>10s}")
for code in ["ocelot", "cheetah", "xsuite"]:
    run = OUT / f"dogleg_{code}"
    f = load(def_file, run, code)
    f.track()
    rho, gap = bend_geometry(f, theta)
    expected = 2 * rho * (1 / np.cos(theta) - 1) + gap * np.tan(theta) / np.cos(theta) ** 2
    screen = rbf.load_file(str(run / f"{SCREEN}.openpmd.hdf5"))
    eta = linear_coefficients(start, screen.x.val)[2]
    eta_prime = linear_coefficients(start, screen.cpx.val / screen.cpz.val)[2]
    print(f"{code:8s} {eta:9.6f} {expected:9.6f} {eta / expected - 1:9.1e} {eta_prime:10.1e}")
    if abs(eta / expected - 1) > 1e-4 or abs(eta_prime) > 1e-5:
        problems.append(f"{code}: eta {eta:.6f} (expected {expected:.6f}), eta' {eta_prime:.1e}")

# --- 2. Rotation in a solenoid -----------------------------------------------
S0L, P0 = 0.2, 100e6
brho = P0 / C
solenoid = Solenoid(name="SOL", machine_area=AREA, magnetic={"length": 0.4, "fields": {"S0L": S0L}},
                    physical={"length": 0.4, "middle": {"z": 0.5}})
files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
lattice = OUT / "solenoid" / "lattice"
settings = write_master_lattice(lattice, [marker("START", 0.0), solenoid, marker("END", 0.7)], files, "sol.def")
flat = make_beam(lattice / "START.openpmd.hdf5", beta=(1000.0, 1000.0), enx=10e-6, eny=0.1e-6, p0=P0,
                 particles=2000)


def tilt(beam):
    """Angle of the beam's long axis in the x-y plane (rad)."""
    x, y = beam.x.val - np.mean(beam.x.val), beam.y.val - np.mean(beam.y.val)
    return 0.5 * np.arctan2(2 * np.mean(x * y), np.mean(x * x) - np.mean(y * y))


larmor = S0L / (2 * brho)
print(f"\nSolenoid: S0L {S0L} T m, B rho {brho:.4f} T m, expected rotation {larmor:.5f} rad "
      f"(starting tilt {tilt(flat):.5f} rad)")
print(f"{'code':8s} {'rotation':>9s} {'rel diff':>9s}")
for code in ["ocelot", "cheetah", "xsuite", "madx"]:
    run = OUT / "solenoid" / code
    f = fw.Framework(master_lattice=str(lattice), directory=str(run), clean=True, verbose=False)
    f.loadSettings(settings)
    f.change_Lattice_Code("LINE", code)
    f.track()
    rotation = tilt(rbf.load_file(str(run / "END.openpmd.hdf5"))) - tilt(flat)
    print(f"{code:8s} {rotation:9.5f} {rotation / larmor - 1:9.1e}")
    if abs(rotation / larmor - 1) > 1e-3:
        problems.append(f"{code}: solenoid rotated the beam by {rotation:.5f} rad, expected {larmor:.5f}")

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print("\nOK: the dogleg's dispersion and the solenoid's rotation match their formulas in every code")
