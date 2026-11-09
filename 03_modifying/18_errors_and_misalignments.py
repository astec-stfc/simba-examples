"""
18 - Misaligned quadrupoles and random errors
=============================================

Shows: moving an element off axis with `physical.error.position.{x,y}`
(also `rotation.{theta,phi,psi}`, and `.z`), set with modifyElement, and
what it does to the beam's centroid:

1. One quadrupole offset, in Ocelot, Cheetah, Xsuite and MAD-X.
2. Random offsets on every quadrupole of three FODO cells, drawn from a
   seeded numpy generator: the orbit they give, and that the same seed
   gives the same orbit.
3. SIMBA's own random errors, `loadElementErrors` and `setSeedValue`.
   They are written into elegant's `&error_element`, so they work only in
   elegant, and only for strengths (k1l, ...), not positions. On SIMBA
   main they fail with KeyError for every element (ISSUES.md #25); this
   part runs only when the simcodes image is there, and prints XFAIL.

Run:    python 03_modifying/18_errors_and_misalignments.py
Tier:   A (part 3 needs tier B)
Needs:  SIMBA, LAURA, Ocelot, Cheetah, Xsuite and MAD-X (cpymad); docker
        with the simcodes image for part 3.
Output: output/18_errors_and_misalignments/: one run directory per case.

Units: offsets and x in m; x' = px/pz (rad); k1l in 1/m.

A quadrupole displaced by d acts on x - d: its centroid map is
u -> Mq (u - d) + d, with u = (x, x') and Mq the quad's matrix. For a thin
quad that is a kick k1l * d. With drifts D1 before and D2 after,
u_end = D2 [Mq (D1 u0 - d) + d]. Through many quads, apply that one quad at
a time (orbit() below).

Check: in each code the beam's centroid at END must follow the map above,
in x and y, to 1e-3 of the centroid's change. The maps are first order and
use the reference momentum, so the beam's energy spread and second-order
terms make a small difference (measured 4e-6 in Ocelot, Cheetah and
Xsuite); MAD-X tracks a sliced, thin-lens copy of the quad, which gives
5e-5. For the random offsets (Ocelot, 50 um rms), the orbit at every screen
must follow the map to 1e-3 of the largest orbit offset (measured 6e-6); a
second run with the same seed must
give exactly the same orbit, and another seed an orbit that differs by
more than 10% of it.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura.models.element import Quadrupole

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
from simple_lattice import fodo, make_beam, marker, quad, transfer_matrix, write_master_lattice  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"


def docker_ready():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


def orbit(elements, offsets, plane, u0, z_to):
    """Centroid (u, u') at z_to from u0 at z = 0, with quadrupole `name` displaced by offsets[name]."""
    u, z = np.asarray(u0, dtype=float), 0.0
    quads = sorted((e for e in elements if isinstance(e, Quadrupole)), key=lambda e: e.physical.middle.z)
    for q in quads:
        start = q.physical.middle.z - q.physical.length / 2
        end = start + q.physical.length
        if end > z_to + 1e-9:
            break
        d = np.array([offsets.get(q.name, 0.0), 0.0])
        u = transfer_matrix(elements, plane, z, start) @ u  # the drift up to the quad
        u = transfer_matrix(elements, plane, start, end) @ (u - d) + d
        z = end
    return transfer_matrix(elements, plane, z, z_to) @ u


def centroid(beam, plane):
    return np.array([np.mean(getattr(beam, plane).val), np.mean(getattr(beam, f"cp{plane}").val / beam.cpz.val)])


def framework_for(lattice, settings, run, code, **kwargs):
    f = fw.Framework(master_lattice=str(lattice), directory=str(run), clean=True, verbose=False, **kwargs)
    f.loadSettings(settings)
    f.change_Lattice_Code("LINE", code)
    return f


files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
problems = []

# --- 1. One quad, offset -------------------------------------------------------
lattice = OUT / "one_quad" / "lattice"
elements = [marker("START", 0.0), quad("Q1", 0.5, 0.5), marker("END", 2.5)]
settings = write_master_lattice(lattice, elements, files, "quad.def")
start = make_beam(lattice / "START.openpmd.hdf5", particles=2000)
offset = {"x": 1e-4, "y": -2e-4}
print(f"Q1 (k1l = 0.5 /m) offset by x {offset['x']:.0e} m, y {offset['y']:.0e} m; "
      f"thin-lens kick k1l*dx = {0.5 * offset['x']:.1e} rad")
print(f"{'code':8s} {'x_end [um]':>11s} {'expected':>9s} {'y_end [um]':>11s} {'expected':>9s} {'worst rel':>9s}")
for code in ["ocelot", "cheetah", "xsuite", "madx"]:
    run = OUT / "one_quad" / code
    f = framework_for(lattice, settings, run, code)
    for plane, d in offset.items():
        f.modifyElement("Q1", f"physical.error.position.{plane}", d)
    f.track()
    end = rbf.load_file(str(run / "END.openpmd.hdf5"))
    row, worst = f"{code:8s}", 0.0
    for plane, d in offset.items():
        u0 = centroid(start, plane)
        expected = orbit(elements, {"Q1": d}, plane, u0, 2.5)
        aligned = orbit(elements, {}, plane, u0, 2.5)
        got = centroid(end, plane)
        worst = max(worst, abs(got[0] - expected[0]) / abs(expected[0] - aligned[0]))
        row += f" {got[0] * 1e6:11.3f} {expected[0] * 1e6:9.3f}"
    print(row + f" {worst:9.1e}")
    if worst > 1e-3:
        problems.append(f"{code}: centroid differs from the offset-quad map by {worst:.1e} of its change")

# --- 2. Random offsets on every quad, seeded ---------------------------------------
lattice = OUT / "fodo" / "lattice"
elements = fodo(cells=3)
settings = write_master_lattice(lattice, elements, files, "fodo.def")
start = make_beam(lattice / "START.openpmd.hdf5", particles=2000)
names = [e.name for e in elements if isinstance(e, Quadrupole)]
screens = [e for e in elements if e.hardware_class == "Marker" and e.name != "START"]


def random_orbit(seed, tag):
    """Offset every quad by a seeded random amount (50 um rms), track, return the orbit at each screen."""
    rng = np.random.default_rng(seed)
    offsets = {plane: dict(zip(names, rng.normal(0.0, 50e-6, len(names)))) for plane in ("x", "y")}
    f = framework_for(lattice, settings, OUT / "fodo" / tag, "ocelot")
    for plane in ("x", "y"):
        for name, d in offsets[plane].items():
            f.modifyElement(name, f"physical.error.position.{plane}", float(d))
    f.track()
    got, expected = [], []
    for screen in screens:
        beam = rbf.load_file(str(OUT / "fodo" / tag / f"{screen.name}.openpmd.hdf5"))
        for plane in ("x", "y"):
            got.append(centroid(beam, plane)[0])
            expected.append(orbit(elements, offsets[plane], plane, centroid(start, plane), screen.physical.middle.z)[0])
    return np.array(got), np.array(expected)


orbit_1, expected_1 = random_orbit(1, "seed1")
orbit_1b, _ = random_orbit(1, "seed1_again")
orbit_2, _ = random_orbit(2, "seed2")
scale = np.max(np.abs(expected_1))
worst = np.max(np.abs(orbit_1 - expected_1)) / scale
print(f"\nRandom offsets, seed 1: largest orbit offset {scale * 1e6:.1f} um; worst difference from the map "
      f"{worst:.1e} of it")
print(f"Same seed again: largest difference {np.max(np.abs(orbit_1b - orbit_1)):.1e} m; "
      f"seed 2: {np.max(np.abs(orbit_2 - orbit_1)) * 1e6:.1f} um")
if worst > 1e-3:
    problems.append(f"random offsets: orbit differs from the map by {worst:.1e} of its size")
if not np.array_equal(orbit_1b, orbit_1):
    problems.append("the same seed gave a different orbit")
if np.max(np.abs(orbit_2 - orbit_1)) < 0.1 * scale:
    problems.append("seed 2 gave nearly the same orbit as seed 1")

# --- 3. SIMBA's random errors (elegant only) ------------------------------------------
if docker_ready():
    f = framework_for(lattice, settings, OUT / "elegant_errors", "elegant", container_runtime="docker")
    f.loadElementErrors({"elements": {"QD0": {"k1l": {"amplitude": 0.05, "fractional": 1}}}, "nruns": 2, "seed": 1})
    try:
        f.track()
    except KeyError as exc:
        if exc.args != ("Quadrupole",):
            raise
        print("\nXFAIL #25: loadElementErrors in elegant raises KeyError 'Quadrupole'")
    else:
        print("\nXPASS #25: loadElementErrors ran in elegant; check the seeds here and update ISSUES.md")
else:
    print("\nPart 3 skipped: no simcodes image (elegant is needed for loadElementErrors)")

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print("\nOK: offset quads steer the beam as the maps say, in every code; the same seed gives the same orbit")
