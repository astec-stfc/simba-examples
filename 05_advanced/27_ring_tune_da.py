"""
27 - A ring: tunes, closed orbit, dynamic aperture, frequency map
=================================================================

Shows: SIMBA's ring studies. A LAURA section with `geometry: closed` is a
ring. Then, on its line:
- `tracking: {periodic: True}`, track(), and `ring_parameters()` give the
  tunes, periodic Twiss, chromaticity and the closed orbit;
- `tracking: {turns: N, dynamic_aperture: {...}}` and
  `run_dynamic_aperture()` track starts along rays from the origin (as
  elegant's find_aperture) and return (x, y, turns survived);
  `dynamic_aperture_boundary()` gives the last survivor on each ray;
- `run_frequency_map()` gives (x, y, Qx, Qy, diffusion) over a grid.

Run:    python 05_advanced/27_ring_tune_da.py
Tier:   A
Needs:  SIMBA, LAURA, Ocelot, Xsuite, MAD-X (cpymad); nafflib for the
        frequency map's tunes.
Output: output/27_ring_tune_da/<code>/ and dynamic_aperture.png.

Units: tunes in turns; positions in mm; k1l in 1/m, k2l in 1/m^2.

The ring is SIMBA's examples/ring_studies/da_fma.py (05_advanced/ring.py):
ten 4 m FODO cells, a sextupole beside each quad, 1 GeV, no RF. The
reference for the nonlinear checks is ring.py's own tracker: exact
thick-quad matrices and sliced thin sextupoles, symplectic, independent of
SIMBA and the codes.

Checks:
- Tunes: each code's equals the thick-quad matrix tune to 1e-6 (the
  sextupoles do nothing on axis). Measured: identical to 1e-9.
- Closed orbit with QF0 displaced 1 mm: MAD-X's equals the tracker's to
  1e-3 (measured 2.4e-4; MAD-X tracks thin slices). The linear formula
  (I - M)^-1 kick, printed, is 7% off: off axis the sextupoles add
  focusing. XFAIL #30: Xsuite finds no closed orbit (its 6D search, in a
  ring without RF), and so gives no optics at all; `line.twiss(method="4d")`
  gets the tracker's orbit to 4e-7. XFAIL #31: Ocelot reports a zero orbit.
- Dynamic aperture on the +x ray, 512 turns: Xsuite's equals the
  tracker's to one grid step (1.03 mm); measured 16.55 against 16.55 (the
  tracker gives 15.5-16.6 with 4 to 50 slices). elegant agrees too (SIMBA's
  own da_fma.png). XFAIL #32: Ocelot gives 12.4 mm and MAD-X 11.4. For
  Ocelot that's its second-order maps (SecondTM, not symplectic), which
  LAURA chooses for the sextupoles: with kick maps it gives the tracker's
  one-kick value. For MAD-X the cause isn't found; more slices don't
  change it.
- Frequency map: each code's tune at the smallest amplitude (1 mm) equals
  the linear tune to 1e-3 (measured 2-4e-4, the amplitude detuning).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
import ring  # noqa: E402
from simple_lattice import transfer_matrix  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
CODES = ["ocelot", "xsuite", "madx"]
TURNS, OFFSET = 512, 1e-3
DA = {"nx": 30, "ny": 15, "n_lines": 11, "x_max": 0.03, "y_max": 0.015}
X_RAY = DA["x_max"] * np.arange(1, DA["nx"]) / (DA["nx"] - 1)  # the +x ray's starts
STEP = X_RAY[0]


def matrix_tune(plane):
    m = transfer_matrix(ring.elements(), plane, 0.0, ring.CELLS * ring.CELL_LENGTH)
    mu = np.arccos(np.trace(m) / 2)
    return (2 * np.pi - mu if m[0, 1] < 0 else mu) / (2 * np.pi)


def linear_orbit():
    """(I - M)^-1 times QF0's kick, sextupoles off."""
    quads = [e for e in ring.elements() if type(e).__name__ == "Quadrupole"]
    qf0 = quads[0]
    start, end = qf0.physical.middle.z - qf0.physical.length / 2, qf0.physical.middle.z + qf0.physical.length / 2
    m_q = transfer_matrix([qf0], "x", start, end)
    rest = transfer_matrix(quads[1:], "x", end, ring.CELLS * ring.CELL_LENGTH)
    m = rest @ m_q @ transfer_matrix([], "x", 0.0, start)
    kick = rest @ (np.array([OFFSET, 0.0]) - m_q @ np.array([OFFSET, 0.0]))  # u -> M_q (u - d) + d
    return np.linalg.solve(np.eye(2) - m, kick)


tunes = {"matrix": (matrix_tune("x"), matrix_tune("y"))}
orbits, apertures, boundaries, small = {}, {}, {}, {}
for code in CODES:
    f = ring.framework_for(code, OUT / code / "optics", {"periodic": True})
    f.track()
    p = f["RING"].ring_parameters()
    tunes[code] = (p["tune_x"], p["tune_y"])

    f = ring.framework_for(code, OUT / code / "orbit", {"periodic": True})
    f.modifyElement("QF0", "physical.error.position.x", OFFSET)
    f.track()
    orbits[code] = f["RING"].ring_parameters().get("closed_orbit_x")

    f = ring.framework_for(code, OUT / code / "aperture", {"turns": TURNS, "dynamic_aperture": DA})
    line = f["RING"]
    line.preProcess()
    line.write()
    boundaries[code] = line.dynamic_aperture_boundary(line.run_dynamic_aperture())
    apertures[code] = boundaries[code][0][0]  # the first ray is +x
    footprint = line.run_frequency_map()
    small[code] = min(footprint, key=lambda r: r[0] ** 2 + r[1] ** 2)[2:4]

reference_orbit = ring.closed_orbit(OFFSET)[0]
reference_aperture = ring.on_axis_aperture(X_RAY, TURNS)

print(f"\n{'':8s} {'tune_x':>10s} {'tune_y':>10s}   {'orbit x [mm]':>12s}   {'DA +x [mm]':>10s}   {'FMA tunes at 1 mm':>18s}")
for name, (qx, qy) in tunes.items():
    if name == "matrix":
        print(f"{'matrix':8s} {qx:10.7f} {qy:10.7f}   {linear_orbit()[0] * 1e3:12.5f}   {'':10s}   (linear formula)")
        continue
    orbit = "none" if orbits[name] is None else f"{orbits[name] * 1e3:.5f}"
    print(f"{name:8s} {qx:10.7f} {qy:10.7f}   {orbit:>12s}   {apertures[name] * 1e3:10.2f}   "
          f"{small[name][0]:.5f} {small[name][1]:.5f}")
print(f"{'tracker':8s} {'':10s} {'':10s}   {reference_orbit * 1e3:12.5f}   {reference_aperture * 1e3:10.2f}")

plt.figure(figsize=(7, 4))
for code in CODES:
    xy = np.array(boundaries[code]) * 1e3
    plt.plot(xy[:, 0], xy[:, 1], "o-", label=code)
plt.plot([reference_aperture * 1e3], [0], "k*", markersize=12, label="independent tracker (+x)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(f"Dynamic aperture, {TURNS} turns")
plt.legend()
plt.savefig(OUT / "dynamic_aperture.png", dpi=120, bbox_inches="tight")
print(f"\nPlot: {OUT / 'dynamic_aperture.png'}\n")

# --- Checks ----------------------------------------------------------------
problems = []
for code in CODES:
    if max(abs(a - b) for a, b in zip(tunes[code], tunes["matrix"])) > 1e-6:
        problems.append(f"{code}: tunes {tunes[code]}, matrices {tunes['matrix']}")
    if max(abs(a - b) for a, b in zip(small[code], tunes["matrix"])) > 1e-3:
        problems.append(f"{code}: frequency-map tunes at 1 mm {small[code]}, linear {tunes['matrix']}")

if abs(orbits["madx"] / reference_orbit - 1) > 1e-3:
    problems.append(f"madx: closed orbit {orbits['madx']}, tracker {reference_orbit}")
if orbits["xsuite"] is None:
    print("XFAIL #30: Xsuite found no closed orbit (6D search without RF), so no optics with the offset quad")
elif abs(orbits["xsuite"] / reference_orbit - 1) < 1e-3:
    print("XPASS #30: Xsuite finds the closed orbit; check it here and update ISSUES.md")
if orbits["ocelot"] == 0.0:
    print("XFAIL #31: Ocelot reports a zero closed orbit with QF0 displaced 1 mm")
elif abs(orbits["ocelot"] / reference_orbit - 1) < 1e-3:
    print("XPASS #31: Ocelot's closed orbit is right; check it here and update ISSUES.md")

if abs(apertures["xsuite"] - reference_aperture) > STEP * 1.01:
    problems.append(f"xsuite: DA {apertures['xsuite']:.4f}, tracker {reference_aperture:.4f}")
for code in ("ocelot", "madx"):
    if abs(apertures[code] - reference_aperture) > STEP * 1.01:
        print(f"XFAIL #32: {code}'s dynamic aperture on +x is {apertures[code] * 1e3:.2f} mm, "
              f"the tracker's {reference_aperture * 1e3:.2f}")
    else:
        print(f"XPASS #32: {code}'s dynamic aperture now agrees; update ISSUES.md")
assert not problems, "\n".join(problems)
print("OK: tunes from every code, MAD-X's closed orbit and Xsuite's dynamic aperture match the references")
