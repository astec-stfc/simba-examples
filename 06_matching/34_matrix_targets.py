"""
34 - Transfer-matrix targets: point-to-point imaging
====================================================

Shows: targets on the transfer matrix. `R11` to `R66` at an element are the
matrix from the start of the line to the end of that element, in Xsuite's
coordinates (x, px, y, py, zeta, delta) whatever the code matches with: each
code's own matrix is converted (Ocelot's tau and MAD-X's t by a sign and a
power of beta0, elegant's path length by its own rule). The block

    match:
      variables: [JFEL-S13-MAG-QUAD-01, ..., -04]
      targets: {JFEL-FEL-SIM-MARK-01: {R12: 0, R34: 0}}

asks for point-to-point imaging from S13 to the undulator entrance: a
particle's position there does not depend on its angle at S13. Xsuite,
elegant and Bmad match on R natively. Ocelot's matcher has no matrix targets,
and MAD-X's MATCH refuses more variables (four quads) than constraints (two),
so for those SIMBA's least-squares solver works on the code's own matrices.
Cheetah takes no matrix targets (SIMBA has no convention for its matrices)
and says so.

Run:    python 06_matching/34_matrix_targets.py
Tier:   A (elegant and Bmad run if installed, as in example 33)
Needs:  SIMBA, LAURA, laura-lattices (JFEL), Ocelot, Xsuite, MAD-X.
Output: output/34_matrix_targets/run/.

Units: R12 and R34 in m (x in m over px in rad); R11, R33 dimensionless.

Two targets on four quads leave a family of solutions, so the codes land on
different strengths (elegant on much stronger quads); each must give imaging.
That is checked by tracking, not with any code's matrix: the Match line is
tracked (in Ocelot, collective effects off) after each match, and each
particle's x and y at the entrance are fitted against its x, px, y, py and
delta at S13. The fit has to carry the chromatic terms (x delta, px delta,
and with delta^2): a fit linear in delta returns the matrix averaged over
the energy spread, R12 + U1266 <delta^2>, which is 0.02 to 0.26 m off here,
most for the strongest quads. With them, the fitted R12 is the on-energy one
the target asks for, and the unmatched one equals the codes' (4.122 m).

Checks:
- After every code's match, the tracked R12 and R34 are below 1e-4 m, against
  4.1 and 2.8 m unmatched (measured: at most 3.5e-5 m, elegant's stopping point;
  1e-5 m or less for the others).
- Cheetah declines, giving its reason.
"""

from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from jfel import ENTRANCE, JFEL_DIR, S13, S13_QUADS

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
RUN = OUT / "run"
CODES = ("ocelot", "xsuite", "madx", "elegant", "bmad", "cheetah")
IMAGING = {ENTRANCE: {"R12": 0.0, "R34": 0.0}}


def tracked_matrix(start, end):
    """
    On-energy R11, R12, R33, R34, fitted from the tracked beam: each particle's x
    and y at the end against its x, px, y, py and delta at the start. The fit
    carries the chromatic terms (x delta, px delta, and their delta^2), or it
    would return the matrix averaged over the energy spread, R12 + U1266 <delta^2>,
    not the R12 the target is for.
    """
    def coordinates(beam):
        return beam.x.val, beam.cpx.val / beam.cpz.val, beam.y.val, beam.cpy.val / beam.cpz.val
    x0, xp0, y0, yp0 = coordinates(start)
    d0 = start.cp.val / np.mean(start.cp.val) - 1
    linear = [x0, xp0, y0, yp0]
    chromatic = [u * d0**n for u in linear for n in (1, 2)]
    a = np.column_stack(linear + [d0, d0**2] + chromatic + [np.ones_like(d0)])
    x1, _, y1, _ = coordinates(end)
    rx = np.linalg.lstsq(a, x1, rcond=None)[0]
    ry = np.linalg.lstsq(a, y1, rcond=None)[0]
    return {"R11": rx[0], "R12": rx[1], "R33": ry[2], "R34": ry[3]}


framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(RUN), clean=True, verbose=False)
framework.loadSettings("Lattices/jfel_match_undulator.def")
for line in ("Linac", "Match"):
    framework[line].csr_enable = False
# the matrix is the lattice's: compare it with a beam that sees only the lattice
framework["Match"].lsc_enable = framework["Match"].wakefield_enable = False
stored = [framework.getElement(q).k1l for q in S13_QUADS]
framework.track(files=["Linac"], check_lattice=False)
start = rbf.load_file(str(RUN / f"{S13}.openpmd.hdf5"))


def match_in(code):
    """Match imaging from half strength in `code`, track, and return the k1l and the tracked R."""
    for q, k1l in zip(S13_QUADS, stored):
        framework.modifyElement(q, "k1l", 0.5 * k1l)
    framework["Match"].file_block["match"].update(targets=IMAGING, code=code)
    framework.track(files=["Match"], check_lattice=False)
    end = rbf.load_file(str(RUN / f"{ENTRANCE}.openpmd.hdf5"))
    return [framework.getElement(q).k1l for q in S13_QUADS], tracked_matrix(start, end)


# Unmatched, at half strength, for scale
framework["Match"].file_block["match"]["enable"] = False
for q, k1l in zip(S13_QUADS, stored):
    framework.modifyElement(q, "k1l", 0.5 * k1l)
framework.track(files=["Match"], check_lattice=False)
unmatched = tracked_matrix(start, rbf.load_file(str(RUN / f"{ENTRANCE}.openpmd.hdf5")))
framework["Match"].file_block["match"]["enable"] = True

results, declined = {}, {}
for code in CODES:
    try:
        results[code] = match_in(code)
    except ValueError as error:
        declined[code] = str(error)
        print(f"SKIP {code}: {error}")

def row(label, k1l, r):
    return (f"{label:10s}" + "".join(f"{v:9.4f}" for v in k1l)
            + f"{r['R11']:9.4f}{r['R12']:10.1e}{r['R33']:9.4f}{r['R34']:10.1e}")


print(f"\n{'':10s}" + "".join(f"{q[-7:]:>9s}" for q in S13_QUADS) + "      R11       R12      R33       R34   (tracked)")
print(row("unmatched", [0.5 * k for k in stored], unmatched))
problems = []
for code, (k1l, r) in results.items():
    print(row(code, k1l, r))
    if abs(r["R12"]) > 1e-4 or abs(r["R34"]) > 1e-4:
        problems.append(code)

# --- Checks ----------------------------------------------------------------
if "cheetah" not in declined:
    problems.append("Cheetah did not decline")
assert not problems, f"failed: {problems}"
print(f"\nOK: {len(results)} codes match point-to-point imaging, confirmed by tracking; Cheetah declines")
