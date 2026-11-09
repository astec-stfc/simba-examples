"""
15 - Read and change element settings
=====================================

Shows: the ways to read and change an element's settings once a lattice is
loaded, and how to save the changes to a file and apply them again later:

- `framework["QD0"]` is the LAURA element; `.k1l` reads its strength.
- `getElement(name, param)` and `modifyElement(name, param, value)` read and
  set one parameter. `param` can be a dotted path into the element
  ("magnetic.k1l", "physical.error.position.x").
- `getElementType(type, param)` and `setElementType(type, param, values)` do
  the same for every element of a type.
- `apply_changes({name: {param: value}})` sets several at once.
- `save_changes_file(filename=...)` writes every change made since loading,
  old and new values. `load_changes_file(filename)` applies them to a fresh
  lattice.

Run:    python 03_modifying/15_modify_elements.py
Tier:   A
Needs:  SIMBA, LAURA and Ocelot.
Output: output/15_modify_elements/: changes.yaml, the lattice, and the run
        (LINE.py is the Ocelot deck SIMBA wrote).

Units: k1l in 1/m (k1 = k1l / length in 1/m^2); x in m; x' = px/pz (rad).

getElementType returns the elements in the order they were loaded, which is
not lattice order (here QD0 before QF0A). setElementType pairs its values
with that same order, so build the values from getElementType, as here,
rather than from a list in lattice order.

The script flips the sign of every quadrupole in 02_tracking's three FODO
cells, saves that as a changes file, loads it into a new Framework and
tracks.

Check: the strength SIMBA wrote into the Ocelot deck must be the new k1l
divided by the length, to 1e-12. Flipping every quadrupole swaps the
focusing planes, so the tracked horizontal sigma matrix at END must be
M_y Sigma0 M_y^T, with M_y the original lattice's vertical matrix, to 1e-4
relative (the same limit and reason as 09). The reloaded lattice must hold
exactly the values saved, and the changes file must list only the changed
quadrupoles.
"""

import re
import sys
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
from simple_lattice import fodo, make_beam, sigma_matrix, transfer_matrix, write_master_lattice  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"

elements = fodo(cells=3)
files = {
    "LINE": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
settings = write_master_lattice(LATTICE, elements, files, "fodo.def")
start = make_beam(LATTICE / "START.openpmd.hdf5", beta=(5.0, 5.0), alpha=(0.0, 0.0))

framework = fw.Framework(master_lattice=str(LATTICE), directory=str(OUT / "edit"), clean=True, verbose=False)
framework.loadSettings(settings)

# --- Reading -----------------------------------------------------------------
print(f'framework["QD0"]: a LAURA {type(framework["QD0"]).__name__}, k1l = {framework["QD0"].k1l}')
print(f'getElement("QD0", "k1l") = {framework.getElement("QD0", "k1l")}')
names = [e["name"] for e in framework.getElementType("quadrupole")]
print(f'getElementType("quadrupole") order: {names}  (load order, not lattice order)')

# --- Changing one element ----------------------------------------------------
framework.modifyElement("QD0", "k1l", -0.5)
print(f'modifyElement("QD0", "k1l", -0.5): k1l = {framework["QD0"].k1l}')
framework.modifyElement("QD0", "magnetic.k1l", -0.6)  # the same parameter, by its full path
print(f'modifyElement("QD0", "magnetic.k1l", -0.6): k1l = {framework["QD0"].k1l}')
framework.apply_changes({"QD1": {"k1l": -0.5}, "QD2": {"k1l": -0.5}})
print(f'apply_changes: QD1, QD2 k1l = {framework["QD1"].k1l}, {framework["QD2"].k1l}')
framework.apply_changes({"QD1": {"k1l": -0.6}, "QD2": {"k1l": -0.6}})  # back, before the flip

# --- Changing every quadrupole: flip the sign --------------------------------
k1l = framework.getElementType("quadrupole", "k1l")
framework.setElementType("quadrupole", "k1l", [-v for v in k1l])  # paired with getElementType's order
flipped = dict(zip(names, framework.getElementType("quadrupole", "k1l")))
print(f"flipped: {flipped}")

changes_file = OUT / "changes.yaml"
framework.save_changes_file(filename=str(changes_file))
changes = yaml.safe_load(changes_file.read_text())
print(f"\n{changes_file.name}, first entry:\n" + yaml.safe_dump({"QD0": changes["QD0"]}))

# --- A fresh lattice with the changes applied, tracked -----------------------
run = OUT / "run"
reloaded = fw.Framework(master_lattice=str(LATTICE), directory=str(run), clean=True, verbose=False)
reloaded.loadSettings(settings)
reloaded.load_changes_file(str(changes_file))
got_k1l = dict(zip(names, reloaded.getElementType("quadrupole", "k1l")))
reloaded.track()

deck = (run / "LINE.py").read_text()
deck_k1 = {m[1].upper(): float(m[0]) for m in re.findall(r"k1=([-\d.e]+), eid='(\w+)'", deck)}
lengths = {e.name: e.physical.length for e in elements if e.name in names}
end = rbf.load_file(str(run / "END.openpmd.hdf5"))
m_y = transfer_matrix(elements, "y", 0.0, elements[-1].physical.middle.z)  # the original lattice, y plane
expected = m_y @ sigma_matrix(start, "x") @ m_y.T
got = sigma_matrix(end, "x")
scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
worst = np.max(np.abs(got - expected) / scale)
print(f"Deck strengths: {deck_k1}")
print(f"sigma_x at END: {np.sqrt(got[0, 0]) * 1e6:.2f} um; M_y Sigma0 M_y^T gives "
      f"{np.sqrt(expected[0, 0]) * 1e6:.2f} um; worst difference {worst:.1e} (relative)")

# --- Check -----------------------------------------------------------------
quads = {e.name: e.magnetic.k1l for e in elements if e.name in names}
assert all(flipped[n] == -quads[n] for n in names), f"setElementType didn't flip every quad: {flipped}"
assert got_k1l == flipped, f"load_changes_file gave {got_k1l}, saved {flipped}"
assert set(changes) == set(names), f"changes file lists {sorted(changes)}, changed {sorted(names)}"
for name in names:
    assert abs(deck_k1[name] - flipped[name] / lengths[name]) < 1e-12, \
        f"{name}: deck k1 {deck_k1[name]}, expected {flipped[name] / lengths[name]}"
assert worst < 1e-4, f"flipped lattice's sigma_x differs from M_y Sigma0 M_y^T by {worst:.1e}"
print("\nOK: changes reach the deck, survive a save and reload, and flipping the quads swaps the planes")
