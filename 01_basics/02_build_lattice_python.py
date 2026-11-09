"""
02 - Build a lattice in Python with LAURA
=========================================

Shows: how to make LAURA elements (Marker, Quadrupole, Drift) in Python, put
them in a section and a layout, export the machine to LAURA YAML, and load
that YAML back. The YAML directory is the same format the lattices in
laura-lattices use, so SIMBA can load and track it.

Run:    python 01_basics/02_build_lattice_python.py
Tier:   A
Needs:  LAURA only (no tracking code).
Output: output/02_build_lattice_python/lattice/: layouts.yaml, sections.yaml
        and YAML/ with one file per element (the same layout as a machine in
        laura-lattices).

Units: lengths and positions in m; k1l in 1/m (integrated, k1 * length).
An element's position is the middle of the element (physical.middle.z).

Check: after the round trip through YAML, every element has the same
position, length and k1l, and the section length is the sum of the element
lengths (the drifts fill every gap).
"""

from pathlib import Path

import yaml
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Drift, Marker, Quadrupole

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
AREA = "FODO"  # LAURA's machine_area: which section an element belongs to


def quad(name, z, k1l, length=0.2):
    return Quadrupole(name=name, machine_area=AREA,
                      physical={"length": length, "middle": {"z": z}},
                      magnetic={"length": length, "k1l": k1l})


def drift(name, start, end):
    return Drift(name=name, machine_area=AREA,
                 physical={"length": end - start, "middle": {"z": (start + end) / 2}})


def marker(name, z):
    return Marker(name=name, machine_area=AREA, physical={"middle": {"z": z}})


# One FODO cell, 2 m long: half QF, drift, QD, drift, half QF, written as a
# full QF at each end so the cell is easy to read.
elements = [
    marker("START", 0.0),
    quad("QF1", 0.1, 0.5),
    drift("D1", 0.2, 0.9),
    quad("QD1", 1.0, -0.5),
    drift("D2", 1.1, 1.8),
    quad("QF2", 1.9, 0.5),
    marker("END", 2.0),
]
names = [e.name for e in elements]

machine = LAURA(
    element_list=elements,
    section={"sections": {AREA: names}},
    layout={"default_layout": "line", "layouts": {"line": [AREA]}},
)
print(f"Built section {AREA}: {machine.sections[AREA].names}")

# export_machine writes the elements; the section order and the layouts are
# two small files, written here directly.
lattice = OUT / "lattice"
export_machine(path=str(lattice / "YAML"), machine=machine, overwrite=True)
(lattice / "sections.yaml").write_text(yaml.safe_dump({"sections": {AREA: names}}, sort_keys=False))
(lattice / "layouts.yaml").write_text(
    yaml.safe_dump({"default_layout": "line", "layouts": {"line": [AREA]}}, sort_keys=False))
print(f"Exported to {lattice}:")
for p in sorted(lattice.rglob("*.yaml")):
    print("   ", p.relative_to(lattice))

# Load it back from the YAML files only.
reloaded = LAURA(
    layout=str(lattice / "layouts.yaml"),
    section=str(lattice / "sections.yaml"),
    element_list=str(lattice / "YAML"),
)

# --- Check -----------------------------------------------------------------
assert reloaded.sections[AREA].names == names, reloaded.sections[AREA].names
for original in elements:
    copy = reloaded.elements[original.name]
    assert type(copy) is type(original), (original.name, type(copy))
    assert abs(copy.physical.middle.z - original.physical.middle.z) < 1e-12, original.name
    assert abs(copy.physical.length - original.physical.length) < 1e-12, original.name
    if isinstance(original, Quadrupole):
        assert copy.magnetic.k1l == original.magnetic.k1l, original.name
total = sum(e.physical.length for e in elements)
span = elements[-1].physical.middle.z - elements[0].physical.middle.z
assert abs(total - span) < 1e-12, f"lengths add up to {total} m, markers are {span} m apart"
print(f"\nOK: round trip kept every position, length and k1l; section length {total:.3f} m")
