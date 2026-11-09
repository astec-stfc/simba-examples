"""
03 - Load a real machine
========================

Shows: how to point SIMBA at a machine from laura-lattices (JFEL, an example
S-band FEL linac), load one of its lattice definition (.def) files, and look
at what was loaded: the lines (one per simulation code run), the elements in
a line with their s positions and types, and one element's settings. It also
loads the same machine with LAURA alone, without SIMBA.

Run:    python 01_basics/03_load_machine.py
Tier:   A
Needs:  SIMBA, LAURA and laura-lattices (no tracking code; nothing is tracked).
Output: tables on screen. SIMBA's working directory is
        output/03_load_machine/ (empty here: nothing is run).

Units: s and lengths in m; k1l in 1/m; dipole angles in rad.

Things that catch people out:
- `framework["NAME"]` gives a line if NAME is a line, otherwise an element.
- `element.length` is ambiguous for most elements (there is a physical,
  a magnetic and sometimes a field-map length); write
  `element.physical.length`. Magnet strengths such as `element.k1l` can be
  read directly.
- SIMBA adds drifts between elements when it builds a line; their names are
  `<line>_drift_<n>`.

Check: the SIMBA line holds the same elements, in the same order, as LAURA's
layout between the line's start and end elements; and SIMBA's s positions
agree with LAURA's arc lengths up to the chicane. Downstream of the chicane
they differ by exactly the chicane's extra path length (see ISSUES.md: the
JFEL section files state the straight-line s there).
"""

import os
from collections import Counter
from pathlib import Path

import simba.Framework as fw
from laura import LAURA
from laura_lattices import JFEL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
JFEL_DIR = os.path.dirname(JFEL.__file__)

# master_lattice is the machine's folder; .def files and the data they name
# ($master_lattice$...) are looked up relative to it.
framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(OUT), clean=True, verbose=False)
framework.loadSettings("Lattices/jfel_combined.def")

print(f"Lines in {framework.settingsFilename}:")
for name in framework.lines:
    line = framework[name]
    if name == "generator":
        print(f"  {name:12s} {line.code:8s} (makes the bunch at the cathode)")
    else:
        print(f"  {name:12s} {line.code:8s} {line.start} -> {line.end}")

line = framework["Linac"]
s = line.getSValues(as_dict=True, at_entrance=True)  # includes SIMBA's drifts
elements = {n: e for n, e in line.elements.items()}  # no drifts
print(f"\nLinac: {len(elements)} elements, {len(s) - len(elements)} drifts added by SIMBA, "
      f"{list(s.values())[-1]:.3f} m long")
print("Element types:", dict(Counter(type(e).__name__ for e in elements.values()).most_common()))

print("\nFirst quadrupoles and cavities (s of the entrance):")
for n, e in list(elements.items()):
    if type(e).__name__ in ("Quadrupole", "RFCavity") and s[n] < 12:
        what = f"k1l {e.k1l:+.3f} /m" if type(e).__name__ == "Quadrupole" else \
            f"phase {e.phase:+.1f} deg, field {e.field_amplitude / 1e6:.1f} MV/m"
        print(f"  {s[n]:8.3f}  {n:24s} L {e.physical.length:.3f} m  {what}")

q = framework["JFEL-S02-MAG-QUAD-01"]
print(f"\n{q.name}: type {q.hardware_type}, section {q.machine_area}, k1l {q.k1l} /m, "
      f"length {q.physical.length} m, middle z {q.physical.middle.z} m")
print("All quadrupole k1l in the machine (first 5):",
      dict(zip(framework.getElementType("quadrupole", "name")[:5],
               framework.getElementType("quadrupole", "k1l")[:5])))

# The same machine with LAURA alone: `lattice=JFEL` takes the paths from the module.
machine = LAURA(lattice=JFEL)
layout = machine.lattices[machine.default_path]
print(f"\nLAURA: {len(machine.elements)} elements, layouts {list(machine.lattices)}, "
      f"default {machine.default_path}")
print(f"Sections of {machine.default_path}: {' '.join(layout.names)}")

# --- Check -----------------------------------------------------------------
# elements_between takes (end, start) positionally: use keywords.
laura_names = layout.elements_between(start=line.start, end=line.end)
assert list(elements) == laura_names, "SIMBA's line and LAURA's layout list different elements"

arc = layout.arc_lengths()  # entrance of each element along the layout
s0 = arc[line.start]
dipoles = [e for e in elements.values() if type(e).__name__ == "Dipole"]
first, last = dipoles[0], dipoles[-1]
# Chicane path length (SIMBA's s) minus its length along z (from the dipole positions).
path = s[last.name] + last.physical.length - s[first.name]
span = last.physical.end.z - first.physical.start.z
excess = path - span
print(f"\nChicane: path {path:.5f} m, along z {span:.5f} m, excess {excess * 1e3:.2f} mm")
names = list(elements)
# The offset starts with the first section after the chicane's own (VBC).
after = max(i for i, n in enumerate(names) if elements[n].machine_area == last.machine_area) + 1
downstream = set(names[after:])
for n, e in elements.items():
    # Diagnostics are thin (zero length) in SIMBA's lines: compare the rest.
    if type(e).__name__ not in ("Marker", "Quadrupole", "Dipole", "RFCavity"):
        continue
    expected = arc[n] - s0 + (excess if n in downstream else 0.0)
    assert abs(s[n] - expected) < 1e-6, f"{n}: SIMBA s {s[n]:.6f} m, LAURA {expected:.6f} m"
print(f"OK: {len(laura_names)} elements in the same order; s agrees with LAURA "
      f"(downstream of the chicane, after adding its {excess * 1e3:.2f} mm excess)")
