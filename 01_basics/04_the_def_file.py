"""
04 - The lattice definition (.def) file
=======================================

Shows: what each part of a .def file does. 01_basics/jfel_example.def is a
commented .def for JFEL with two lines in two codes (Cheetah to just after
the chicane, then Ocelot to the undulator entrance), a starting beam file, a
Twiss rematch, a chicane group, a generator and inline changes. This script
loads it and prints what SIMBA made of each part. Read the .def alongside.

Run:    python 01_basics/04_the_def_file.py
Tier:   A
Needs:  SIMBA, LAURA and laura-lattices (nothing is tracked).
Output: on screen.

Check: the lines, their codes and ends are as written in the .def; the
starting beam file named with $master_lattice$ exists; the `changes:` block
has changed the elements from their lattice-file values; the chicane group
holds its four dipoles.
"""

import os
from pathlib import Path

import simba.Framework as fw
import yaml
from laura import LAURA
from laura_lattices import JFEL

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output" / Path(__file__).stem
DEF = HERE / "jfel_example.def"

framework = fw.Framework(master_lattice=os.path.dirname(JFEL.__file__), directory=str(OUT),
                         generator_defaults="jfel.yaml",  # for the generator: block
                         clean=True, verbose=False)
framework.loadSettings(str(DEF))  # a path; a bare name is looked up in the master lattice
written = yaml.safe_load(DEF.read_text())

print("files: (one line per entry, tracked in order)")
for name in written["files"]:
    line = framework[name]
    print(f"  {name:10s} code {line.code:8s} {line.start} -> {line.end}")
    print(f"             space charge: {line.space_charge_mode or 'off'}")

first = framework["Compressor"]
beam_file = first.read_input_file(first.prefix, written["files"]["Compressor"]["input"]["particle_definition"],
                                  read_file=False)
print(f"\nCompressor's starting beam: {beam_file}")
print(f"  rematched to: {first.getInitialTwiss()}")
print("Main has no input: it starts from Compressor's output file, "
      f"{framework['Compressor'].end}.openpmd.hdf5")

chicane = framework["bunch_compressor"]
print(f"\ngroups: bunch_compressor ({type(chicane).__name__}) with {chicane.elements}")
print(f"  angle {chicane.angle:.4f} rad, R56 {chicane.r56 * 1e3:.2f} mm")

gen = framework.generator
print(f"\ngenerator: {type(gen).__name__}, {gen.charge * 1e12:.0f} pC, "
      f"{gen.number_of_particles} particles, flat-top {gen.plateau_bunch_length * 1e12:.1f} ps")

lattice_only = LAURA(lattice=JFEL)  # the same elements without the .def's changes
print("\nchanges: (lattice file -> after the .def)")
for element, params in written["changes"].items():
    for param, value in params.items():
        before = getattr(lattice_only.elements[element], param)
        print(f"  {element}.{param}: {before} -> {getattr(framework[element], param)}")

# --- Check -----------------------------------------------------------------
for name, block in written["files"].items():
    line = framework[name]
    assert line.code == block["code"], (name, line.code)
    assert (line.start, line.end) == (block["output"]["start_element"], block["output"]["end_element"]), name
assert framework["Main"].start == framework["Compressor"].end, "Main must start where Compressor ends"
assert os.path.isfile(beam_file), f"starting beam not found: {beam_file}"
for element, params in written["changes"].items():
    for param, value in params.items():
        assert getattr(framework[element], param) == value, (element, param)
        assert getattr(lattice_only.elements[element], param) != value, f"{element}.{param} was already {value}"
assert chicane.elements == written["groups"]["bunch_compressor"]["elements"]
print("\nOK: lines, codes, ends, starting beam, changes and group are as the .def says")
