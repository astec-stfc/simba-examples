"""
Helpers shared by 16_groups_chicane.py and 19_dispersion_and_solenoid.py.

- Write a .def that runs JFEL's bunch compressor (the VBC chicane, 5.5 m) as
  a line of its own, from a beam you give, with the four dipoles as a
  `groups:` chicane. The .def lives in output/; its lattice paths are
  relative to the master lattice (JFEL), so it needs no copy of JFEL.
- Load it in a given code at a given angle, with CSR, space charge and
  wakefields off (JFEL asks for them in this section).
- Measure the linear coefficients of a tracked beam against the start.
"""

import os

import numpy as np
import simba.Framework as fw
import yaml
from laura_lattices import JFEL

JFEL_DIR = os.path.dirname(JFEL.__file__)
DIPOLES = [f"JFEL-VBC-MAG-DIP-0{i}" for i in range(1, 5)]
SCREEN = "JFEL-VBC-DIA-SCR-01"  # in the middle straight, between dipoles 2 and 3
START, END = "JFEL-S05-SIM-MARK-01", "JFEL-VBC-SIM-MARK-01"


def write_def(path, beam):
    """
    Write the chicane .def to `path`. `beam` is the starting beam's absolute
    path without `.openpmd.hdf5`: SIMBA adds it, and with the extension
    given it reports a missing file under a wrong name.
    """
    settings = {
        "files": {"BC": {"code": "ocelot", "input": {"particle_definition": str(beam)},
                         "output": {"start_element": START, "end_element": END}}},
        "groups": {"bunch_compressor": {"type": "chicane", "elements": DIPOLES}},
        "layout": "./layouts.yaml", "section": "./sections.yaml", "element_list": "./YAML/",
    }
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def load(def_file, directory, code, angle=None):
    """A Framework with the chicane line in `code`, collective effects off, and (optionally) a new angle."""
    framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(directory), clean=True, verbose=False)
    framework.loadSettings(str(def_file))
    framework.change_Lattice_Code("BC", code)
    line = framework["BC"]
    line.csr_enable = line.lsc_enable = line.wakefield_enable = False
    if angle is not None:
        framework["bunch_compressor"].set_angle(angle)
    return framework


def linear_coefficients(start, values):
    """
    Fit `values` (one per particle) as a + b x0 + c x0' + d delta0 over the
    starting beam, and return (b, c, d): the first-order coefficients R_i1,
    R_i2 and R_i6. Fitting all three at once matters: a sample of N particles
    has a chance correlation ~1/sqrt(N) between x0 and delta0, and the plain
    <v delta>/<delta^2> picks up R_i1 times that.
    """
    x0, xp0 = start.x.val, start.cpx.val / start.cpz.val
    d0 = start.cp.val / np.mean(start.cp.val) - 1
    a = np.c_[x0, xp0, d0, np.ones_like(d0)]
    return np.linalg.lstsq(a, values, rcond=None)[0][:3]


def bend_geometry(framework, angle):
    """(rho, D): the dipoles' bending radius and the z gap from dipole 1's exit to dipole 2's entrance."""
    d1, d2 = framework[DIPOLES[0]], framework[DIPOLES[1]]
    return d1.magnetic.length / angle, d2.physical.start.z - d1.physical.end.z
