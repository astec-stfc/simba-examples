"""
Helpers shared by the 06_matching examples.

- JFEL's paths and the names used in the match into the undulator.
- The undulator's periodic Twiss, the target of that match, as example 23
  computes it: a line with `tracking: {periodic: True}` over the undulator,
  in a small .def next to the run, then `ring_parameters()`.
- The Twiss of a beam file, and comparisons between Twiss sets.
"""

import os

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
import yaml
from laura_lattices import JFEL

JFEL_DIR = os.path.dirname(JFEL.__file__)
S13_QUADS = [f"JFEL-S13-MAG-QUAD-0{i}" for i in range(1, 5)]
S13, ENTRANCE, EXIT = "JFEL-S13-SIM-MARK-01", "JFEL-FEL-SIM-MARK-01", "JFEL-FEL-SIM-MARK-02"
KEYS = ("beta_x", "alpha_x", "beta_y", "alpha_y")

L02, L03, L4H = "JFEL-L02-LIN-CAV-01", "JFEL-L03-LIN-CAV-01", "JFEL-L4H-LIN-CAV-01"
VBC_START, VBC_END = "JFEL-S05-SIM-MARK-01", "JFEL-VBC-SIM-MARK-01"
DIPOLES = [f"JFEL-VBC-MAG-DIP-0{i}" for i in range(1, 5)]


def compressor_def(path, code, linac=None, bc=None):
    """
    Write a .def to `path` for JFEL from the saved injector beam to the end of
    the VBC chicane, as two lines in `code`: Linac (S02 to the VBC: L02, L03 and
    the X-band L4H) and BC (the chicane). `linac` and `bc` add to the lines'
    blocks (a match, say). The .def also has

      groups:   linac, the S-band cavities L02 and L03 (an element_group), and
                bunch_compressor, the chicane;
      functional_definitions: {linac_phase: 18.0}, and changes that set L02's
                and L03's phase to it, so one number sets both.
    """
    settings = {
        "files": {
            "Linac": {
                "code": code,
                "input": {
                    "particle_definition": "$master_lattice$Data_Files/jfel_injector_S02",
                    "twiss": {"beta_x": 20, "alpha_x": 0, "beta_y": 10, "alpha_y": 0},
                },
                "output": {"start_element": "JFEL-S02-SIM-APER-01", "end_element": VBC_START},
                **(linac or {}),
            },
            "BC": {
                "code": code,
                "output": {"start_element": VBC_START, "end_element": VBC_END},
                **(bc or {}),
            },
        },
        "groups": {
            "linac": {"type": "element_group", "elements": [L02, L03]},
            "bunch_compressor": {"type": "chicane", "elements": DIPOLES},
        },
        "functional_definitions": {"linac_phase": 18.0},
        "changes": {L02: {"cavity": {"phase": "linac_phase"}}, L03: {"cavity": {"phase": "linac_phase"}}},
        "layout": "./layouts.yaml", "section": "./sections.yaml", "element_list": "./YAML/",
    }
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def longitudinal_shape(path):
    """
    (energy at the bunch centre [eV], [chirp, curvature, skewness]) of the beam
    in `path`: a fifth-order fit of energy against s = c(<t> - t), the head at
    positive s, so the RF's own fourth and fifth orders do not leak into the
    first three. The three are the coefficients of s, s^2, s^3 in (E - E0)/E0.
    """
    from scipy.constants import c

    beam = rbf.load_file(str(path))
    t, energy = np.asarray(beam.t.val), np.asarray(beam.energy.val)
    e = np.polyfit(c * (t.mean() - t), energy, 5)[::-1]
    return float(e[0]), e[1:4] / e[0]


def periodic_target(out, beam):
    """
    The undulator's periodic Twiss, for a beam of the energy of `beam` (an
    absolute path without `.openpmd.hdf5`). Runs in `out`/periodic.
    """
    periodic_def = out / "periodic.def"
    periodic_def.write_text(yaml.safe_dump({
        "files": {"Periodic": {
            "code": "ocelot",
            "input": {"particle_definition": str(beam)},
            "tracking": {"periodic": True},
            "output": {"start_element": ENTRANCE, "end_element": EXIT},
        }},
        "layout": "./layouts.yaml", "section": "./sections.yaml", "element_list": "./YAML/",
    }, sort_keys=False))
    periodic = fw.Framework(master_lattice=JFEL_DIR, directory=str(out / "periodic"), clean=True, verbose=False)
    periodic.loadSettings(str(periodic_def))
    periodic.track()
    ring = periodic["Periodic"].ring_parameters()
    return {k: float(ring[k]) for k in KEYS}


def beam_twiss(path):
    """beta and alpha in both planes of the beam in `path` (an .openpmd.hdf5 file)."""
    t = rbf.load_file(str(path)).twiss
    return {k: float(np.ravel(getattr(t, k).val)[0]) for k in KEYS}


def differences(got, want):
    """Largest relative beta difference and absolute alpha difference."""
    beta = max(abs(got[k] / want[k] - 1) for k in ("beta_x", "beta_y"))
    alpha = max(abs(got[k] - want[k]) for k in ("alpha_x", "alpha_y"))
    return beta, alpha
