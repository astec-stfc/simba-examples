"""
Helpers shared by the 02_tracking scripts.

- Build a straight lattice in Python from markers and quadrupoles. SIMBA
  fills the gaps between elements with drifts, so there's no need to write
  them.
- Write it out as a master lattice: the same files as a machine in
  laura-lattices (YAML/, sections.yaml, layouts.yaml) plus a .def holding the
  `files:` block. SIMBA then loads it exactly as it loads JFEL.
- Generate a starting beam with given Twiss parameters.
- Give the linear transfer matrices of such a lattice, worked out here from
  first principles, to check the codes against.

01_basics/02_build_lattice_python.py, 04_the_def_file.py and
05_generate_beam.py show these steps one at a time.
"""

import shutil
from pathlib import Path

import numpy as np
import yaml
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole
from simba.Codes.Generators import frameworkGenerator

AREA = "LINE"  # the one section (LAURA machine_area) of these lattices
M_E = 0.51099895e6  # electron rest energy, eV


def marker(name, z):
    """A marker at z (m). A marker inside a line is a screen: SIMBA writes the beam there."""
    return Marker(name=name, machine_area=AREA, physical={"middle": {"z": z}})


def quad(name, z, k1l, length=0.2):
    """A quadrupole centred at z (m); k1l in 1/m, positive focuses horizontally."""
    return Quadrupole(name=name, machine_area=AREA,
                      physical={"length": length, "middle": {"z": z}},
                      magnetic={"length": length, "k1l": k1l})


def fodo(cells=1, length=4.0, k1l=0.6, quad_length=0.2):
    """
    FODO cells starting and ending in the middle of a focusing quad: a half
    QF (k1l/2) at each end of a cell, a full QD in the middle, and a screen
    halfway along each drift. Markers START and END bound the line.
    """
    half = quad_length / 2
    elements = [marker("START", 0.0)]
    for c in range(cells):
        z0 = c * length
        elements += [
            quad(f"QF{c}A", z0 + half / 2, k1l / 2, half),
            marker(f"SCR{c}A", z0 + length / 4),
            quad(f"QD{c}", z0 + length / 2, -k1l, quad_length),
            marker(f"SCR{c}B", z0 + 3 * length / 4),
            quad(f"QF{c}B", z0 + length - half / 2, k1l / 2, half),
        ]
    return elements + [marker("END", cells * length)]


def write_master_lattice(path, elements, files, name="line.def", **extra):
    """
    Write `elements` (one section, one layout) as a master lattice in `path`,
    with a .def called `name` holding `files` and any other top-level keys
    given as keywords (e.g. global=...). Anything already in `path` is
    deleted first. Returns `name`, ready for Framework.loadSettings.
    """
    path = Path(path)
    shutil.rmtree(path, ignore_errors=True)
    names = [e.name for e in elements]
    sections = {"sections": {AREA: names}}
    layouts = {"default_layout": "line", "layouts": {"line": [AREA]}}
    machine = LAURA(element_list=elements, section=sections, layout=layouts)
    export_machine(path=str(path / "YAML"), machine=machine, overwrite=True)
    (path / "sections.yaml").write_text(yaml.safe_dump(sections, sort_keys=False))
    (path / "layouts.yaml").write_text(yaml.safe_dump(layouts, sort_keys=False))
    settings = {"files": files, **extra, "layout": "./layouts.yaml", "section": "./sections.yaml",
                "element_list": "./YAML/"}
    (path / name).write_text(yaml.safe_dump(settings, sort_keys=False))
    return name


def make_beam(filename, beta=(5.0, 5.0), alpha=(0.0, 0.0), enx=1e-6, eny=1e-6, p0=100e6,
              sigma_z=100e-6, sigma_pz=10e3, particles=10_000, charge=10e-12, seed=1):
    """
    Generate an electron bunch with these Twiss parameters (beta in m),
    normalised emittances (m rad), mean momentum and spread (eV/c), rms
    length (m) and charge (C), and write it to `filename` (openPMD). Returns
    the beam. The generator's 3-sigma cut makes the emittance 2.7% smaller
    than asked (05_generate_beam.py); the Twiss parameters are unchanged.
    """
    def transverse(en, b, a):
        eps = en / (p0 / M_E)
        return np.sqrt(eps * b), p0 * np.sqrt(eps * (1 + a ** 2) / b), -a / np.sqrt(1 + a ** 2)

    sx, spx, cx = transverse(enx, beta[0], alpha[0])
    sy, spy, cy = transverse(eny, beta[1], alpha[1])
    np.random.seed(seed)
    beam = frameworkGenerator(
        code="simba", species="electron", number_of_particles=particles, charge=charge,
        initial_momentum=p0, sigma_x=sx, sigma_px=spx, covariance_xxp=cx,
        sigma_y=sy, sigma_py=spy, covariance_yyp=cy, sigma_z=sigma_z, sigma_pz=sigma_pz,
    ).generate()
    beam.write_openpmd_beam_file(str(filename))
    return beam


def transfer_matrix(elements, plane, z_from, z_to):
    """
    The 2x2 linear transfer matrix (x, x') or (y, y') from z_from to z_to
    through `elements`: thick-lens quadrupoles and drifts, for the reference
    momentum. Positions must not fall inside a quadrupole.
    """
    sign = 1 if plane == "x" else -1
    m, z = np.eye(2), z_from
    quads = sorted((e for e in elements if isinstance(e, Quadrupole)), key=lambda e: e.physical.middle.z)
    for q in quads:
        length = q.physical.length
        start = q.physical.middle.z - length / 2
        if start < z_from - 1e-9 or start + length > z_to + 1e-9:  # 1 nm: rounding at a quad edge
            continue
        m = np.array([[1, start - z], [0, 1]]) @ m
        k = sign * q.magnetic.k1l / length
        r = np.sqrt(abs(k)) * length
        if k > 0:
            mq = [[np.cos(r), np.sin(r) / np.sqrt(k)], [-np.sqrt(k) * np.sin(r), np.cos(r)]]
        elif k < 0:
            mq = [[np.cosh(r), np.sinh(r) / np.sqrt(-k)], [np.sqrt(-k) * np.sinh(r), np.cosh(r)]]
        else:
            mq = [[1, length], [0, 1]]
        m = np.array(mq) @ m
        z = start + length
    return np.array([[1, z_to - z], [0, 1]]) @ m


def sigma_matrix(beam, plane):
    """The 2x2 second-moment matrix of (x, x') or (y, y'), with x' = px/pz."""
    u = getattr(beam, plane).val
    up = getattr(beam, f"cp{plane}").val / beam.cpz.val
    return np.cov(np.vstack([u, up]), bias=True)
