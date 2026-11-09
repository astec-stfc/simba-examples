"""
The small ring of 27 (SIMBA's examples/ring_studies/da_fma.py), and an
independent tracker for it.

Ten 4 m FODO cells with a sextupole beside each quadrupole, 1 GeV. The
ring is a LAURA section with `geometry: closed`, which SIMBA treats as a
ring (periodic optics, turns).

The tracker (`one_turn`, `on_axis_aperture`, `closed_orbit`) knows nothing
of SIMBA: exact thick-quad matrices, sextupoles as SLICES drift-kick-drift
slices, paraxial and on momentum. It is symplectic, so it is a fair
reference for long-term tracking.
"""

import numpy as np
import simba.Framework as fw
from laura import LAURA
from laura.exporters.yaml_exporter import export_machine
from laura.models.element import Marker, Quadrupole, Sextupole
from scipy.optimize import fsolve
from simba.Codes.Generators import frameworkGenerator

CELLS, CELL_LENGTH, ENERGY = 10, 4.0, 1e9
# per cell: (name, class, centre [m], length [m], k1l [1/m] or k2l [1/m^2])
CELL = [("QF", Quadrupole, 0.15, 0.3, 0.7), ("SF", Sextupole, 0.45, 0.2, 8.0),
        ("QD", Quadrupole, 2.15, 0.3, -0.6), ("SD", Sextupole, 2.45, 0.2, -8.0)]
SLICES = 10


def element(cls, name, z, length=0.0, strength=None):
    physical = {"length": length, "middle": {"x": 0.0, "y": 0.0, "z": z}}
    if cls is Marker:
        return Marker(name=name, machine_area="RING", hardware_class="Marker", physical=physical)
    key = "k1l" if cls is Quadrupole else "k2l"
    return cls(name=name, machine_area="RING", physical=physical, magnetic={"length": length, key: strength})


def elements():
    ring = [element(Marker, "START", 0.0)]
    for n in range(CELLS):
        ring += [element(cls, f"{name}{n}", n * CELL_LENGTH + z, length, k) for name, cls, z, length, k in CELL]
    return ring + [element(Marker, "END", CELLS * CELL_LENGTH)]


def framework_for(code, directory, tracking):
    """The ring as one line in `code`, with this `tracking:` block, ready to track."""
    ring = elements()
    section = {"sections": {"RING": {"elements": [e.name for e in ring], "geometry": "closed",
                                     "reference_energy": ENERGY}}}
    machine = LAURA(element_list=ring, layout={"default_layout": "ring", "layouts": {"ring": ["RING"]}},
                    section=section)
    export_machine(path=str(directory / "lattice"), machine=machine, overwrite=True)
    settings = fw.FrameworkSettings()
    settings.files = {"RING": {"code": code, "output": {"start_element": "START", "end_element": "END"},
                               "tracking": tracking}}
    settings.layout, settings.section, settings.element_list = machine.layout, section, str(directory / "lattice")
    framework = fw.Framework(machine=machine, directory=str(directory), clean=True, verbose=False)
    framework.loadSettings(settings=settings)
    # A ring line still reads a beam, for its reference momentum
    frameworkGenerator(global_parameters={"master_subdir": framework.subdirectory}, filename="START.openpmd.hdf5",
                       initial_momentum=ENERGY, number_of_particles=16, sigma_x=1e-5, sigma_px=1e3, sigma_y=1e-5,
                       sigma_py=1e3, sigma_z=1e-4, sigma_pz=1e3, charge=1e-15).write()
    return framework


# --- The independent tracker ------------------------------------------------

def _quad(u, k1l, length, offset):
    x, px, y, py = u
    out = []
    for q, p, k in ((x - offset, px, k1l / length), (y, py, -k1l / length)):
        r = np.sqrt(abs(k)) * length
        if k > 0:
            c, s, sp = np.cos(r), np.sin(r) / np.sqrt(k), -np.sqrt(k) * np.sin(r)
        else:
            c, s, sp = np.cosh(r), np.sinh(r) / np.sqrt(-k), np.sqrt(-k) * np.sinh(r)
        out += [c * q + s * p, sp * q + c * p]
    return out[0] + offset, out[1], out[2], out[3]


def _drift(u, length):
    x, px, y, py = u
    return x + length * px, px, y + length * py, py


def one_turn(u, offset_qf0=0.0):
    """(x, px, y, py) after one turn from START; QF0 displaced by offset_qf0 in x."""
    for n in range(CELLS):
        z = 0.0
        for name, cls, centre, length, k in CELL:
            u = _drift(u, centre - length / 2 - z)
            if cls is Quadrupole:
                u = _quad(u, k, length, offset_qf0 if (n, name) == (0, "QF") else 0.0)
            else:
                for _ in range(SLICES):
                    x, px, y, py = _drift(u, length / SLICES / 2)
                    kick = k / SLICES
                    u = _drift((x, px - kick / 2 * (x ** 2 - y ** 2), y, py + kick * x * y), length / SLICES / 2)
            z = centre + length / 2
        u = _drift(u, CELL_LENGTH - z)
    return u


def on_axis_aperture(starts, turns):
    """The last start along `starts` (x, with y = 0) surviving `turns` turns within 1 m."""
    x = np.array(starts, dtype=float)
    u = (x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x))
    alive = np.ones_like(x, dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(turns):
            u = one_turn(u)
            alive &= np.isfinite(u[0]) & (np.abs(u[0]) < 1) & (np.abs(u[2]) < 1)
    lost = np.flatnonzero(~alive)
    return x[lost[0] - 1] if len(lost) and lost[0] > 0 else (x[-1] if not len(lost) else 0.0)


def closed_orbit(offset_qf0):
    """(x, px, y, py) at START that one_turn maps onto itself."""
    return fsolve(lambda v: np.array(one_turn(tuple(v), offset_qf0)) - v, np.zeros(4), xtol=1e-12)
