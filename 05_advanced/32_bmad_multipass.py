"""
32 - Bmad: the ERL flattened, and as Bmad multipass
===================================================

Shows: Bmad through SIMBA (pytao and libtao), and LAURA's Bmad multipass
export. Every other code tracks a multipass layout flattened, with a copy
of each cavity per pass (28). Bmad can hold the shared hardware itself:
`multipass: native` on a line (ERL/Lattices/erl_bmad_multipass.def) writes
each cavity once, as a multipass lord with one slave per pass, and the
second pass's phase per slave:

    LIN: line[multipass] = (..., ERL_LIN_RF_CAV_01, ...)
    ERL_LIN_RF_CAV_01\\2[phi0_multipass] = -0.5

Run:    python 05_advanced/32_bmad_multipass.py
Tier:   C
Needs:  SIMBA, LAURA, laura-lattices (ERL); Bmad: pytao and
        $ACC_ROOT_DIR/production/lib/libtao.so (export ACC_ROOT_DIR, and put
        $ACC_ROOT_DIR/production/lib first on LD_LIBRARY_PATH; README,
        troubleshooting). Prints SKIP and exits 0 without them.
Output: output/32_bmad_multipass/<flattened|native>/ (ERL.bmad is the
        lattice Bmad read).

Units: momentum in MeV/c; spread relative, rms; emittance (normalised) in
um; sizes in mm; sigma_t in ps.

Both runs are single-particle: SIMBA lines start with CSR and wakes on, so
they are switched off here. Each takes ~10 s.

Checks:
- Native multipass equals the flattened lattice at the dump screen, in
  momentum, spread, emittances, sizes and sigma_t, to 1e-4 (measured
  9e-6; the README says "identical"; the two lattices are sliced
  differently).
- Flattened, the ERL README's Bmad column: 10.000, 100.000, 10.000 MeV/c
  (to 1e-4; measured 2e-6), top and dump spreads 1.98e-4, 1.00e-3 and the
  dump's sigma_x / sigma_y 0.579 / 0.580 mm, sigma_t 2.01 ps (to 1%, the
  README's rounding plus version drift).
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import ERL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
ERL_DIR = os.path.dirname(ERL.__file__)
LIBTAO = os.path.join(os.environ.get("ACC_ROOT_DIR", ""), "production", "lib", "libtao.so")
SCREEN = "ERL-DMP-DIA-SCR-01"
QUANTITIES = ("p", "spread", "eps_nx", "eps_ny", "sigma_x", "sigma_y", "sigma_t")
README = {"injected": {"p": 10.000}, "top": {"p": 100.000, "spread": 1.98e-4},  # ERL README, Bmad column
          "dump": {"p": 10.000, "spread": 1.00e-3, "sigma_x": 0.579, "sigma_y": 0.580, "sigma_t": 2.01}}
POINTS = {"injected": "ERL-LIN-SIM-START-01.1", "top": "ERL-LIN-SIM-END-01.1", "dump": "ERL-LIN-SIM-END-01.2"}


def summary(path):
    b = rbf.load_file(str(path))
    cp = b.cp.val
    return dict(zip(QUANTITIES, (np.mean(cp) / 1e6, np.std(cp) / np.mean(cp), b.Particles.emittance.enx.val * 1e6,
                                 b.Particles.emittance.eny.val * 1e6, np.std(b.x.val) * 1e3, np.std(b.y.val) * 1e3,
                                 np.std(b.t.val) * 1e12)))


def track(native):
    directory = OUT / ("native" if native else "flattened")
    f = fw.Framework(master_lattice=ERL_DIR + "/", directory=str(directory), clean=True, verbose=False)
    f.executables.define_tao_command(location=[LIBTAO])
    f.loadSettings("Lattices/erl_bmad_multipass.def" if native else "Lattices/erl.def")
    for name in [n for n in f.latticeObjects if n != "generator"]:
        if not native:
            f.change_Lattice_Code(name, "bmad")
        f[name].csr_enable = f[name].wakefield_enable = False
    f.track(check_lattice=False)
    return directory


if not os.path.isfile(LIBTAO) or importlib.util.find_spec("pytao") is None:
    print("SKIP: needs pytao and $ACC_ROOT_DIR/production/lib/libtao.so (export ACC_ROOT_DIR)")
    sys.exit(0)

flattened, native = track(False), track(True)
print("Bmad multipass in the native lattice (ERL.bmad):")
for line in (native / "ERL.bmad").read_text().splitlines():
    if "line[multipass]" in line or "RF_CAV_01\\" in line:
        print("   ", line[:100] + (" ..." if len(line) > 100 else ""))

at = {"flattened": summary(flattened / f"{SCREEN}.openpmd.hdf5"), "native": summary(native / f"{SCREEN}.openpmd.hdf5")}
points = {k: summary(flattened / f"{p}.openpmd.hdf5") for k, p in POINTS.items()}
print(f"\n{'':22s}" + "".join(f"{q:>10s}" for q in QUANTITIES))
for name, s in [(f"screen, {k}", v) for k, v in at.items()] + [(f"flattened, {k}", v) for k, v in points.items()]:
    print(f"{name:22s}" + "".join(f"{s[q]:10.4g}" for q in QUANTITIES))

# --- Checks ----------------------------------------------------------------
problems = []
worst = max(abs(at["native"][q] / at["flattened"][q] - 1) for q in QUANTITIES)
print(f"\nnative against flattened at the screen: largest relative difference {worst:.1e}")
if worst > 1e-4:
    problems.append(f"native multipass differs from the flattened lattice by {worst:.1e}")
for point, refs in README.items():
    for q, ref in refs.items():
        limit = 1e-4 if q == "p" else 0.01
        if abs(points[point][q] / ref - 1) > limit:
            problems.append(f"{point} {q}: {points[point][q]:.5g}, README {ref}")
assert not problems, "\n".join(problems)
print("OK: Bmad multipass equals the flattened lattice, and the README's Bmad column is reproduced")
