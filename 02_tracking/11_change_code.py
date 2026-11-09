"""
11 - Run the same line in different codes
=========================================

Shows: `framework.change_Lattice_Code(line, code)`, which switches a line to
another tracking code without touching the lattice or the .def. The script
tracks the same beam through the same three FODO cells (09's) in Ocelot,
Cheetah, Xsuite and MAD-X. If a container runtime with the simcodes image is
there, it also runs elegant. Each run goes in its own directory
(setSubDirectory), and the script compares the codes particle by particle at
the end.

Run:    python 02_tracking/11_change_code.py
Tier:   A (elegant as well, if docker or apptainer and the simcodes image are there)
Needs:  SIMBA, LAURA, Ocelot, Cheetah, Xsuite and cpymad (MAD-X); the
        [simcodes] extra installs them all.
Output: output/11_change_code/<code>/: each code's input deck (LINE.py for
        Ocelot, LINE.madx, LINE.lte for elegant, ...) and END.openpmd.hdf5.

Units: x, y in m.

What differs between the codes:
- MAD-X tracks a thin-lens (sliced) copy of the line. SIMBA uses 4 slices per
  quadrupole by default and warns that the tunes can then differ from the
  thick optics by up to 0.01. Here that costs about 1e-4 in the particle
  positions. Set e.g. `tracking: {nslice_quadrupole: 16}` in the line's
  `files:` entry for more slices.
- Cheetah's default (`cheetah_tracking_method: auto`) tracks first-order
  maps unless that changes a beam size or emittance by more than 1%. First
  order leaves out chromatic focusing, which here moves particles by 1e-3 of
  the beam size (an energy spread of 1e-4 over 2/3 of a betatron period),
  so the script asks for `second_order` in the .def's `global:` block.

Check: every code must put every particle within 1e-3 of the beam's largest
offset from where Xsuite puts it, in x and y. That is the limit SIMBA's own
cross-code test uses (unit_tests/test_solenoid_and_alignment.py). The codes
must also keep every particle and the charge.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from simple_lattice import fodo, make_beam, write_master_lattice

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"


def container_runtime():
    """
    docker if the simcodes image has been pulled, else apptainer if installed, else None.
    Docker comes first: with apptainer, SIMBA downloads its own image (several GB) on first use.
    """
    if shutil.which("docker") and subprocess.run(["docker", "image", "inspect", IMAGE],
                                                 capture_output=True).returncode == 0:
        return "docker"
    return "apptainer" if shutil.which("apptainer") else None


runtime = container_runtime()
codes = ["xsuite", "ocelot", "cheetah", "madx"] + (["elegant"] if runtime else [])
files = {
    "LINE": {
        "code": codes[0],
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
# Cheetah's first-order maps leave out chromatic focusing (see above), so ask
# for second order. `global:` in a .def holds per-code settings.
cheetah = {"CHEETAHsettings": {"cheetah_tracking_method": "second_order"}}
settings = write_master_lattice(LATTICE, fodo(cells=3), files, "fodo.def", **{"global": cheetah})
start = make_beam(LATTICE / "START.openpmd.hdf5", beta=(5.0, 5.0), alpha=(0.0, 0.0))

framework = fw.Framework(master_lattice=str(LATTICE), directory=str(OUT / codes[0]), clean=True,
                         verbose=False, container_runtime=runtime)
framework.loadSettings(settings)
ends = {}
stale = []
for code in codes:
    framework.change_Lattice_Code("LINE", code)
    framework.setSubDirectory(str(OUT / code))
    if framework["LINE"].code != code:
        stale.append(f"{code} (line.code {framework['LINE'].code!r})")
    framework.track()
    ends[code] = rbf.load_file(str(OUT / code / "END.openpmd.hdf5"))
# ISSUES.md #23: the tracking does switch code, but line.code (and the saved
# settings.def) keep the .def's code.
if stale:
    print(f"XFAIL #23: line.code not updated by change_Lattice_Code for {', '.join(stale)}")
else:
    print("XPASS #23: line.code follows change_Lattice_Code; remove this check and the note in ISSUES.md")
if not runtime:
    print("(elegant skipped: no container runtime with the simcodes image)")

ref = ends["xsuite"]
print(f"\nAt END, against xsuite (worst |difference| / largest |offset|):")
print(f"{'code':8s} {'N':>6s} {'sigma_x [um]':>13s} {'sigma_y [um]':>13s} {'x':>9s} {'y':>9s}")
problems = []
for code, beam in ends.items():
    row = f"{code:8s} {len(beam.x):6d} {np.std(beam.x.val) * 1e6:13.3f} {np.std(beam.y.val) * 1e6:13.3f}"
    if len(beam.x) != len(start.x) or not np.isclose(beam.Q, start.Q, rtol=1e-9, atol=0):
        problems.append(f"{code}: N {len(beam.x)}, charge {beam.Q} C")
        print(row)
        continue
    for plane in ("x", "y"):
        got, want = getattr(beam, plane).val, getattr(ref, plane).val
        worst = np.max(np.abs(got - want)) / np.max(np.abs(want))
        row += f" {worst:9.1e}"
        if worst > 1e-3:
            problems.append(f"{code}: {plane} differs from xsuite by {worst:.1e} of the largest offset")
    print(row)

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print(f"\nOK: {', '.join(codes)} agree particle by particle to 1e-3")
