"""
12 - Hand the beam from one code to the next
============================================

Shows: a .def with several lines, each run by a different code, tracked in
one go. Each line runs from its start_element to its end_element. A line
with no `input:` starts from the beam the previous line left at its
start_element. SIMBA converts the beam between the codes' file formats
on the way.

Here three FODO cells (09's) are split at the screens into three lines:
  FIRST   START -> SCR0B   Ocelot
  SECOND  SCR0B -> SCR1B   elegant if a container runtime is there, else Cheetah
  THIRD   SCR1B -> END     Xsuite
Then the same beam goes through the whole channel in Xsuite alone.

Run:    python 02_tracking/12_start_to_end_handoff.py
Tier:   A (elegant as well, if docker or apptainer and the simcodes image are there)
Needs:  SIMBA, LAURA, Ocelot, Cheetah, Xsuite.
Output: output/12_start_to_end_handoff/chain/ (one input deck per line,
        e.g. FIRST.py, SECOND.lte, THIRD's Xsuite files, and the beam at
        each screen) and output/12_start_to_end_handoff/xsuite/ (the
        single-code run).

Units: x, y in m; charge in C.

Check: the beam must keep every particle and its charge across each hand-off,
and at the two boundaries and at END it must agree particle by particle with
the single-code run, to 1e-3 of the largest offset (the limit 11 uses). A
beam that jumps at a boundary (a lost particle, a sign or unit slip in a
conversion) would fail this.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml
import simba.Framework as fw
import simba.Modules.Beams as rbf
from simple_lattice import fodo, make_beam, write_master_lattice

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"


def container_runtime():
    """docker if the simcodes image has been pulled, else apptainer if installed, else None (see 11)."""
    if shutil.which("docker") and subprocess.run(["docker", "image", "inspect", IMAGE],
                                                 capture_output=True).returncode == 0:
        return "docker"
    return "apptainer" if shutil.which("apptainer") else None


runtime = container_runtime()
middle = "elegant" if runtime else "cheetah"
files = {
    "FIRST": {
        "code": "ocelot",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "SCR0B"},
    },
    # no input: start from the beam FIRST wrote at SCR0B
    "SECOND": {"code": middle, "output": {"start_element": "SCR0B", "end_element": "SCR1B"}},
    "THIRD": {"code": "xsuite", "output": {"start_element": "SCR1B", "end_element": "END"}},
}
single = {
    "WHOLE": {
        "code": "xsuite",
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
cheetah = {"CHEETAHsettings": {"cheetah_tracking_method": "second_order"}}  # see 11
elements = fodo(cells=3)
write_master_lattice(LATTICE, elements, files, "chain.def", **{"global": cheetah})
start = make_beam(LATTICE / "START.openpmd.hdf5", beta=(5.0, 5.0), alpha=(0.0, 0.0))
# a second .def in the same folder, the same but for its files: block
deck = yaml.safe_load((LATTICE / "chain.def").read_text())
(LATTICE / "single.def").write_text(yaml.safe_dump({**deck, "files": single}, sort_keys=False))

results = {}
for name, settings in (("chain", "chain.def"), ("xsuite", "single.def")):
    framework = fw.Framework(master_lattice=str(LATTICE), directory=str(OUT / name), clean=True,
                             verbose=False, container_runtime=runtime)
    framework.loadSettings(settings)
    if name == "chain":
        for line in framework.lines:
            print(f"  {line:7s} {framework[line].code:8s} {framework[line].start} -> {framework[line].end}")
    framework.track()
    results[name] = {s: rbf.load_file(str(OUT / name / f"{s}.openpmd.hdf5")) for s in ("SCR0B", "SCR1B", "END")}

print(f"\n{'screen':7s} {'N':>6s} {'Q [pC]':>8s} {'sigma_x [um]':>13s} {'single code':>12s} {'worst x':>9s} {'worst y':>9s}")
problems = []
for screen, beam in results["chain"].items():
    ref = results["xsuite"][screen]
    row = f"{screen:7s} {len(beam.x):6d} {beam.Q * 1e12:8.3f} {np.std(beam.x.val) * 1e6:13.3f} {np.std(ref.x.val) * 1e6:12.3f}"
    if len(beam.x) != len(start.x) or not np.isclose(beam.Q, start.Q, rtol=1e-9, atol=0):
        problems.append(f"{screen}: N {len(beam.x)}, charge {beam.Q} C (started with {len(start.x)}, {start.Q} C)")
        print(row)
        continue
    for plane in ("x", "y"):
        got, want = getattr(beam, plane).val, getattr(ref, plane).val
        worst = np.max(np.abs(got - want)) / np.max(np.abs(want))
        row += f" {worst:9.1e}"
        if worst > 1e-3:
            problems.append(f"{screen}: {plane} differs from the single-code run by {worst:.1e}")
    print(row)

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print(f"\nOK: the beam crossed ocelot -> {middle} -> xsuite with every particle and its charge, "
      "matching the single-code run")
