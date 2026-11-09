"""
06 - Read, write and convert beam files
=======================================

Shows: loading a beam file into a SIMBA `beam`, writing it in the formats of
the different codes (openPMD, ASTRA, elegant SDDS, Ocelot npz, GPT GDF) and
reading each back. SIMBA picks the reader from the file name, so a beam
written by any code can be loaded the same way: `rbf.load_file(path)`.

Run:    python 01_basics/06_beam_io_and_conversion.py
Tier:   A
Needs:  SIMBA and laura-lattices (the JFEL injector beam); Ocelot for the
        .ocelot.npz format.
Output: output/06_beam_io_and_conversion/beam.* in each format.

The beam holds x, y, z (m), t (s) and px, py, pz (kg m/s); cpx, cpy, cpz
are the momenta in eV/c. Name files so SIMBA can tell the format:
  *.openpmd.hdf5  openPMD (SIMBA's own format between codes)
  *.astra         ASTRA        *.sdds        elegant
  *.ocelot.npz    Ocelot       *.gdf         GPT
  *.hdf5          SIMBA's older HDF5 format (see the XFAIL below)

What a format keeps:
- elegant SDDS and Ocelot are time-based (t, or c*t relative to the bunch):
  z comes back rebuilt from t, centred on zero, so the bunch's absolute
  position (3.297 m for this beam) is not stored.
- A GPT initial distribution (GDF) holds the particles at one instant, so it
  has no t: on reading, t comes from z and the momentum.
- openPMD and ASTRA keep everything.

Check: every format gives back the same number of particles, the same total
charge, and the same means and rms values of x, y, z, t, px, py, pz to 1e-6
relative, apart from what the format does not store (above).

Known bugs, run as XFAIL (see ISSUES.md): an SDDS file reads back with
the wrong sign of charge, and write_HDF5_beam_file fails for a beam that was
not read from a code's output. If one of them passes, the script says so.
"""

import os
from pathlib import Path

import numpy as np
import simba.Modules.Beams as rbf
from laura_lattices import JFEL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = os.path.join(JFEL.data_files, "jfel_injector_S02.openpmd.hdf5")

beam = rbf.load_file(SOURCE)
print(f"Read {SOURCE}\n  {len(beam.x)} particles, {beam.Q * 1e12:.1f} pC, "
      f"<cp> {np.mean(beam.cp) / 1e6:.2f} MeV/c, species {beam.species}")

writers = {
    "openPMD": ("beam.openpmd.hdf5", beam.write_openpmd_beam_file),
    "ASTRA": ("beam.astra", beam.write_astra_beam_file),
    "SDDS": ("beam.sdds", beam.write_SDDS_beam_file),
    "Ocelot": ("beam.ocelot.npz", beam.write_ocelot_beam_file),
    "GDF": ("beam.gdf", beam.write_gdf_beam_file),
}
KEYS = ["x", "y", "z", "t", "cpx", "cpy", "cpz"]


def moments(b):
    return {k: (np.mean(getattr(b, k).val), np.std(getattr(b, k).val)) for k in KEYS}


ref = moments(beam)
problems, xfail = [], []
print(f"\n{'format':8s} {'N':>6s} {'Q [pC]':>8s}  worst relative difference in a mean or rms")
for fmt, (name, write) in writers.items():
    path = OUT / name
    write(str(path))
    back = rbf.load_file(str(path))
    got = moments(back)
    no_mean = {"GDF": ["t"], "SDDS": ["z"], "Ocelot": ["z"]}.get(fmt, [])  # not stored, see above
    no_rms = {"GDF": ["t"]}.get(fmt, [])
    # Means are compared on the scale of the rms (a mean can be ~0).
    worst = max([abs(got[k][0] - ref[k][0]) / ref[k][1] for k in KEYS if k not in no_mean]
                + [abs(got[k][1] / ref[k][1] - 1) for k in KEYS if k not in no_rms])
    print(f"{fmt:8s} {len(back.x):6d} {back.Q * 1e12:8.1f}  {worst:.1e}")
    if len(back.x) != len(beam.x) or worst > 1e-6:
        problems.append(f"{fmt}: N {len(back.x)}, worst difference {worst:.1e}")
    if not np.isclose(back.Q, beam.Q, rtol=1e-9, atol=0):  # atol=0: the default 1e-8 C would pass anything
        if fmt == "SDDS" and np.isclose(back.Q, -beam.Q, rtol=1e-9, atol=0):
            xfail.append("SDDS charge sign")  # ISSUES.md: SDDS charge sign
        else:
            problems.append(f"{fmt}: charge {back.Q} C, expected {beam.Q} C")
    elif fmt == "SDDS":
        print("XPASS: the SDDS charge sign is right now; remove its XFAIL")

try:
    beam.write_HDF5_beam_file(str(OUT / "beam.hdf5"))
    print("XPASS: write_HDF5_beam_file works now; add it to the table and remove its XFAIL")
except TypeError:
    xfail.append("write_HDF5_beam_file")  # ISSUES.md: HDF5 writer and beam.code

for x in xfail:
    print(f"XFAIL: {x} (known bug, see ISSUES.md)")
# --- Check -----------------------------------------------------------------
assert not problems, "Round trips changed the beam:\n  " + "\n  ".join(problems)
print("\nOK: every format kept N, charge (apart from the XFAIL) and the means and rms values")
