"""
13 - Track a real machine: the JFEL linac
=========================================

Shows: tracking JFEL (laura-lattices) from the saved injector beam (35 MeV,
250 pC, at the end of S02) through the linac, the X-band linearizer, the
bunch compressor and the main linac to the undulator entrance (81 m;
s in the table is measured from the start of the line), with
jfel_linac_codes.def. Its one line, Linac, runs in elegant by default;
the script switches it to Ocelot (tier A) unless you ask for another code.
It then summarises the beam at the end of each section, as JFEL's
Scripts/run_example.py does.

Run:    python 02_tracking/13_jfel_linac.py [--code ocelot|cheetah|madx|xsuite|elegant]
Tier:   A (elegant needs a container runtime)
Needs:  SIMBA, LAURA, laura-lattices and the chosen code.
Output: output/13_jfel_linac/run/: the input deck and a beam file at every
        screen and BPM (JFEL-<section>-....openpmd.hdf5).

Units: E in MeV (mean total momentum times c); energy spread in %;
emittance (normalised) in um; sizes in um; sigma_t in fs; current in A.

The .def rematches the injector beam to beta_x = 20 m, beta_y = 10 m
(`input: twiss:`) before tracking, so the first rows differ from the
injector file in size, not in energy or charge.

Collective effects are switched off here (csr_enable, lsc_enable and
wakefield_enable on the line), as for the reference table in the JFEL
README. Not every code models them, and 25 (phase 2) is about them.

Check: against the JFEL README's code-comparison table ("single-particle
optics"), at JFEL-FEL-SIM-MARK-01:
- mean energy within 0.2% of that code's value, the spread the README
  gives between codes;
- normalised emittances and beam sizes within 5%, the README's "a few %"
  between codes;
- no particle and no charge lost.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import JFEL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
RUN = OUT / "run"
JFEL_DIR = os.path.dirname(JFEL.__file__)
END = "JFEL-FEL-SIM-MARK-01"

# JFEL README, "Code comparison", single-particle optics, at END:
# E [MeV], eps_nx, eps_ny, sigma_x, sigma_y [um]
README = {
    "elegant": (576.9, 0.762, 0.798, 49.3, 40.6),
    "ocelot": (576.8, 0.762, 0.812, 48.5, 40.1),
    "cheetah": (576.8, 0.762, 0.797, 48.4, 39.5),
    "madx": (576.7, 0.762, 0.798, 48.5, 39.5),
    "xsuite": (576.8, 0.756, 0.814, 49.1, 40.4),
}

parser = argparse.ArgumentParser()
parser.add_argument("--code", default="ocelot", choices=README)
code = parser.parse_args().code


def summary(beam):
    """E [MeV], spread [%], eps_nx, eps_ny, sigma_x, sigma_y [um], sigma_t [fs], peak current [A]."""
    P = beam.Particles
    cp = beam.cp.val
    t = beam.t.val
    counts, edges = np.histogram(t, bins=max(int(np.sqrt(len(t))), 10))
    q_per_particle = abs(beam.Q) / len(t)
    return (np.mean(cp) / 1e6, 100 * np.std(cp) / np.mean(cp), P.emittance.enx.val * 1e6,
            P.emittance.eny.val * 1e6, P.sigmas.sigma_x.val * 1e6, P.sigmas.sigma_y.val * 1e6,
            np.std(t) * 1e15, counts.max() * q_per_particle / (edges[1] - edges[0]))


framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(RUN), clean=True, verbose=False,
                         container_runtime="docker" if code == "elegant" else None)
framework.loadSettings("Lattices/jfel_linac_codes.def")
framework.change_Lattice_Code("Linac", code)
line = framework["Linac"]
line.csr_enable = line.lsc_enable = line.wakefield_enable = False
# (line.code still names the .def's code after change_Lattice_Code on SIMBA main: ISSUES.md #23)
print(f"Linac: {code}, {line.start} -> {line.end}")
framework.track()

start = rbf.load_file(os.path.join(JFEL_DIR, "Data_Files", "jfel_injector_S02.openpmd.hdf5"))
# The beam at the last screen of each section, in order along the line. The
# accelerating sections (L02 ... L09, L4H) are left out: their screens sit
# at their entrance, so the next section's row shows what they did.
s = line.getSValues(as_dict=True)
last = {}
for name in s:
    if (RUN / f"{name}.openpmd.hdf5").exists() and not name.split("-")[1].startswith("L"):
        last[name.split("-")[1]] = name

row = "{:8.2f} {:7.3f} {:6.3f} {:6.3f} {:6.1f} {:6.1f} {:6.0f} {:6.0f}"
print(f"\n{'section':8s} {'s [m]':>6s} {'E [MeV]':>8s} {'dE [%]':>7s} {'enx':>6s} {'eny':>6s} "
      f"{'sig_x':>6s} {'sig_y':>6s} {'sig_t':>6s} {'I_pk':>6s}")
print(f"{'injector':8s} {0:6.1f} " + row.format(*summary(start)))
for section, name in last.items():
    beam = rbf.load_file(str(RUN / f"{name}.openpmd.hdf5"))
    print(f"{section:8s} {s[name]:6.1f} " + row.format(*summary(beam)))

end = rbf.load_file(str(RUN / f"{END}.openpmd.hdf5"))
E, _, enx, eny, sx, sy, _, _ = summary(end)
ref = README[code]
print(f"\nAt {END}, {code}: E {E:.1f} MeV (README {ref[0]}), eps_n {enx:.3f}/{eny:.3f} um "
      f"(README {ref[1]}/{ref[2]}), sigma {sx:.1f}/{sy:.1f} um (README {ref[3]}/{ref[4]})")

# --- Check -----------------------------------------------------------------
assert len(end.x) == len(start.x) and np.isclose(end.Q, start.Q, rtol=1e-9, atol=0), \
    f"lost particles or charge: {len(start.x)} -> {len(end.x)}, {start.Q} -> {end.Q} C"
assert abs(E / ref[0] - 1) < 0.002, f"E {E:.2f} MeV, README {ref[0]} MeV"
for label, got, want in zip(("eps_nx", "eps_ny", "sigma_x", "sigma_y"), (enx, eny, sx, sy), ref[1:]):
    assert abs(got / want - 1) < 0.05, f"{label} {got:.3f}, README {want}"
print("OK: energy, emittance and beam size at the undulator entrance agree with the JFEL README")
