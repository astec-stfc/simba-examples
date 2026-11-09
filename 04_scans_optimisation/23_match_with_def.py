"""
23 - Match into the JFEL undulator with a `match:` block
========================================================

Shows: SIMBA's matcher, set up in the .def file. JFEL's
Lattices/jfel_match_undulator.def has three lines: Linac (the injector
beam to S13), Match (S13 to the undulator entrance) and Undulator. The
Match line has a block

    match:
      enable: True
      variables: [JFEL-S13-MAG-QUAD-01, ..., -04]
      targets: {}

Before tracking that line, SIMBA runs Ocelot's matcher on a copy of it,
from the line's input beam. It varies the listed quads to hit the targets
and writes their k1l back into the lattice. Then it tracks. The targets
are filled in from code here, as JFEL's match_to_undulator.py does:

    framework["Match"].file_block["match"]["targets"] = {ELEMENT: {"beta_x": ..., ...}}

The target is the undulator's periodic Twiss, also from SIMBA: a line with
`tracking: {periodic: True}` over the undulator, written to a small .def
next to the run, then `framework["Periodic"].ring_parameters()`.

Run:    python 04_scans_optimisation/23_match_with_def.py
Tier:   A
Needs:  SIMBA, LAURA, laura-lattices (JFEL) and Ocelot.
Output: output/23_match_with_def/: run/ (the three lines) and periodic/.

Units: beta in m; k1l in 1/m.

CSR is off on every line, and the structure wakes stay on, as in the
README's match. The S13 strengths stored in the lattice were matched
without CSR, and since SIMBA PR #67 the .def as it stands includes it
(ISSUES.md #28). The wakes matter through the energy: without them the
beam reaches the undulator ~0.7% higher in energy, its natural focusing is
weaker, and the periodic beta_x is 0.7% larger. The quads start at half
their stored strengths, so the matcher has real work to do.

Checks:
- The tracked beam at the undulator entrance and exit equals the periodic
  target to 2% in beta and 0.02 in alpha (measured 0.4%, 0.009). They
  aren't equal: the matcher works from the beam's Twiss at S13 and linear
  optics, while the tracked beam has energy spread and the S13 quads and
  undulator are chromatic.
- The periodic Twiss and the beam at the entrance and exit equal the JFEL
  README's "Matching into the undulator" table to 1e-3 in beta and 0.002
  in alpha. The README's values are this calculation rounded to 3
  decimals (measured within 1.5e-4 and 0.0005).
- The matched k1l equal the strengths stored in the lattice, which came
  from the same match, to 1e-3 (measured 7e-6).
"""

import os
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
import yaml
from laura_lattices import JFEL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
RUN = OUT / "run"
JFEL_DIR = os.path.dirname(JFEL.__file__)
QUADS = [f"JFEL-S13-MAG-QUAD-0{i}" for i in range(1, 5)]
S13, ENTRANCE, EXIT = "JFEL-S13-SIM-MARK-01", "JFEL-FEL-SIM-MARK-01", "JFEL-FEL-SIM-MARK-02"
KEYS = ("beta_x", "alpha_x", "beta_y", "alpha_y")
README = {  # JFEL README, "Matching into the undulator"
    "periodic": dict(zip(KEYS, (3.448, 0.185, 2.208, -0.274))),
    "entrance": dict(zip(KEYS, (3.460, 0.184, 2.211, -0.280))),
    "exit": dict(zip(KEYS, (3.461, 0.181, 2.204, -0.283))),
}


def twiss(name):
    t = rbf.load_file(str(RUN / f"{name}.openpmd.hdf5")).twiss
    return {k: float(np.ravel(getattr(t, k).val)[0]) for k in KEYS}


def differences(got, want):
    """Largest relative beta difference and absolute alpha difference."""
    beta = max(abs(got[k] / want[k] - 1) for k in ("beta_x", "beta_y"))
    alpha = max(abs(got[k] - want[k]) for k in ("alpha_x", "alpha_y"))
    return beta, alpha


framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(RUN), clean=True, verbose=False)
framework.loadSettings("Lattices/jfel_match_undulator.def")
for line in ("Linac", "Match", "Undulator"):
    framework[line].csr_enable = False
stored = [framework.getElement(q).k1l for q in QUADS]
framework.track(files=["Linac"], check_lattice=False)

# The undulator's periodic Twiss. The input beam (for its energy) is the
# Linac's output, given as an absolute path without its extension.
periodic_def = OUT / "periodic.def"
periodic_def.write_text(yaml.safe_dump({
    "files": {"Periodic": {
        "code": "ocelot",
        "input": {"particle_definition": str(RUN / S13)},
        "tracking": {"periodic": True},
        "output": {"start_element": ENTRANCE, "end_element": EXIT},
    }},
    "layout": "./layouts.yaml", "section": "./sections.yaml", "element_list": "./YAML/",
}, sort_keys=False))
periodic = fw.Framework(master_lattice=JFEL_DIR, directory=str(OUT / "periodic"), clean=True, verbose=False)
periodic.loadSettings(str(periodic_def))
periodic.track()
ring = periodic["Periodic"].ring_parameters()
target = {k: float(ring[k]) for k in KEYS}

# Detune, set the targets, then track: the match runs first
for q, k1l in zip(QUADS, stored):
    framework.modifyElement(q, "k1l", 0.5 * k1l)
framework["Match"].file_block["match"]["targets"] = {ENTRANCE: target}
framework.track(files=["Match", "Undulator"], check_lattice=False)
matched = [framework.getElement(q).k1l for q in QUADS]

rows = {"periodic (SIMBA)": target, "beam at entrance": twiss(ENTRANCE), "beam at exit": twiss(EXIT)}
rows.update({f"README {k}": v for k, v in README.items()})
print(f"\n{'':18s}" + "".join(f"{k:>9s}" for k in KEYS))
for label, row in rows.items():
    print(f"{label:18s}" + "".join(f"{row[k]:9.4f}" for k in KEYS))
print(f"\n{'S13 k1l [1/m]':18s}" + "".join(f"{q[-7:]:>9s}" for q in QUADS))
for label, values in (("stored", stored), ("start (x0.5)", [0.5 * k for k in stored]), ("matched", matched)):
    print(f"{label:18s}" + "".join(f"{v:9.4f}" for v in values))

# --- Checks ----------------------------------------------------------------
problems = []
for label, got, want, beta_tol, alpha_tol in (
    ("entrance vs target", rows["beam at entrance"], target, 0.02, 0.02),
    ("exit vs target", rows["beam at exit"], target, 0.02, 0.02),
    ("periodic vs README", target, README["periodic"], 1e-3, 0.002),
    ("entrance vs README", rows["beam at entrance"], README["entrance"], 1e-3, 0.002),
    ("exit vs README", rows["beam at exit"], README["exit"], 1e-3, 0.002),
):
    beta, alpha = differences(got, want)
    print(f"{label:20s} beta {beta:.1e} (limit {beta_tol}), alpha {alpha:.4f} (limit {alpha_tol})")
    if beta > beta_tol or alpha > alpha_tol:
        problems.append(label)
k_diff = max(abs(m / s - 1) for m, s in zip(matched, stored))
print(f"{'matched vs stored':20s} k1l {k_diff:.1e} (limit 1e-3)")
if k_diff > 1e-3:
    problems.append("matched k1l vs stored")
assert not problems, f"failed: {problems}"
print("OK: from half strength, the match block brings the beam onto the undulator's periodic Twiss")
