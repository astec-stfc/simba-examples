"""
33 - The same match in every code
=================================

Shows: where a `match:` block runs. SIMBA's matching is its own module
(simba.Modules.Matching); a `.def` names what to vary and what to hit, and
the match runs in a code with a matcher. By default that is the line's own
code; `code:` in the block picks another:

    match:
      variables: [JFEL-S13-MAG-QUAD-01, ..., -04]
      targets: {JFEL-FEL-SIM-MARK-01: {beta_x: ..., alpha_x: ..., ...}}
      code: xsuite

Ocelot, Xsuite, MAD-X and Bmad use their own matchers (Tao's for Bmad),
elegant its optimiser; Cheetah, which has none, SIMBA's least-squares
solver on Cheetah's optics. Targets are SIMBA's Twiss names whatever the code.

The script runs example 23's match (JFEL's S13 quads, from half strength,
onto the undulator's periodic Twiss), once per code, on the same Ocelot line
and beam. It then shows the default: a Cheetah line with a dispersion target,
which Cheetah's optics cannot give, is matched in the first code that can
(simba.Modules.Matching.transverse.FALLBACK), with a warning that says why.

Run:    python 06_matching/33_match_in_any_code.py
Tier:   A (elegant runs if it is installed or in the container; Bmad if
        pytao and $ACC_ROOT_DIR's libtao.so are; each prints SKIP otherwise)
Needs:  SIMBA, LAURA, laura-lattices (JFEL), Ocelot, Xsuite, MAD-X, Cheetah.
Output: output/33_match_in_any_code/: run/ and periodic/.

Units: k1l in 1/m; beta in m.

Four quads and four targets can have more than one exact solution. From the
same start, Ocelot, MAD-X, Cheetah, elegant and Bmad land on the strengths
stored in the lattice (which came from Ocelot's match) to ~6e-6; Xsuite's
solver lands on another (k1l about 1.28, -1.21, -0.46, 1.44 1/m), which
meets the targets as well. Which solution is wanted (the weaker quads, a
smaller beta inside) is the user's to say, with more targets or bounds.

Checks:
- After each match, the tracked beam at the undulator entrance is on the
  target to 2% in beta and 0.02 in alpha, as in example 23.
- Ocelot's match gives the stored strengths to 1e-3 (measured 6e-6).
- The Cheetah line hands over with a warning, and its match meets the target too.
"""

import warnings
from pathlib import Path

import simba.Framework as fw
from jfel import ENTRANCE, JFEL_DIR, S13, S13_QUADS, beam_twiss, differences, periodic_target

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
RUN = OUT / "run"
CODES = ("ocelot", "xsuite", "madx", "cheetah", "elegant", "bmad")

framework = fw.Framework(master_lattice=JFEL_DIR, directory=str(RUN), clean=True, verbose=False)
framework.loadSettings("Lattices/jfel_match_undulator.def")
for line in ("Linac", "Match", "Undulator"):
    framework[line].csr_enable = False
stored = [framework.getElement(q).k1l for q in S13_QUADS]
framework.track(files=["Linac"], check_lattice=False)
target = periodic_target(OUT, RUN / S13)


def match_in(code=None, extra_targets=None):
    """Match from half strength, in `code` (None: the default), and track; the matched k1l and entrance Twiss."""
    for q, k1l in zip(S13_QUADS, stored):
        framework.modifyElement(q, "k1l", 0.5 * k1l)
    block = framework["Match"].file_block["match"]
    block["targets"] = {ENTRANCE: {**target, **(extra_targets or {})}}
    block.pop("code", None)
    if code:
        block["code"] = code
    framework.track(files=["Match"], check_lattice=False)
    return [framework.getElement(q).k1l for q in S13_QUADS], beam_twiss(RUN / f"{ENTRANCE}.openpmd.hdf5")


results, problems = {}, []
for code in CODES:
    try:
        results[code] = match_in(code)
    except ValueError as error:  # the code cannot run here; says why
        print(f"SKIP {code}: {error}")

# The default: a Cheetah line with a target Cheetah cannot give hands over
framework.change_Lattice_Code("Match", "cheetah")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    handed = match_in(extra_targets={"eta_x": ["a<", 0.01]})
handover = [str(w.message) for w in caught if "cannot match this" in str(w.message)]
print("\n" + (handover[0] if handover else "no hand-over warning"))
results["cheetah line (default)"] = handed

print(f"\n{'k1l [1/m]':24s}" + "".join(f"{q[-7:]:>9s}" for q in S13_QUADS) + "   max |k1l/stored - 1|")
print(f"{'stored':24s}" + "".join(f"{v:9.4f}" for v in stored))
for label, (k1l, _) in results.items():
    worst = max(abs(m / s - 1) for m, s in zip(k1l, stored))
    print(f"{label:24s}" + "".join(f"{v:9.4f}" for v in k1l) + f"   {worst:.1e}")
if max(abs(m / s - 1) for m, s in zip(results["ocelot"][0], stored)) > 1e-3:
    problems.append("Ocelot's match is not the stored strengths")

print(f"\n{'beam at entrance':24s} beta vs target  alpha vs target")
for label, (_, twiss) in results.items():
    beta, alpha = differences(twiss, target)
    print(f"{label:24s} {beta:13.1e}  {alpha:15.4f}")
    if beta > 0.02 or alpha > 0.02:
        problems.append(f"{label} Twiss")

# --- Checks ----------------------------------------------------------------
if not handover:
    problems.append("no hand-over warning")
assert not problems, f"failed: {problems}"
print(f"\nOK: {len(results) - 1} codes match the beam onto the undulator, and the Cheetah line hands over")
