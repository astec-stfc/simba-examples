"""
28 - An energy-recovery linac: one cryomodule, two passes
=========================================================

Shows: a LAURA multipass layout in SIMBA. laura-lattices' ERL crosses the
same cavities twice: 10 -> 100 MeV/c on crest, round a racetrack, then back
through them half an RF period late, 100 -> 10 MeV/c, handing the energy
back to the RF. The layout names the linac's elements per pass, `NAME#1`
and `NAME#2`; in a tracking line (and its beam files) they are `NAME.1` and
`NAME.2`. The second pass's cavity phase (180 deg) belongs to the pass, so
a path-length error is a change to the `.2` cavities only:

    for name, el in f["ERL"].elements.items():
        if name.startswith("ERL-LIN-RF-CAV") and name.endswith(".2"):
            el.cavity.phase = 180 + offset

Run:    python 05_advanced/28_erl_multipass.py
Tier:   A (Ocelot); B for elegant (docker).
Needs:  SIMBA and LAURA with multipass layouts, laura-lattices (ERL) and
        Ocelot; docker and ghcr.io/astec-stfc/simcodes-docker:master for
        elegant (skipped otherwise).
Output: output/28_erl_multipass/<code>_<offset>/.

Units: momentum in MeV/c; spread relative, rms; emittance (normalised) in
um; sizes in mm; sigma_t in ps; phase offset in degrees.

The lattice is ERL/Lattices/erl.def: line ERL (injector to the linac exit
on pass 2) and line DUMP. No CSR or synchrotron radiation. Each code runs
at offset 0; Ocelot also at +-5, +-10, +-20 deg, elegant at +10 (18 s a run).

Checks:
- Offset 0: the momentum is 10, 100, 10 MeV/c at injection, the top and the
  dump, to 1e-4 (measured 1.2e-5, elegant at the top); the RF takes back
  all the energy it gave to 1e-4 (measured 5e-6).
- Offset 0, at the linac exit on pass 2: the beam is the injected beam
  again (the racetrack is mirror-symmetric). eps_nx, eps_ny, sigma_x,
  sigma_y and sigma_t equal the injected ones to 1% (measured 0.1% or
  less).
- At the dump screen, after the dump dipole, eps_nx reads 5 um: that is
  dispersion (D = -0.58 m) times the spread, not growth. Taking out the
  x-delta correlation gives the injected eps_nx to 1% (measured 0.998).
- Offset 0: the ERL README's single-particle table (top and dump momentum
  and spread, dump sizes and sigma_t), to 1%; same input and code, so this
  is the README's rounding plus version drift.
- Phase offset: the dump momentum is 10 + 90 (1 - cos offset) MeV/c to
  0.05 MeV/c. The formula ignores the bunch's phase spread and its
  velocity (the README's elegant scan spans 15.40-15.46 at +-20 deg).
  Measured: Ocelot within 0.007, elegant 0.019 low at +10. The spread is
  90 sin(offset) x the RF phase spread (2 pi 1.3 GHz sigma_t = 0.016 rad),
  with the injected 0.01 MeV/c in quadrature, relative to the dump
  momentum, to 3% (measured 1.2% or less).
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura_lattices import ERL

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
ERL_DIR = os.path.dirname(ERL.__file__)
IMAGE = "ghcr.io/astec-stfc/simcodes-docker:master"
MASS = 0.51099895  # MeV/c^2
POINTS = {"injected": "ERL-LIN-SIM-START-01.1", "top": "ERL-LIN-SIM-END-01.1",
          "dump": "ERL-LIN-SIM-END-01.2", "screen": "ERL-DMP-DIA-SCR-01"}
OFFSETS = {"ocelot": [0, -20, -10, -5, 5, 10, 20], "elegant": [0, 10]}
GAIN, RF = 90.0, 1.3e9  # MeV/c per pass; Hz
README = {  # ERL README, single-particle: top p, spread; dump p, spread, sigma_x, sigma_y [mm], sigma_t [ps]
    "elegant": (99.999, 2.00e-4, 10.000, 1.01e-3, 0.581, 0.581, 2.01),
    "ocelot": (100.000, 1.98e-4, 10.000, 1.00e-3, 0.580, 0.581, 2.01),
}


def docker_ready():
    return bool(shutil.which("docker")) and subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


def summary(beam):
    cp, d = beam.cp.val / 1e6, beam.cp.val / np.mean(beam.cp.val) - 1
    x, xp = beam.x.val, beam.xp.val
    dx, dxp = np.polyfit(d, x, 1)[0], np.polyfit(d, xp, 1)[0]
    eps_dispersion_free = np.sqrt(np.linalg.det(np.cov(x - dx * d, xp - dxp * d))) * np.mean(cp) / MASS
    return {"p": np.mean(cp), "spread": np.std(d), "eps_nx": beam.Particles.emittance.enx.val * 1e6,
            "eps_ny": beam.Particles.emittance.eny.val * 1e6, "sigma_x": np.std(x) * 1e3,
            "sigma_y": np.std(beam.y.val) * 1e3, "sigma_t": np.std(beam.t.val) * 1e12,
            "eps_nx_no_D": eps_dispersion_free * 1e6, "D": dx}


def track(code, offset):
    """The ERL in `code`, return-pass phase 180 + offset; a summary at each of POINTS."""
    directory = OUT / f"{code}_{offset:+d}"
    f = fw.Framework(master_lattice=ERL_DIR + "/", directory=str(directory), clean=True, verbose=False,
                     container_runtime="docker" if code == "elegant" else None)
    f.loadSettings("Lattices/erl.def")
    for line in ("ERL", "DUMP"):
        f.change_Lattice_Code(line, code)
        f[line].csr_enable = False
    for name, el in f["ERL"].elements.items():
        if name.startswith("ERL-LIN-RF-CAV") and name.endswith(".2"):
            el.cavity.phase = 180.0 + offset
    f.track(check_lattice=False)
    return {k: summary(rbf.load_file(str(directory / f"{p}.openpmd.hdf5"))) for k, p in POINTS.items()}


def energy(p):
    return np.hypot(p, MASS)


codes = ["ocelot"] + (["elegant"] if docker_ready() else [])
if "elegant" not in codes:
    print(f"elegant skipped: docker or {IMAGE} not available\n")
runs = {(code, offset): track(code, offset) for code in codes for offset in OFFSETS[code]}

print(f"{'code':8s} {'point':9s} {'p':>9s} {'spread':>9s} {'eps_nx':>7s} {'eps_ny':>7s} {'sig_x':>7s} {'sig_y':>7s} {'sig_t':>6s}   eps_nx without D")
for code in codes:
    for point, s in runs[(code, 0)].items():
        print(f"{code:8s} {point:9s} {s['p']:9.4f} {s['spread']:9.3e} {s['eps_nx']:7.3f} {s['eps_ny']:7.3f} "
              f"{s['sigma_x']:7.4f} {s['sigma_y']:7.4f} {s['sigma_t']:6.3f}   {s['eps_nx_no_D']:.3f} (D {s['D']:+.3f} m)")
    r = runs[(code, 0)]
    recovered = (energy(r["top"]["p"]) - energy(r["dump"]["p"])) / (energy(r["top"]["p"]) - energy(r["injected"]["p"]))
    print(f"{code:8s} energy taken back by the RF: {recovered:.5%}\n")


def law(offset, sigma_t):
    p = 10.0 + GAIN * (1 - np.cos(np.radians(offset)))
    spread = np.hypot(GAIN * np.sin(np.radians(offset)) * 2 * np.pi * RF * sigma_t * 1e-12, 10.0 * 1e-3) / p
    return p, spread


print(f"{'code':8s} {'offset':>6s} {'dump p':>8s} {'formula':>8s} {'spread':>9s} {'formula':>9s}")
for (code, offset), r in runs.items():
    p, spread = law(offset, r["injected"]["sigma_t"])
    print(f"{code:8s} {offset:+6d} {r['dump']['p']:8.4f} {p:8.4f} {r['dump']['spread']:9.3e} {spread:9.3e}")

# --- Checks ----------------------------------------------------------------
problems = []
for code in codes:
    r = runs[(code, 0)]
    for point, target in (("injected", 10.0), ("top", 100.0), ("dump", 10.0)):
        if abs(r[point]["p"] / target - 1) > 1e-4:
            problems.append(f"{code}: {point} momentum {r[point]['p']:.5f}, expected {target}")
    recovered = (energy(r["top"]["p"]) - energy(r["dump"]["p"])) / (energy(r["top"]["p"]) - energy(r["injected"]["p"]))
    if abs(recovered - 1) > 1e-4:
        problems.append(f"{code}: energy recovered {recovered:.5%}")
    for q in ("eps_nx", "eps_ny", "sigma_x", "sigma_y", "sigma_t"):
        if abs(r["dump"][q] / r["injected"][q] - 1) > 0.01:
            problems.append(f"{code}: dump {q} {r['dump'][q]:.4f}, injected {r['injected'][q]:.4f}")
    if abs(r["screen"]["eps_nx_no_D"] / r["injected"]["eps_nx"] - 1) > 0.01:
        problems.append(f"{code}: eps_nx at the screen without dispersion {r['screen']['eps_nx_no_D']:.3f}")
    got = (r["top"]["p"], r["top"]["spread"], r["dump"]["p"], r["dump"]["spread"],
           r["dump"]["sigma_x"], r["dump"]["sigma_y"], r["dump"]["sigma_t"])
    if max(abs(g / ref - 1) for g, ref in zip(got, README[code])) > 0.01:
        problems.append(f"{code}: {np.round(got, 5)} against the README's {README[code]}")
for (code, offset), r in runs.items():
    p, spread = law(offset, r["injected"]["sigma_t"])
    if abs(r["dump"]["p"] - p) > 0.05:
        problems.append(f"{code} {offset:+d} deg: dump momentum {r['dump']['p']:.4f}, formula {p:.4f}")
    if abs(r["dump"]["spread"] / spread - 1) > 0.03:
        problems.append(f"{code} {offset:+d} deg: dump spread {r['dump']['spread']:.3e}, formula {spread:.3e}")
assert not problems, "\n".join(problems)
print(f"\nOK ({', '.join(codes)}): 10 -> 100 -> 10 MeV/c, all the energy recovered, the injected beam back at "
      "the dump, and the phase-offset law")
