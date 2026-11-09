"""
17 - Scan an RF cavity's phase and find the crest
=================================================

Shows: an RF cavity built in Python (LAURA `RFCavity`), its `cavity.phase`
and `simulation.field_amplitude`, and a phase scan with `modifyElement`:
the beam's energy gain against phase, fitted to find the crest and the
cavity's voltage. Run in Ocelot, Cheetah and Xsuite.

Run:    python 03_modifying/17_rf_phase_scan.py
Tier:   A
Needs:  SIMBA, LAURA, Ocelot, Cheetah and Xsuite.
Output: output/17_rf_phase_scan/: the lattice, one run directory per code,
        and phase_scan.png.

Units: phase in degrees; field_amplitude in V/m (travelling wave) or V
(standing wave); energy in eV; frequency in Hz.

The cavity: a 1 m S-band (2998.5 MHz) travelling-wave structure, 30 cells
in 2pi/3 mode, field_amplitude 20 MV/m, no field map. The beam: 100 MeV,
10 um long, so all of it sees the same phase.

SIMBA's conventions:
- phase 0 is on crest (largest energy gain), whatever the code's own
  convention; SIMBA converts.
- A standing-wave cavity's field_amplitude is its voltage.
- A travelling-wave cavity's field_amplitude is a gradient, and the voltage
  is field_amplitude * (N + 3.8) * cell_length / sqrt(2), with N the number
  of cells rounded down to a multiple of 3. That is LAURA's
  RFCavity.voltage_factor (laura/translator/converters/cavity.py), which
  every code's deck uses: 15.93 MV here.
- LAURA has a second estimate, tw_cavity_energy_gain, which gives
  field_amplitude * sin(pi/3) * N * cell_length = 17.32 MV, 8.7% more.
  Only GPT uses it (to scale magnets after a cavity with no field map), so
  it does not affect these codes.

Avoid +-90 deg exactly: Cheetah's cavity matrix divides by cos(phase) and
returns NaN there (both cavity types), as does Ocelot's travelling-wave
cavity at +90 deg. The scan below steps from -165 to 165 deg by 30 deg.

Fit the energy, sqrt((cp)^2 + (mc^2)^2), not the momentum: at 100 MeV the
momentum gain is (1 + (mc^2)^2/(2E^2)) times the energy gain, which puts
1.3e-5 on V and a curvature on the scan.

Check: in every code the mean energy gain must follow
dE = V cos(phase - phi0) with the fit's worst residual below 1e-4 of V.
Ocelot and Cheetah fit it exactly (residuals of 1e-8 eV, rounding). Xsuite's
residual is 5e-5 of V (0.8 keV) and odd in phase: it follows the beam's arrival time through the cavity, and at
100 MeV the beam is not quite at the speed of light, so a particle that
gains energy moves forward in phase by a little. It falls as 1/E^3
(measured: 27 times smaller at 300 MeV). The crest phi0 must be 0 to
0.01 deg, and V must equal the voltage_factor formula to 1e-4 (measured
2e-7).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simba.Framework as fw
import simba.Modules.Beams as rbf
from laura.models.element import RFCavity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_tracking"))
from simple_lattice import AREA, M_E, make_beam, marker, write_master_lattice  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LATTICE = OUT / "lattice"
CODES = ["ocelot", "cheetah", "xsuite"]
GRADIENT, N_CELLS, CELL = 20e6, 30, 1 / 30

cavity = RFCavity(
    name="CAV", machine_area=AREA, physical={"length": 1.0, "middle": {"z": 1.0}},
    cavity={"frequency": 2998.5e6, "phase": 0.0, "n_cells": N_CELLS, "cell_length": CELL,
            "structure_type": "TravellingWave", "mode_numerator": 1, "mode_denominator": 3},
    simulation={"field_amplitude": GRADIENT},
)
files = {
    "LINE": {
        "code": CODES[0],
        "input": {"particle_definition": "$master_lattice$START"},
        "output": {"start_element": "START", "end_element": "END"},
    }
}
settings = write_master_lattice(LATTICE, [marker("START", 0.0), cavity, marker("END", 2.0)], files, "rf.def")
start = make_beam(LATTICE / "START.openpmd.hdf5", particles=1000, sigma_z=10e-6)


def energy(beam):
    """Mean total energy (eV)."""
    return np.mean(np.hypot(beam.cp.val, M_E))


e0 = energy(start)

voltage = GRADIENT * (N_CELLS - N_CELLS % 3 + 3.8) * CELL / np.sqrt(2)
phases = np.arange(-165.0, 166.0, 30.0)
fit_matrix = np.c_[np.cos(np.radians(phases)), np.sin(np.radians(phases))]

print(f"Expected voltage (voltage_factor): {voltage / 1e6:.4f} MV\n")
print(f"{'code':8s} {'V [MV]':>9s} {'V/expected-1':>13s} {'crest [deg]':>12s} {'worst resid / V':>16s}")
scans, problems = {}, []
for code in CODES:
    framework = fw.Framework(master_lattice=str(LATTICE), directory=str(OUT / code), clean=True, verbose=False)
    framework.loadSettings(settings)
    framework.change_Lattice_Code("LINE", code)
    gain = []
    for phase in phases:
        framework.modifyElement("CAV", "phase", float(phase))
        framework.track()
        gain.append(energy(rbf.load_file(str(OUT / code / "END.openpmd.hdf5"))) - e0)
    gain = np.array(gain)
    (a, b), *_ = np.linalg.lstsq(fit_matrix, gain, rcond=None)  # V cos(p - p0) = a cos p + b sin p
    v, crest = np.hypot(a, b), np.degrees(np.arctan2(b, a))
    resid = np.max(np.abs(fit_matrix @ [a, b] - gain))
    scans[code] = gain
    print(f"{code:8s} {v / 1e6:9.4f} {v / voltage - 1:13.1e} {crest:12.4f} {resid / v:16.1e}")
    if resid > 1e-4 * voltage or abs(crest) > 0.01 or abs(v / voltage - 1) > 1e-4:
        problems.append(f"{code}: V {v:.1f} (expected {voltage:.1f}), crest {crest:.4f} deg, residual {resid:.1e} eV")

fine = np.linspace(-180, 180, 361)
for code, gain in scans.items():
    plt.plot(phases, gain / 1e6, "o", label=code, alpha=0.7)
plt.plot(fine, voltage / 1e6 * np.cos(np.radians(fine)), "k-", lw=0.8, label="V cos(phase)")
plt.xlabel("phase [deg]")
plt.ylabel("energy gain [MeV]")
plt.legend()
plt.savefig(OUT / "phase_scan.png", dpi=120, bbox_inches="tight")
print(f"\nPlot: {OUT / 'phase_scan.png'}")

# --- Check -----------------------------------------------------------------
assert not problems, "\n".join(problems)
print("\nOK: every code puts the crest at 0 deg and gives the voltage_factor voltage")
