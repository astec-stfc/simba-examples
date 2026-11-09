"""
07 - Analyse and plot a beam
============================

Shows: the beam properties SIMBA computes (sizes, emittance, Twiss
parameters, slice properties, current profile) and its plotting functions:
plotScreenImage (a camera-like image), marginal_plot (a 2D histogram with
its projections) and slice_plot (slice properties along the bunch). The
beam is a flat-top bunch with a linear energy chirp, so the plots show
something.

Run:    python 01_basics/07_beam_analysis_and_plots.py [--particles N]
Tier:   A
Needs:  SIMBA (and matplotlib).
Output: output/07_beam_analysis_and_plots/*.png and the beam file.

Where things are on a beam `b` (all are UnitValue arrays or numbers in SI
units; `.val` gives the plain numpy value):
  b.x, b.y, b.z, b.t, b.cpx, b.cpy, b.cpz (eV/c), b.cp, b.Q (C)
  b.Particles.sigmas.sigma_x / sigma_z / sigma_t / momentum_spread (eV/c, not relative)
  b.Particles.emittance.enx / eny          normalised, m rad
  b.Particles.twiss.beta_x / alpha_x / ...
  b.Particles.slice.set_slices(n), then slice_t, slice_current,
      slice_momentum, slice_normalized_horizontal_emittance, peak_current
The plotting functions draw into the current matplotlib figure and return
nothing; give filename= to save the figure.

Known bug: plotScreenImage needs a beam that was read from a file (it uses
the file name for the title), so the beam is written and read back first.
See ISSUES.md.

Check:
- the current on the flat top equals Q divided by the flat-top duration (the
  generator's profile: flat over +-3 sigma_z with short cosine edges);
- the slope of slice momentum against time equals the chirp put in;
- SIMBA's Twiss parameters equal those computed directly from the second
  moments of x and x' = px/pz.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; remove to show the figures on screen
import matplotlib.pyplot as plt
import numpy as np
import simba.Modules.Beams as rbf
from simba.Codes.Generators import frameworkGenerator
from simba.Modules.Beams.plot import marginal_plot, plotScreenImage, slice_plot

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--particles", type=int, default=200_000)
N = parser.parse_args().particles

c = 299_792_458.0
p0, charge, sigma_z = 100e6, 100e-12, 100e-6   # eV/c, C, m
chirp = 2e9                                    # eV/c per m of z (head, larger z, has more momentum)

np.random.seed(1)
generated = frameworkGenerator(
    code="simba", number_of_particles=N, charge=charge, initial_momentum=p0,
    sigma_x=100e-6, sigma_px=1e3, covariance_xxp=0.5, sigma_y=100e-6, sigma_py=1e3,
    sigma_z=sigma_z, sigma_pz=10e3, distribution_type_z="flattop", chirp=chirp,
).generate()
path = OUT / "flattop_chirped.openpmd.hdf5"
generated.write_openpmd_beam_file(str(path))
b = rbf.load_file(str(path))  # read back: plotScreenImage needs b.filename (see above)

P = b.Particles
print(f"{len(b.x)} particles, Q {b.Q * 1e12:.1f} pC, <cp> {np.mean(b.cp) / 1e6:.3f} MeV/c")
print(f"sigma_x {P.sigmas.sigma_x * 1e6:.1f} um, sigma_t {P.sigmas.sigma_t * 1e15:.1f} fs, "
      f"rms momentum spread {P.sigmas.momentum_spread.val / 1e3:.1f} keV/c")
print(f"enx {P.emittance.enx * 1e6:.4f} um, beta_x {P.twiss.beta_x:.4f} m, alpha_x {P.twiss.alpha_x:.4f}")

P.slice.set_slices(50)
t, current = P.slice.slice_t, P.slice.slice_current.val
momentum = P.slice.slice_momentum.val
print(f"peak current {P.slice.peak_current:.2f} A (from 50 slices)")

plotScreenImage(b, keys=["x", "y"], scale=[1e3, 1e3], filename=str(OUT / "screen_xy"))
plt.close("all")
marginal_plot(b, key1="t", key2="cpz", bins=60, subtract_mean=[True, False], filename=str(OUT / "t_cpz"))
plt.close("all")
slice_plot(b, bins=50, filename=str(OUT / "slices"))
plt.close("all")
print(f"Plots written to {OUT}")

# --- Check -----------------------------------------------------------------
# Flat-top current. The generator samples z flat over +-3 sigma_z with cosine
# edges 0.1 sigma_z wide; the duration of an equivalent flat bunch is that
# profile's integral (in units of sigma_z) times sigma_z / (beta c).
u = np.linspace(-3, 3, 600_001)
w = np.where(np.abs(u) > 2.9, 0.5 * (1 + np.cos(np.pi * (np.abs(u) - 2.9) / 0.1)), 1.0)
beta = np.mean(b.Bz.val)
duration = np.trapezoid(w, u) * sigma_z / (beta * c)
expected_current = charge / duration
middle = np.abs(t - np.mean(b.t.val)) < 0.25 * duration  # the central half of the flat top
plateau = np.mean(current[middle])
# Each slice holds ~N/50 particles: noise ~sqrt(50/N) per slice, averaged over the middle ones.
tol = 4 * np.sqrt(50 / N) / np.sqrt(middle.sum())
print(f"\nFlat-top current {plateau:.3f} A, expected Q/T = {expected_current:.3f} A "
      f"(the rough Q/(sqrt(12) sigma_t) gives {charge / (np.sqrt(12) * P.sigmas.sigma_t.val):.3f} A)")
assert abs(plateau / expected_current - 1) < tol, f"plateau {plateau} A vs {expected_current} A (+-{tol:.1%})"

# Chirp: z = -beta c t, so the momentum changes with t at -chirp * beta * c.
filled = current > 0
slope = np.polyfit(t[filled], momentum[filled], 1)[0]
print(f"Slice momentum slope {slope:.4e} eV/c per s, expected {-chirp * beta * c:.4e}")
assert abs(slope / (-chirp * beta * c) - 1) < 0.01, slope

# Twiss parameters from the second moments.
x, xp = b.x.val, b.cpx.val / b.cpz.val
cov = np.cov(x, xp, bias=True)
eps = np.sqrt(np.linalg.det(cov))
print(f"From moments: eps_x {eps * 1e9:.4f} nm, beta_x {cov[0, 0] / eps:.4f} m, alpha_x {-cov[0, 1] / eps:.4f}")
assert np.isclose(P.twiss.beta_x.val, cov[0, 0] / eps, rtol=1e-6), P.twiss.beta_x
assert np.isclose(P.twiss.alpha_x.val, -cov[0, 1] / eps, rtol=1e-6, atol=1e-9), P.twiss.alpha_x
assert np.isclose(P.emittance.ex.val, eps, rtol=1e-6), P.emittance.ex
print("\nOK: flat-top current, chirp and Twiss parameters as expected")
