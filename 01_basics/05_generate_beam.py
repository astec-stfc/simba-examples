"""
05 - Generate a beam
====================

Shows: SIMBA's Python beam generator (frameworkGenerator) for a Gaussian and
a flat-top bunch: how to set it from Twiss parameters and an emittance, the
units it expects, how its Gaussian cut-off changes the rms values you get,
species and charge, and how to make it reproducible.

Run:    python 01_basics/05_generate_beam.py [--particles N]
Tier:   A
Needs:  SIMBA only.
Output: output/05_generate_beam/gaussian.openpmd.hdf5 and flattop.openpmd.hdf5.

Units the generator expects:
  sigma_x, sigma_y, sigma_z, offset_*:   m
  sigma_t:                               s (used only if sigma_z is 0)
  initial_momentum, sigma_px/py/pz:      eV/c
  charge:                                C, a positive number (the sign
                                         comes from the species)
  covariance_xxp, covariance_yyp:        correlation coefficient (-1, 1)

Things that catch people out:
- The generator takes sizes and divergences, not Twiss parameters; convert
  as below (sigma_x = sqrt(eps beta), sigma_px = p sqrt(eps gamma),
  covariance = -alpha / sqrt(1 + alpha^2)).
- Each Gaussian is cut at gaussian_cutoff_* sigma (default 3). That trims
  the rms size to 0.9866 sigma and the emittance to 0.9733 eps. The Twiss
  parameters are unchanged. Raise the cut-off if you need the exact values.
- For a flat-top, sigma_z is not the rms length: the bunch spans
  +-gaussian_cutoff_z * sigma_z, with short cosine edges.
- There is no seed argument: the generator uses numpy's global random
  state, so call np.random.seed() first for a reproducible beam.

Check: the generated rms sizes, Twiss parameters, emittance, mean momentum
and charge match what was asked for, with the 3-sigma cut taken into
account, to within 5 standard errors of sampling (about 1/sqrt(N)). The same
seed gives the same beam, and a different seed a different one.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm
from simba.Codes.Generators import frameworkGenerator

OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OUT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--particles", type=int, default=100_000)
N = parser.parse_args().particles

# What we want: a 100 MeV/c electron bunch, 100 pC.
p0 = 100e6        # eV/c
charge = 100e-12  # C
m_e = 0.51099895e6
enx, eny = 1e-6, 1e-6           # normalised emittance, m rad
beta_x, alpha_x = 10.0, -1.5    # m, -
beta_y, alpha_y = 5.0, 0.8
sigma_z, sigma_pz = 300e-6, 100e3  # m, eV/c


def transverse(en, beta, alpha):
    """Generator inputs (sigma, sigma_p in eV/c, correlation) from Twiss and emittance."""
    eps = en / (p0 / m_e)  # geometric emittance
    gamma = (1 + alpha ** 2) / beta
    return np.sqrt(eps * beta), p0 * np.sqrt(eps * gamma), -alpha / np.sqrt(1 + alpha ** 2)


sx, spx, cx = transverse(enx, beta_x, alpha_x)
sy, spy, cy = transverse(eny, beta_y, alpha_y)
settings = dict(
    code="simba", species="electron", number_of_particles=N, charge=charge,
    initial_momentum=p0,
    sigma_x=sx, sigma_px=spx, covariance_xxp=cx,
    sigma_y=sy, sigma_py=spy, covariance_yyp=cy,
    sigma_z=sigma_z, sigma_pz=sigma_pz,
)

np.random.seed(2024)
gaussian = frameworkGenerator(distribution_type_z="gaussian", **settings).generate()
gaussian.write_openpmd_beam_file(str(OUT / "gaussian.openpmd.hdf5"))

np.random.seed(2024)
flattop_gen = frameworkGenerator(distribution_type_z="flattop", **settings)
flattop = flattop_gen.generate()
flattop.write_openpmd_beam_file(str(OUT / "flattop.openpmd.hdf5"))

P = gaussian.Particles
print(f"Gaussian beam, {len(gaussian.x)} particles, Q = {gaussian.Q * 1e12:.2f} pC "
      f"({gaussian.species})")
print(f"  <cp> {np.mean(gaussian.cp) / 1e6:.4f} MeV/c, sigma_x {P.sigmas.sigma_x * 1e6:.2f} um, "
      f"sigma_z {P.sigmas.sigma_z * 1e6:.2f} um, sigma_t {P.sigmas.sigma_t * 1e12:.3f} ps")
print(f"  enx {P.emittance.enx * 1e6:.4f} um, beta_x {P.twiss.beta_x:.3f} m, alpha_x {P.twiss.alpha_x:.3f}")
print(f"  eny {P.emittance.eny * 1e6:.4f} um, beta_y {P.twiss.beta_y:.3f} m, alpha_y {P.twiss.alpha_y:.3f}")
print(f"Flat-top beam: sigma_z {flattop.Particles.sigmas.sigma_z * 1e6:.2f} um "
      f"(spans +-{flattop_gen.gaussian_cutoff_z * sigma_z * 1e6:.0f} um)")

# --- Check -----------------------------------------------------------------
# A normal distribution cut at +-a sigma keeps a fraction f of its variance.
a = 3.0
f = 1 - 2 * a * norm.pdf(a) / (2 * norm.cdf(a) - 1)
tol = 5 / np.sqrt(N)  # relative; the standard error of an rms or emittance is ~1/sqrt(2N)..1/sqrt(N)


def close(name, got, want):
    assert abs(got / want - 1) < tol, f"{name}: got {got:.6g}, expected {want:.6g} (+-{tol:.1%})"


close("sigma_x", P.sigmas.sigma_x.val, sx * np.sqrt(f))
close("sigma_y", P.sigmas.sigma_y.val, sy * np.sqrt(f))
close("sigma_z", P.sigmas.sigma_z.val, sigma_z * np.sqrt(f))
close("enx", P.emittance.enx.val, enx * f)
close("eny", P.emittance.eny.val, eny * f)
close("beta_x", P.twiss.beta_x.val, beta_x)
close("beta_y", P.twiss.beta_y.val, beta_y)
# alpha can be near zero, so compare it in absolute terms (it is a ratio).
assert abs(P.twiss.alpha_x.val - alpha_x) < tol * np.sqrt(1 + alpha_x ** 2) * 2, P.twiss.alpha_x
assert abs(P.twiss.alpha_y.val - alpha_y) < tol * np.sqrt(1 + alpha_y ** 2) * 2, P.twiss.alpha_y
assert abs(np.mean(gaussian.cp) / p0 - 1) < 5 * sigma_pz / p0 / np.sqrt(N)
assert np.isclose(gaussian.Q, -charge), gaussian.Q  # electrons: negative
# Arrival time: t = -z / (beta c), so the head (larger z) arrives first.
assert np.corrcoef(gaussian.z, gaussian.t)[0, 1] < -0.999

# Flat-top: the rms of the profile the generator samples (flat over
# +-a sigma_z, cosine edges 0.1 sigma_z wide), by numerical integration.
u = np.linspace(-a, a, 200_001)
edge = np.abs(u) > a - 0.1
w = np.where(edge, 0.5 * (1 + np.cos(np.pi * (np.abs(u) - (a - 0.1)) / 0.1)), 1.0)
flat_rms = sigma_z * np.sqrt(np.sum(w * u ** 2) / np.sum(w))
close("flat-top sigma_z", flattop.Particles.sigmas.sigma_z.val, flat_rms)
print(f"Flat-top: expected rms {flat_rms * 1e6:.2f} um (a full-width W/sqrt(12) estimate gives "
      f"{2 * a * sigma_z / np.sqrt(12) * 1e6:.2f} um)")

# Reproducibility: the generator follows numpy's global seed.
np.random.seed(2024)
again = frameworkGenerator(distribution_type_z="gaussian", **settings).generate()
np.random.seed(7)
other = frameworkGenerator(distribution_type_z="gaussian", **settings).generate()
assert np.array_equal(again.x, gaussian.x) and np.array_equal(again.cpz, gaussian.cpz)
assert not np.array_equal(other.x, gaussian.x)

# Species: a proton beam at the same momentum has the proton rest energy and positive charge.
np.random.seed(1)
proton = frameworkGenerator(**{**settings, "species": "proton", "number_of_particles": 1000}).generate()
assert np.isclose(proton.E0_eV, 938.272e6, rtol=1e-5) and np.isclose(proton.Q, charge)
print(f"\nOK: sizes, Twiss, emittance (x {f:.4f} for the 3-sigma cut), momentum, charge and "
      f"seeding as requested, within +-{tol:.1%}")
