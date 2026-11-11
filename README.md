# SIMBA examples

Examples for [SIMBA](https://github.com/astec-stfc/simba) (running and
comparing beam-tracking codes) and [LAURA](https://github.com/astec-stfc/laura)
(the lattice description it reads). Each example is a short script on one
topic. It ends with a physics check (`assert`s on a quantity with a known
answer) so you can tell it worked. Each group also has a notebook that
explains the scripts and runs them.

The examples follow SIMBA `main` and LAURA `main` (see *Versions* below) and
use the public example machines in
[laura-lattices](https://github.com/astec-stfc/laura-lattices), mostly JFEL.

## Install

Needs Python 3.10 or newer and git.

```bash
git clone https://github.com/astec-stfc/simba
git clone https://github.com/astec-stfc/laura
git clone https://github.com/astec-stfc/laura-lattices
git clone <this repository> simba-examples
cd simba-examples
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ../laura -e "../simba[simcodes]" -e ../laura-lattices jupyter nbconvert ipykernel
.venv/bin/python 01_basics/01_environment_check.py
```

The `[simcodes]` extra installs the Python tracking codes (Ocelot, Cheetah,
Xsuite, MAD-X through cpymad, and others). It is large, because it pulls in
torch.

To reproduce the exact versions instead (no editable clones):
`.venv/bin/pip install -r requirements.txt`.

`01_environment_check.py` lists what is installed and which tiers of
examples you can run. It exits with an error if SIMBA, LAURA or
laura-lattices is missing or too old.

## Tiers: what each example needs

Every script states its tier in its docstring (`Tier:`).

| Tier | Needs | Codes |
|------|-------|-------|
| **A** | the pip install above | Ocelot, Cheetah, Xsuite, MAD-X (cpymad), Wake-T |
| **B** | a container runtime and the simcodes image | elegant, ASTRA, OPAL, CSRTrack, Genesis |
| **C** | a specialist install | Bmad/Tao, ImpactX, FBPIC, BDSIM; GPT (own licence) |

**Tier B (containers).** SIMBA runs the compiled codes in a container when
you pass `container_runtime="docker"` or `"apptainer"` to `Framework`. Get
the image with

```bash
docker pull ghcr.io/astec-stfc/simcodes-docker:master
```

(about 17 GB). Do not try to build the image from source: desy.de, where
some of its sources come from, is down, so the build fails. Apptainer
(`container_runtime="apptainer"`) works where docker is not allowed, for
example on clusters.

**Tier C (specialist installs).** Only the two tier-C examples need these,
and each prints `SKIP` without its code:

- BDSIM (31): the `bdsim` binary on `PATH`, ROOT's Python bindings and
  `pip install pybdsim`. Source ROOT's `thisroot.sh` and BDSIM's `bdsim.sh`
  before starting Python. If you set `PYTHONPATH` yourself, keep ROOT's
  `lib` on it.
- Bmad (32): `pip install pytao` and a Bmad build, with
  `export ACC_ROOT_DIR=<bmad-ecosystem>` (SIMBA loads
  `$ACC_ROOT_DIR/production/lib/libtao.so`). See *Troubleshooting* for
  `LD_LIBRARY_PATH`.

**GPT** has a proprietary licence. No example uses it. **Not covered:**
FBPIC (needs a GPU; 29 shows the one-line switch) and Ptarmigan (not on
SIMBA's `main` yet).

## Examples

| Group | Script | Shows | Check |
|-------|--------|-------|-------|
| 01 basics | `01_environment_check.py` | installed versions, tracking codes, container runtime | everything required is present |
| | `02_build_lattice_python.py` | build a FODO lattice in Python with LAURA, write it as YAML, read it back | names, positions, strengths and length survive the round trip |
| | `03_load_machine.py` | load JFEL in SIMBA and LAURA; lines, elements, s positions | SIMBA and LAURA list the same elements at the same s |
| | `04_the_def_file.py` | a commented `.def` (`jfel_example.def`): lines, codes, beam files, groups, changes | each setting is what the file says |
| | `05_generate_beam.py` | the Python beam generator: Gaussian and flat-top, Twiss inputs, units, seeding | rms sizes, emittance, Twiss and charge to within sampling error |
| | `06_beam_io_and_conversion.py` | read and write openPMD, ASTRA, SDDS, Ocelot, GDF | round trips keep N, charge and moments |
| | `07_beam_analysis_and_plots.py` | emittance, Twiss, slices, current; SIMBA's plot functions | flat-top current, chirp and Twiss from moments |
| 02 tracking | `08_track_drift.py` | the smallest tracking run: a drift, one line, Ocelot | beam size from the starting moments; N, charge, emittance kept |
| | `09_track_fodo.py` | FODO cells; reading a run back with `load_directory`; envelope plot | Σ = MΣ₀Mᵀ at every screen (thick-lens matrices) |
| | `10_twiss_without_particles.py` | periodic optics (`periodic: true`, `ring_parameters()`) | tune and β from the cell matrix |
| | `11_change_code.py` | `change_Lattice_Code`: one line in Xsuite, Ocelot, Cheetah, MAD-X (and elegant) | codes agree particle by particle to 1e-3 |
| | `12_start_to_end_handoff.py` | lines in different codes chained in one `.def` | the chain matches a single-code run at each boundary |
| | `13_jfel_linac.py` | JFEL from the injector beam to the undulator; summary per section | energy, emittance and size match the JFEL README |
| | `14_compare_codes.py` | JFEL in several codes, compared at every screen | each pair within SIMBA's cross-code ceilings |
| 03 modifying | `15_modify_elements.py` | `getElement`, `modifyElement`, `setElementType`, `apply_changes`, changes files | new strengths reach the deck; flipped quads swap the focusing planes |
| | `16_groups_chicane.py` | `groups:` chicane, `set_angle`; JFEL's bunch compressor on its own | tracked R56 = exact formula + S/γ², at two angles, in three codes |
| | `17_rf_phase_scan.py` | RF cavity phase scan; SIMBA's phase and voltage conventions | ΔE = V cos(φ − φ₀), crest at 0, V from the travelling-wave formula |
| | `18_errors_and_misalignments.py` | `physical.error.position`, seeded random offsets, `loadElementErrors` | centroid follows the offset-quad map; same seed, same orbit |
| | `19_dispersion_and_solenoid.py` | dispersion from a dogleg; solenoid rotation | η and η′ = 0 at the chicane centre; rotation = S0L/(2Bρ) |
| 04 scans, optimisation | `20_quad_scan_emittance.py` | quad scan; thin-lens parabola and exact fits | fitted ε = the beam's ε (exact fit 1e-3, thin lens 5%) |
| | `21_optimise_scipy.py` | `scipy.optimize` around `track()`; trial log; best setting to a changes file | optimum = thick-quad matrix optimum to 1e-4 |
| | `22_optimise_compression.py` | JFEL L4H phase for 1000 A peak current; the same answer in other codes | 1000 A within 1%; without wakes Cheetah = Ocelot; wakes change I > 1.5x |
| | `23_match_with_def.py` | `match:` block in a `.def` (SIMBA's matching, in the line's code, Ocelot); periodic target from SIMBA | beam Twiss at the undulator = periodic Twiss; README table; stored S13 strengths |
| | `24_parameter_scan_parallel.py` | angle × phase scan over processes, one directory per run, CSV | compression valley rises with the angle, within 5° of the README's |
| 05 advanced | `25_collective_effects.py` | `csr_enable`, `lsc_enable`, `wakefield_enable`, one at a time, on JFEL's linac (Ocelot, elegant) | wakes shorten σz > 30%; CSR grows εx > 2x, not εy; README table |
| | `26_injector_from_cathode.py` | JFEL's ASTRA injector from the cathode, seeded (B) | the saved S02 beam: energy, charge, σt, σx; ε within sampling |
| | `27_ring_tune_da.py` | a ring (`geometry: closed`): tunes, closed orbit, dynamic aperture, frequency map | tunes = matrix tunes to 1e-6; orbit and DA against an independent tracker (XFAIL #30–#32) |
| | `28_erl_multipass.py` | an ERL: LAURA multipass, per-pass cavity phase (Ocelot, elegant) | 10 → 100 → 10 MeV/c; energy recovered; dump beam = injected; 10 + 90(1 − cos φ) |
| | `29_plasma_wake_t.py` | beam-driven plasma stage in Wake-T, with and without matching ramps | driver loses, witness gains; README witness table; ramps grow β 5x |
| | `30_fel_genesis.py` | elegant linac into Genesis 4, steady-state (B) | λ = λu(1+K²)/(2γ²); gain length 1–2x Ming Xie; README row |
| | `31_secondaries_bdsim.py` | `Target`, `Crystal`, secondaries as species; one species on to Xsuite (C) | E0 exp(−t/X0); README yields within statistics; channelled fraction |
| | `32_bmad_multipass.py` | 28's ERL in Bmad, flattened and as Bmad multipass (C) | native = flattened to 1e-4; README Bmad column |
| 06 matching | `33_match_in_any_code.py` | 23's match in each code (`code:`); the default, and a hand-over with its reason | the tracked beam on target in every code; Ocelot's = stored strengths |
| | `34_matrix_targets.py` | `R12 = R34 = 0`: point-to-point imaging, in Xsuite's coordinates whatever the code | tracked R12, R34 < 1e-4 m (a fit with chromatic terms); Cheetah declines |
| | `35_chicane_r56.py` | a chicane group's `angle` as the variable, for an R56 | angle back to 0.12 rad; tracked R56 to 1e-5; layout, edges, `dipoleangle` follow |
| | `36_longitudinal_match.py` | `longitudinal_match`: linked S-band + X-band for a chirp, linearised; closing the loop | chirp 0.2%, curvature and skewness < 1% of nominal, with and without wakes |
| | `37_compression_by_matching.py` | chicane R56 and RF together for C = 4, linear (curvature cancels T566) | 1/a1 = C to 2e-3; a2 < 2% of the uncompensated, which = R56 h2 + T566 h² |

Notebooks: `01_basics/01_basics.ipynb`, `02_tracking/02_tracking.ipynb`,
`03_modifying/03_modifying.ipynb`,
`04_scans_optimisation/04_scans_optimisation.ipynb`,
`05_advanced/05_advanced.ipynb`, `06_matching/06_matching.ipynb`. Start Jupyter with
`.venv/bin/jupyter lab`. The notebooks are committed without outputs.
Each notebook holds its scripts' code, one script per cell, with the
explanation between them. The scripts are the source: after changing one,
run `.venv/bin/python sync_notebooks.py` to copy it into its notebook
(`check_all.py` fails if a notebook is out of date).

Each script writes its output to `output/<script name>/`, which git
ignores, and never writes into the lattice packages.

## Running all the checks

```bash
.venv/bin/python check_all.py                 # tiers A and B (B is skipped if there is no container runtime)
.venv/bin/python check_all.py --tiers A       # pip-only examples
.venv/bin/python check_all.py --tiers A,B,C   # also BDSIM and Bmad (31, 32)
.venv/bin/python check_all.py 01_basics/05_generate_beam.py
```

Each script runs in a fresh Python process, and then the notebooks run. The
summary table counts `XFAIL` (a known bug, listed in [ISSUES.md](ISSUES.md))
and `XPASS` (a known bug that no longer shows; time to update the example).
Logs go to `output/check_all/`.

## Versions

Tested again, scripts and notebooks, on 2026-11-10 with:

| Package | Commit | Version |
|---------|--------|---------|
| SIMBA `main` | 26aed74 | 0.7.1.post5 |
| LAURA `main` | 2317300 | 1.1.5 |
| laura-lattices `main` | b0cf4d8 | 1.0.0 |

The examples use SIMBA's current names (`simba.Framework`, `loadSettings`,
`beam.Particles`). An open SIMBA PR renames these in PEP-8 style, keeping
the old names with a deprecation warning (see ISSUES.md).

Group 06 (and the updated 16 and 23) use SIMBA's matching module
(`simba.Modules.Matching`), on SIMBA's `feature/matching-overhaul` until it
is released; on SIMBA `main`, 33-37 and 16 fail. Tested on 2026-11-10 with
that branch (b53cef8), LAURA `main` 2317300 and laura-lattices `main` b0cf4d8.

## Troubleshooting

- **`01_environment_check.py` says OLD or MISSING:** the venv is not the one
  you installed into, or a clone is behind. Run `git pull` in the clones,
  then repeat the `pip install -e` line.
- **`ModuleNotFoundError: laura_lattices`:** install the public
  `laura-lattices` clone (the pip line above).
- **Tier B examples are skipped:** `docker image inspect
  ghcr.io/astec-stfc/simcodes-docker:master` must succeed and the docker
  daemon must be running. Otherwise install apptainer.
- **Bmad/Tao (tier C): `OSError: ... libfgsl.so.1: undefined symbol
  gsl_linalg_ldlt_band_solve`:** another GSL on your library path (for
  example `/usr/local/lib`, or ROOT's) is loaded instead of the one built
  with Bmad. Put Bmad's libraries first:
  `export LD_LIBRARY_PATH=<bmad-ecosystem>/production/lib:$LD_LIBRARY_PATH`.
- **An example fails with a bug in SIMBA or LAURA:** check
  [ISSUES.md](ISSUES.md), which lists the known ones with reproducers.
