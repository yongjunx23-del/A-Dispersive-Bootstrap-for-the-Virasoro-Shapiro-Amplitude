# Closed-String Bootstrap Note Reproduction

This folder is a self-contained, GitHub-ready reproduction package for the
main numerical results in the closed-string bootstrap note.

It contains:

- one polished Python module with the reusable functions,
- one short notebook that calls those functions,
- the CSV data needed by the notebook,
- the regenerated PNG/PDF figures,
- this README and a small `requirements.txt`.

The expensive exploratory Gurobi scans are not included here.  This package
reproduces the figures and tables in the note from the saved scan data.

## Main Files

- `closed_string_bootstrap_reproduce.py`
  contains the data loaders, normalizations, summary functions, and plotting
  functions.
- `closed_string_bootstrap_reproduce.ipynb`
  is the notebook version with short explanations.
- `gurobi_c00_bound.py`
  contains a compact Gurobi LP implementation for the gravity-pole
  `max c00/(8 pi G)` illustration.
- `gurobi_c00_bound_illustration.ipynb`
  is a short notebook that uses Gurobi to solve this LP directly.
- `requirements.txt`
  lists the Python packages needed to run the reproduction code.

## Data Files

Gravity-pole bootstrap data:

- `amxsusy_g0_null_convergence.csv`
- `level3_adaptive_final.csv`
- `amxsusy_gravity_pole_c00_c10_direct_c3_scan_band.csv`

Pole-subtracted bootstrap data:

- `pole_subtracted_c10_c20_gurobi_no_NL_boundary.csv`
- `pole_subtracted_c10_c20_gurobi_NL3_boundary.csv`
- `pole_subtracted_c10_c20_gurobi_NL6_band.csv`
- `pole_subtracted_c10_c20_gurobi_NL6_c20_min_spectrum_settings_points.csv`

## Result Figures

Each figure is included as both PNG and PDF:

- `gravity_g0_null_convergence`
- `gravity_allowed_region`
- `pole_subtracted_allowed_region`
- `pole_subtracted_c20_min_spectrum_full`
- `pole_subtracted_c20_min_spectrum_zoom`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The plotting reproduction module only reads saved CSV files and plots results.
The optional `gurobi_c00_bound_illustration.ipynb` notebook does rerun a small
LP and therefore requires `gurobipy` plus a valid Gurobi license.

## Run

From this folder:

```bash
python closed_string_bootstrap_reproduce.py
```

or open:

```text
closed_string_bootstrap_reproduce.ipynb
```

The script regenerates the PNG/PDF figures in this same folder.

To run the Gurobi illustration for the gravity-pole bound, open:

```text
gurobi_c00_bound_illustration.ipynb
```

By default it uses the note resolution

```text
N_mu = 300, J_max = 3000, N_t = 6, 55 null constraints.
```

If this is too slow, set `USE_NOTE_RESOLUTION = False` in the notebook for a
quick smoke test.

## Numerical Normalization

For the gravity-pole `g0` convergence table, the CSV stores the raw solver
normalization.  The note-normalized bound is

```text
c00 M^6 / (8 pi G) = c00_raw / 1e-4.
```

The Virasoro-Shapiro reference values used in the plots are

```text
c00 = 2 zeta(3),
c10 = -2 zeta(5),
c20 = 2 zeta(7).
```

For the pole-subtracted plot, the displayed coordinates are

```text
X = M^4 c10 / c00,
Y = M^8 c20 / c00.
```

## Notes For GitHub

This folder is meant to be uploaded as a clean reproduction package.  The full
research notebooks can remain outside this folder if you want to keep the
repository lightweight and readable.
