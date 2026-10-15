# Numerical companion: A Dispersive Bootstrap for the Virasoro-Shapiro Amplitude

Yongjun Xu. This is the numerical companion to the revised version of the paper
(JHEP submission JHEP_086P_0826). It contains the audited LP records behind
every figure and table, the case configurations, the scripts that solve and
audit the LPs, and the scripts that regenerate the figures and the numbers
quoted in appendix A.

The implementation used for the first version (Gurobi, double precision) is
in the `v1/` folder of
https://github.com/yongjunx23-del/A-Dispersive-Bootstrap-for-the-Virasoro-Shapiro-Amplitude;
this companion is its `revision_v2/` folder.
The LPs here are solved with SDPX, https://github.com/yongjunx23-del/SDPX.rs.

## Contents

| Path | What it is |
|---|---|
| `data/combined_records.json` | All 5245 LP records (5217 accepted by the audit, 28 unresolved). |
| `data/timing_1core_final.json` | SDPX/SDPB comparison behind table 3. |
| `configs/` | Case manifests: truncations, objectives and fixed coefficients for each campaign. |
| `scripts/lp/` | LP construction, SDPX solves, 768-bit audits and the cluster job runner. |
| `scripts/build_figures.py` | Regenerates every figure of the paper from the records. |
| `scripts/build_numerical_summary.py` | Regenerates the appendix A text, table 3 and the numbers quoted in the response. |
| `figures/` | The 15 figure PDFs used in the paper. |

## Settings

Kernels are evaluated at 512 bits, LPs are solved with SDPX at 256 bits, and
every accepted result is audited independently at 768 bits on the original
equations and on all columns of the declared spectral grid. Solver thresholds
are 1e-36 (with the graviton pole) and 1e-30 (pole-subtracted); the audit
threshold is 1e-25. Baselines are
`(N_mu, J_max, N_t, N_null) = (512, 10^4, 6, 55)` with the graviton pole and
`(N_mu, J_max, N_null) = (512, 2000, 78)` pole-subtracted.

## Records

Each record has a `case` name, its `canonical_config` (kind, variant,
truncation, objective, fixed coefficients), a `classification` and the audit
outcome:

- `feasible`: a finite optimum or feasible point, with nonnegative weights,
  equality residuals and objective gap below 1e-25 and nonnegative dual slack
  on every column;
- `infeasible`: excluded by an audited dual ray with `b^T y > 0` and
  `E^T y <= 0` on every column;
- `unbounded`: certified by an audited recession ray;
- `unresolved`: neither check passed. These records never define a plotted
  bound and are never counted as excluded.

The field `verification` names the audit that was applied. Error tracebacks
in a few unresolved records have their cluster paths replaced by `<project>`.

Main record families (`canonical_config.kind`):

| Kind | Used for |
|---|---|
| `glin_direction` | Figure 1 (support function, 192 directions) |
| `glin_c00max` | Linear maximum of c00, figures 2 and 7 |
| `glin_c00max_eikonal` | Supplementary: states appended above the grid (not quoted in the paper); `glin_c00max_E6`, `_E8` feasible, `_E12` and above unbounded |
| `gnl3_slice`, `gnl3_end*`, `gnl3_tip` | Gravitational NL3 region, figures 3 and 4 |
| `gnl6_scan`, `gnl6_tip` | Gravitational NL6 tests, figure 4 |
| `gnl3_convergence`, `gnl6_convergence` | Truncation checks on the string slice, figures 8 and 9; variant `T0` removes the sampled pole rows |
| `pdir`, `pband` | Pole-subtracted NL0/NL3 regions and NL6 island, figures 5 and 11 |
| `pconv` | Pole-subtracted truncation checks and spectra, figures 6 and 10 |
| `pgamma_*`, `pnorm` | Pole-subtracted slices at other values of c00 |

## Regenerating the figures and appendix numbers

Requirements: Python 3.10 or later with NumPy, SciPy, Matplotlib and mpmath
(tested with Python 3.13, NumPy 2.5, SciPy 1.17, Matplotlib 3.10, mpmath 1.3).
From this folder:

```sh
python3 scripts/build_figures.py --data data/combined_records.json
python3 scripts/build_numerical_summary.py --data data/combined_records.json \
  --sdpb data/timing_1core_final.json
```

The first command writes the figures to `figures/` (together with a few
diagnostic figures not used in the paper). The second writes the appendix A
text and table 3 to `tex/`; it reproduces the published appendix A.1 text.

## Rerunning the LPs

`scripts/lp/bootstrap_sdpx_v5.py` builds the kernels and LP for one case and
solves it with SDPX; `verify_results_v9.py` and `audit_unbounded.py` perform
the 768-bit audits; `planner.py`, `queue_worker.py` and `batch_runner.py`
distributed the cases on a PBS cluster. Running them requires an SDPX build.
