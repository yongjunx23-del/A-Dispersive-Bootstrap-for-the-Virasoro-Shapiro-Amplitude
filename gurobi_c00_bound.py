"""
Small Gurobi LP for the gravity-pole c00 bound.

This module is an intentionally compact version of the gravity-pole `g0`
calculation used in the note.  It maximizes

    c00 M^6 / (8 pi G)

with the fixed-t gravity-pole row and crossing null constraints.  The raw LP
normalization uses `GRAVITY_POLE_RHS_SCALE`; the note-normalized value is
`c00_raw / GRAVITY_POLE_RHS_SCALE`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import hyp2f1


D = 10
GRAVITY_POLE_RHS_SCALE = 1.0e-4


@dataclass(frozen=True)
class G0Config:
    n_mu: int = 300
    j_max: int = 3000
    n_t: int = 6
    max_null_power: int = 10
    feasibility_tol: float = 1.0e-9
    optimality_tol: float = 1.0e-9
    numeric_focus: int = 1
    output_flag: int = 0
    method: int = 2
    crossover: int = -1
    presolve: int = -1
    threads: int = 1


def chebyshev_mu_nodes(n_mu):
    z = np.arange(1, int(n_mu) + 1, dtype=float)
    z = 0.5 - 0.5 * np.cos((2.0 * z - 1.0) * np.pi / (2.0 * int(n_mu)))
    return 1.0 / z


def chebyshev_t_nodes(n_t):
    i = np.arange(1, int(n_t) + 1, dtype=float)
    return -0.5 + 0.5 * np.cos((2.0 * i - 1.0) * np.pi / (2.0 * int(n_t)))


def even_spins(j_max):
    return np.arange(0, int(j_max) + 1, 2, dtype=float)


def p_l_d_x(l, d, x):
    """Normalized Gegenbauer partial wave P_l^(d)(x), with P_l(1)=1."""

    l_arr = np.asarray(l)
    x_arr = np.asarray(x, dtype=float)
    if np.all(np.isfinite(l_arr)) and np.all(np.equal(l_arr, np.rint(l_arr))) and np.min(l_arr) >= 0:
        targets = np.asarray(l_arr, dtype=int).reshape(-1)
        max_l = int(np.max(targets))
        out_shape = np.broadcast_shapes(x_arr.shape, l_arr.shape)
        out = np.empty(out_shape, dtype=float)

        p_prev = np.ones_like(x_arr, dtype=float)
        mask = targets == 0
        if np.any(mask):
            out[:, mask] = p_prev
        if max_l == 0:
            return out

        p_curr = x_arr.copy()
        mask = targets == 1
        if np.any(mask):
            out[:, mask] = p_curr

        lam = 0.5 * (float(d) - 3.0)
        for n in range(1, max_l):
            denom = n + 2.0 * lam
            p_next = (2.0 * (n + lam) / denom) * x_arr * p_curr - (n / denom) * p_prev
            mask = targets == (n + 1)
            if np.any(mask):
                out[:, mask] = p_next
            p_prev, p_curr = p_curr, p_next
        return out

    return hyp2f1(-l_arr, l_arr + d - 3.0, (d - 2.0) / 2.0, (1.0 - x_arr) / 2.0)


def flat(matrix):
    return np.asarray(matrix, dtype=float).reshape(-1)


def generalized_binomial(alpha, n):
    out = 1.0
    for r in range(1, int(n) + 1):
        out *= (alpha - (r - 1.0)) / r
    return out


def convolve_truncated(a, b, order):
    out = np.zeros(order + 1, dtype=float)
    for i, ai in enumerate(a[: order + 1]):
        if ai != 0.0:
            out[i : order + 1] += ai * b[: order + 1 - i]
    return out


def polynomial_power_truncated(poly, power, order):
    out = np.zeros(order + 1, dtype=float)
    out[0] = 1.0
    for _ in range(int(power)):
        out = convolve_truncated(out, poly, order)
    return out


def x_of_a_over_mu_series(order):
    lhs = np.asarray([generalized_binomial(0.5, n) * 3.0**n for n in range(order + 1)], dtype=float)
    rhs = np.asarray([generalized_binomial(-0.5, n) * (-1.0) ** n for n in range(order + 1)], dtype=float)
    return convolve_truncated(lhs, rhs, order)


def poly_mul_linear(poly, constant, linear, scale=1.0):
    out = np.zeros(len(poly) + 1, dtype=float)
    out[: len(poly)] += constant * poly
    out[1:] += linear * poly
    return scale * out


def trim_polynomial(poly, tol=0.0):
    poly = np.asarray(poly, dtype=float)
    last = len(poly) - 1
    while last > 0 and abs(poly[last]) <= tol:
        last -= 1
    return poly[: last + 1].copy()


def taylor_polynomial_coefficients(k, power, d=D):
    """Coefficient polynomial b_{k,power}(L) for the fixed-a expansion."""

    k = int(k)
    power = int(power)
    order = power

    x_series = x_of_a_over_mu_series(order)
    u_series = -0.5 * x_series
    u_series[0] += 0.5
    u_powers = [polynomial_power_truncated(u_series, r, order) for r in range(order + 1)]

    partial_wave_polys = [np.zeros(1, dtype=float) for _ in range(order + 1)]
    h_poly = np.ones(1, dtype=float)
    for r in range(order + 1):
        for n in range(order + 1):
            coeff = u_powers[r][n]
            if coeff == 0.0:
                continue
            if len(partial_wave_polys[n]) < len(h_poly):
                enlarged = np.zeros(len(h_poly), dtype=float)
                enlarged[: len(partial_wave_polys[n])] = partial_wave_polys[n]
                partial_wave_polys[n] = enlarged
            partial_wave_polys[n][: len(h_poly)] += coeff * h_poly
        if r < order:
            denominator = (((d - 2.0) / 2.0 + r) * (r + 1.0))
            h_poly = poly_mul_linear(h_poly, r * (r + d - 3.0), -1.0, scale=1.0 / denominator)

    prefactor = np.asarray(
        [generalized_binomial(k / 2.0 - 1.0, n) * (-1.0) ** n for n in range(order + 1)],
        dtype=float,
    )
    prefactor_times_linear = np.zeros(order + 1, dtype=float)
    prefactor_times_linear += 2.0 * prefactor
    prefactor_times_linear[1:] += -3.0 * prefactor[:-1]

    coeff_poly = np.zeros(1, dtype=float)
    for i, ai in enumerate(prefactor_times_linear):
        if ai == 0.0:
            continue
        source = partial_wave_polys[order - i]
        if len(coeff_poly) < len(source):
            enlarged = np.zeros(len(source), dtype=float)
            enlarged[: len(coeff_poly)] = coeff_poly
            coeff_poly = enlarged
        coeff_poly[: len(source)] += ai * source

    return trim_polynomial(coeff_poly)


def evaluate_polynomial(coeffs, x):
    out = np.zeros_like(x, dtype=float) + coeffs[-1]
    for coeff in coeffs[-2::-1]:
        out = out * x + coeff
    return out


def null_tower(max_power):
    """Crossing null rows through a^max_power."""

    return [(k, power) for power in range(1, int(max_power) + 1) for k in range(0, 2 * power, 2)]


def build_g0_lp(config: G0Config):
    """Build dense equality matrix A x = b for x = [c00_raw, rho]."""

    mu_values = chebyshev_mu_nodes(config.n_mu)
    j_values = even_spins(config.j_max)
    mu = mu_values[:, None]
    ell = j_values[None, :]
    ell_casimir = ell * (ell + 7.0)
    n_rho = mu_values.size * j_values.size

    rows = []
    rhs = []
    names = []

    row = np.zeros(1 + n_rho, dtype=float)
    row[0] = -1.0
    row[1:] = 2.0
    rows.append(row)
    rhs.append(0.0)
    names.append("k=0: 2 sum rho - c00_raw")

    for t in chebyshev_t_nodes(config.n_t):
        row = np.zeros(1 + n_rho, dtype=float)
        x = 1.0 + 2.0 * float(t) / mu
        row[1:] = flat(mu * (2.0 * mu + float(t)) * p_l_d_x(ell, D, x))
        rows.append(row)
        rhs.append(-GRAVITY_POLE_RHS_SCALE / float(t))
        names.append(f"fixed-t gravity pole, t={t:.12g}")

    kernel_cache = {}
    for spec in null_tower(config.max_null_power):
        if spec not in kernel_cache:
            coeffs = taylor_polynomial_coefficients(*spec, d=D)
            kernel_cache[spec] = flat(evaluate_polynomial(coeffs, ell_casimir) / mu ** (spec[0] + spec[1]))
        row = np.zeros(1 + n_rho, dtype=float)
        row[1:] = kernel_cache[spec]
        rows.append(row)
        rhs.append(0.0)
        names.append(f"null k={spec[0]}, a^{spec[1]}")

    return np.vstack(rows), np.asarray(rhs, dtype=float), names, mu_values, j_values


def solve_c00_bound(config: G0Config):
    """Solve the c00 bound with Gurobi and return a compact result dictionary."""

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover
        raise ImportError("This illustration notebook needs gurobipy and a valid Gurobi license.") from exc

    A, b, names, mu_values, j_values = build_g0_lp(config)
    n = A.shape[1]
    lb = np.zeros(n, dtype=float)
    lb[0] = -GRB.INFINITY

    model = gp.Model("gravity-pole-c00-bound")
    model.Params.OutputFlag = int(config.output_flag)
    model.Params.Method = int(config.method)
    model.Params.Crossover = int(config.crossover)
    model.Params.Presolve = int(config.presolve)
    model.Params.FeasibilityTol = float(config.feasibility_tol)
    model.Params.OptimalityTol = float(config.optimality_tol)
    model.Params.NumericFocus = int(config.numeric_focus)
    model.Params.Threads = int(config.threads)
    if config.method == 2:
        model.Params.BarHomogeneous = 1
        model.Params.BarConvTol = 1.0e-10

    x = model.addMVar(n, lb=lb, name="x")
    model.addMConstr(A, x, "=", b, name="eq")
    model.setObjective(-x[0], GRB.MINIMIZE)
    model.optimize()

    ok = model.Status == GRB.OPTIMAL
    c00_raw = float(x.X[0]) if ok else np.nan
    return {
        "ok": bool(ok),
        "status": int(model.Status),
        "status_name": {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.NUMERIC: "NUMERIC",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, str(model.Status)),
        "c00_raw": c00_raw,
        "c00_over_8piG": c00_raw / GRAVITY_POLE_RHS_SCALE if ok else np.nan,
        "num_variables": int(n),
        "num_equalities": int(A.shape[0]),
        "num_null_constraints": len(null_tower(config.max_null_power)),
        "n_mu": int(config.n_mu),
        "j_max": int(config.j_max),
        "n_t": int(config.n_t),
        "max_null_power": int(config.max_null_power),
        "mu_values": mu_values,
        "j_values": j_values,
        "row_names": names,
    }
