#!/usr/bin/env python3
"""High-precision finite-grid bootstrap with an audited SDPX exchange solve.

All mathematical kernels use exact rationals and 512-bit MPFR. Binary64 is
used only by the explicitly separate SciPy comparison and plotting stages.
"""
import argparse
import csv
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import time
from fractions import Fraction as F
from functools import lru_cache
from collections import OrderedDict

import gmpy2 as g

g.get_context().precision = 512
from energy_nodes import energy_nodes
M = g.mpfr
ZERO = M(0)
ONE = M(1)
POLE_CACHE = OrderedDict()


def dec(x):
    return format(M(x), ".150g")


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binom(a, n):
    out = F(1)
    for r in range(1, n + 1):
        out *= (a - r + 1) / r
    return out


def convolution(a, b, order):
    out = [F(0)] * (order + 1)
    for i, ai in enumerate(a[:order + 1]):
        for j, bj in enumerate(b[:order + 1 - i]):
            out[i + j] += ai * bj
    return out


def power(a, n, order):
    out = [F(1)] + [F(0)] * order
    for _ in range(n):
        out = convolution(out, a, order)
    return out


@lru_cache(None)
def qpoly(k, order, d=10):
    """Exact coefficient of z^order in the manuscript's fixed-a kernel."""
    left = [binom(F(1, 2), n) * 3**n for n in range(order + 1)]
    right = [binom(F(-1, 2), n) * (-1)**n for n in range(order + 1)]
    xseries = convolution(left, right, order)
    useries = [-x / 2 for x in xseries]
    useries[0] += F(1, 2)
    upowers = [power(useries, r, order) for r in range(order + 1)]
    partial = [[F(0)] * (order + 1) for _ in range(order + 1)]
    h = [F(1)]
    for r in range(order + 1):
        for n in range(order + 1):
            for j, hj in enumerate(h):
                partial[n][j] += upowers[r][n] * hj
        if r < order:
            denom = (F(d - 2, 2) + r) * (r + 1)
            nh = [F(0)] * (len(h) + 1)
            for j, hj in enumerate(h):
                nh[j] += hj * r * (r + d - 3) / denom
                nh[j + 1] -= hj / denom
            h = nh
    pref = [binom(F(k, 2) - 1, n) * (-1)**n for n in range(order + 1)]
    pref = [2 * pref[n] - (3 * pref[n - 1] if n else 0) for n in range(order + 1)]
    out = [sum(pref[i] * partial[order - i][j] for i in range(order + 1))
           for j in range(order + 1)]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def mp_rat(x):
    return M(x.numerator) / M(x.denominator)


def horner(coeff, x):
    out = ZERO
    for a in reversed(coeff):
        out = out * x + a
    return out


def gegenbauer_series(L, x):
    """P_l^(10)(1-2x) = 2F1(-l, l+7; 4; x) as the terminating series in L = l(l+7)."""
    term, total, k = ONE, ONE, 0
    scale = ONE
    while True:
        factor = (k * (k + 7) - L) * x / ((k + 4) * (k + 1))
        if factor == 0:
            return total
        term *= factor
        total += term
        k += 1
        scale = max(scale, abs(term))
        if abs(factor) < 1 and abs(term) < scale * g.exp2(-g.get_context().precision - 8):
            return total


def addfun(dst, src, factor=ONE):
    for key, v in src.items():
        dst[key] = dst.get(key, ZERO) + factor * v
    return {k: v for k, v in dst.items() if v != 0}


def wilson(m, n=0):
    return {(2 * m, n): M((-1)**m)}


def settings(tol="1e-30", max_iter=400):
    return dict(max_iter=max_iter, time_limit=1800, verbose=True,
                tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol,
                tol_infeas_abs=tol, tol_infeas_rel=tol,
                tol_ktratio=tol, tol_dual_qnorm=tol,
                reduced_tol_gap_abs="1e-25", reduced_tol_gap_rel="1e-25",
                reduced_tol_feas="1e-25", reduced_tol_ktratio="1e-25",
                reduced_tol_infeas_abs="1e-35", reduced_tol_infeas_rel="1e-25",
                static_regularization_constant="1e-65",
                static_regularization_proportional="0",
                dynamic_regularization_enable=False,
                iterative_refinement_abstol="1e-55",
                iterative_refinement_reltol="1e-55",
                iterative_refinement_max_iter=20,
                presolve_enable=True, equilibrate_enable=True)


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nmu = int(cfg["n_mu"])
        self.spins = list(range(0, int(cfg["j_max"]) + 1, 2))
        self.ns = len(self.spins)
        self.N = self.nmu * self.ns
        self.z = energy_nodes(cfg)
        self.mu = [1 / z for z in self.z]
        # Positive column rescaling K -> s(mu) K, rho -> rho / s(mu): the LP and its optimum are
        # unchanged, but with s = M^4/mu^2 the pole rows (~ 2 mu^2) are O(1) at every energy and
        # dual slacks are compared on a common scale.
        self.scaled = cfg.get("column_scale") == "mu2"
        self.colscale = [z * z for z in self.z] if self.scaled else [ONE] * self.nmu
        self.L = [ell * (ell + 7) for ell in self.spins]
        self.zpowers = {p: [z**p for z in self.z] for p in range(max(30,3*int(cfg["max_null_power"])-2)+1)}
        self.rows = []
        self.b = []
        self.names = []
        self.c00 = self.parameter("c00", 2 * g.zeta(3))
        self.c10 = self.parameter("c10", -2 * g.zeta(5))
        self.nt = int(cfg.get("n_t", 0)) if cfg.get("gravity", False) else 0
        t_grid = cfg.get("t_grid", "chebyshev")
        if t_grid == "uniform":
            # Equally spaced nodes t_i = -i/(N_t+1): the node nearest the forward limit moves in like
            # 1/N_t, against 1/N_t^2 for the Chebyshev nodes, so less energy is needed to build the pole.
            self.t = [M(-i) / (self.nt + 1) for i in range(1, self.nt + 1)]
        elif t_grid == "cos_closed":
            # t_i = (cos(i pi/(N_t+1)) - 1)/2: Chebyshev points of the second kind without the endpoints.
            self.t = [(g.cos(M(i) * g.const_pi() / (self.nt + 1)) - 1) / 2 for i in range(1, self.nt + 1)]
        elif t_grid == "cos_lobatto":
            # t_i = -1/2 + cos(i pi/N_t)/2, i = 1..N_t: Chebyshev-Lobatto points on [-1, 0] without t = 0.
            self.t = [(g.cos(M(i) * g.const_pi() / self.nt) - 1) / 2 for i in range(1, self.nt + 1)]
        elif t_grid == "half_cos":
            # t_i = -cos(i pi/(2(N_t+1))): clustered at t = -1, nearly uniform near the forward limit,
            # where the node nearest t = 0 sits at -sin(pi/(2(N_t+1))).
            self.t = [-g.cos(M(i) * g.const_pi() / (2 * (self.nt + 1))) for i in range(1, self.nt + 1)]
        elif t_grid == "chebyshev":
            self.t = [(-1 + g.cos(M(2 * i - 1) * g.const_pi() / (2 * self.nt))) / 2
                      for i in range(1, self.nt + 1)]
        else:
            raise ValueError("unknown t_grid " + str(t_grid))
        for power_ in range(1, int(cfg["max_null_power"]) + 1):
            for m in range(power_):
                self.row({(2 * m, power_): ONE}, ZERO, "null_%d_%d" % (2 * m, power_))
        for i, t in enumerate(self.t):
            self.row({("pole", i): ONE}, -1 / t, "pole_%d" % i)
        if cfg.get("fix_c00", True):
            self.row(wilson(0), self.c00, "fix_c00")
        if cfg.get("fix_c10", False):
            self.row(wilson(1), self.c10, "fix_c10")
        nl = int(cfg.get("nl", 0))
        if nl >= 1:
            self.row(wilson(1, 1), self.c00**2 / 2, "NL1")
        if nl >= 2:
            self.row(addfun(wilson(2, 1), wilson(1), -self.c00), ZERO, "NL2")
        if nl >= 3:
            self.row(addfun(wilson(2, 2), wilson(3), M(1) / 3), self.c00**3 / 6, "NL3")
        if nl >= 4:
            if not cfg.get("fix_c10", False):
                raise ValueError("NL4-NL6 require fixing c10")
            self.row(addfun(wilson(3, 1), wilson(2), -self.c00), self.c10**2 / 2, "NL4")
        if nl >= 5:
            self.row(addfun(wilson(3, 2), wilson(4)), self.c00**2 * self.c10 / 2, "NL5")
        if nl >= 6:
            self.row(addfun(wilson(3, 3), wilson(3), self.c00 / 3), self.c00**4 / 24, "NL6")
        self.cost = {}
        for key, coeff in cfg.get("objective", {"c20": "1"}).items():
            pair = {"c00": (0, 0), "c10": (1, 0), "c20": (2, 0)}[key]
            addfun(self.cost, wilson(*pair), M(str(coeff)))
        self.polys = {}
        # Forward coefficients are always reported by the external audit,
        # including when a coefficient is absent from the LP rows/objective.
        for fn in self.rows + [self.cost, wilson(0), wilson(1), wilson(2)]:
            for spec in fn:
                if spec[0] != "pole" and spec not in self.polys:
                    self.polys[spec] = [mp_rat(c) for c in qpoly(*spec)]
        self.qvalues = {spec: [horner(poly, M(L)) for L in self.L]
                        for spec, poly in self.polys.items()}
        self.pole_values = []
        cache_key = (g.get_context().precision, tuple(self.z), int(cfg["j_max"]), tuple(self.t))
        if self.nt and cache_key in POLE_CACHE:
            self.pole_values = POLE_CACHE[cache_key]
            POLE_CACHE.move_to_end(cache_key)
        elif self.nt:
            tic = time.time()
            aa = [M(2*n+7)/(n+7) for n in range(int(cfg["j_max"]))]
            bb = [M(n)/(n+7) for n in range(int(cfg["j_max"]))]
            for t in self.t:
                values = []
                for mu in self.mu:
                    x = 1 + 2 * t / mu
                    pm, pc = ONE, x
                    series = [pm]
                    for n in range(1, int(cfg["j_max"])):
                        pn = aa[n] * x * pc - bb[n] * pm
                        if (n + 1) % 2 == 0:
                            series.append(pn)
                        pm, pc = pc, pn
                    factor = mu * (2 * mu + t)
                    values.extend(v * factor for v in series)
                assert len(values) == self.N
                self.pole_values.append(values)
            print("gravity_kernel_seconds", time.time() - tic, flush=True)
            POLE_CACHE[cache_key] = self.pole_values
            while len(POLE_CACHE) > 2:
                POLE_CACHE.popitem(last=False)
        self.Nmain = self.N
        self.ext_mu, self.ext_spin, self.ext_L, self.ext_nb, self.ext_scale = [], [], [], 0, []
        if cfg.get("eikonal_mu_max"):
            self.eikonal_block(cfg)
        self.cached_cols = {}

    def eikonal_block(self, cfg):
        """Columns above the energy grid at fixed impact parameter b = 2 l / sqrt(mu).

        The k=-2 row receives its small-|t| support from this region. Energies are
        geometric above mu_max of the main grid; at each energy the spins are the even
        integers nearest b sqrt(mu)/2 on a uniform b grid. The Gegenbauer polynomial is
        the terminating series 2F1(-l, l+7; 4; -t/mu) in L = l(l+7).
        """
        top = max(self.mu)
        high = M(str(cfg["eikonal_mu_max"]))
        ratio = g.exp10(ONE / int(cfg.get("eikonal_per_decade", 8)))
        bstep = M(str(cfg.get("eikonal_b_step", "0.25")))
        nb = int(M(str(cfg.get("eikonal_b_max", 120))) / bstep)
        energies = []
        mu = top * ratio
        while mu <= high * (1 + M("1e-12")):
            energies.append(mu)
            mu *= ratio
        for mu in energies:
            for k in range(1, nb + 1):
                ell = 2 * int(g.rint(k * bstep * g.sqrt(mu) / 4))
                self.ext_mu.append(mu)
                self.ext_spin.append(ell)
                self.ext_L.append(M(ell) * (ell + 7))
        self.ext_nb = nb
        self.ext_scale = [1 / (mu * mu) if self.scaled else ONE for mu in self.ext_mu]
        self.ext_qvalues = {spec: [horner(poly, L) for L in self.ext_L] for spec, poly in self.polys.items()}
        tic = time.time()
        self.ext_pole = [[mu * (2 * mu + t) * gegenbauer_series(L, -t / mu)
                          for mu, L in zip(self.ext_mu, self.ext_L)] for t in self.t]
        self.N = self.Nmain + len(self.ext_mu)
        print("eikonal_block", len(energies), nb, "seconds", time.time() - tic, flush=True)

    def ext_neighbours(self, i):
        k = i - self.Nmain
        e, j = divmod(k, self.ext_nb)
        out = [self.Nmain + e * self.ext_nb + jj for jj in (j - 1, j + 1) if 0 <= jj < self.ext_nb]
        out += [self.Nmain + ee * self.ext_nb + j for ee in (e - 1, e + 1) if 0 <= ee < len(self.ext_mu) // self.ext_nb]
        return out

    def parameter(self, name, default):
        value = self.cfg.get(name)
        if value is not None:
            return M(str(value))
        alpha = M(str(self.cfg.get("alpha", "1")))
        if name == "c00":
            return 2 * g.zeta(3) * alpha**3
        if name == "c10":
            return -2 * g.zeta(5) * alpha**5
        return default

    def row(self, fn, rhs, name):
        self.rows.append(fn)
        self.b.append(M(rhs))
        self.names.append(name)

    def value(self, spec, p, j):
        if p >= self.nmu:
            k = (p - self.nmu) * self.ns + j
            if spec[0] == "pole":
                return self.ext_pole[spec[1]][k] * self.ext_scale[k]
            return self.ext_qvalues[spec][k] / self.ext_mu[k] ** sum(spec) * self.ext_scale[k]
        if spec[0] == "pole":
            return self.pole_values[spec[1]][p * self.ns + j] * self.colscale[p]
        return self.qvalues[spec][j] * self.zpowers[sum(spec)][p] * self.colscale[p]

    def scale_of(self, i):
        return self.ext_scale[i - self.Nmain] if i >= self.Nmain else self.colscale[i // self.ns]

    def fnval(self, fn, p, j):
        return sum((a * self.value(spec, p, j) for spec, a in fn.items()), ZERO)

    def column(self, i):
        if i not in self.cached_cols:
            p, j = divmod(i, self.ns)
            self.cached_cols[i] = [self.fnval(fn, p, j) for fn in self.rows]
        return self.cached_cols[i]

    def costval(self, i):
        p, j = divmod(i, self.ns)
        return self.fnval(self.cost, p, j)

    def initial(self):
        count = int(self.cfg.get("seed_energies", 24))
        pp = {round(i * (self.nmu - 1) / max(count - 1, 1)) for i in range(count)}
        # Resolve masses close to the first few string levels, including the gap.
        for n in range(1, 16):
            p = min(range(self.nmu), key=lambda p: abs(self.mu[p] - n))
            pp.update(range(max(0, p - 1), min(self.nmu, p + 2)))
        jj = set(range(min(self.ns, 9)))
        jj.update(round(i * (self.ns - 1) / 8) for i in range(9))
        jj.update(min(self.ns - 1, ell // 2) for ell in [20, 24, 32, 48, 64, 96, 128, 200, 300, 500])
        if self.cfg.get("full_master", False):
            return list(range(self.N))
        return sorted({p * self.ns + j for p in pp for j in jj})

    def problem(self, indices, primal=False):
        cols = [self.column(i) for i in indices]
        fc = [self.costval(i) for i in indices]
        scales = [max([ONE, abs(f)] + [abs(v) for v in col]) for col, f in zip(cols, fc)]
        r, n = len(self.rows), len(indices)
        if primal:
            colptr, rowval, nzval = [0], [], []
            for j, (col, scale) in enumerate(zip(cols, scales)):
                for k, v in enumerate(col):
                    if v:
                        rowval.append(k); nzval.append(dec(v / scale))
                rowval.append(r + j); nzval.append("-1")
                colptr.append(len(rowval))
            A = dict(m=r + n, n=n, colptr=colptr, rowval=rowval, nzval=nzval)
            b = [dec(v) for v in self.b] + ["0"] * n
            q = [dec(f / h) for f, h in zip(fc, scales)]
            cones = [{"ZeroConeT": r}, {"NonnegativeConeT": n}]
            dim = n
        else:
            colptr, rowval, nzval = [0], [], []
            for k in range(r):
                for j, (col, scale) in enumerate(zip(cols, scales)):
                    if col[k]:
                        rowval.append(j); nzval.append(dec(col[k] / scale))
                colptr.append(len(rowval))
            A = dict(m=n, n=r, colptr=colptr, rowval=rowval, nzval=nzval)
            b = [dec(f / h) for f, h in zip(fc, scales)]
            q = [dec(-v) for v in self.b]
            cones = [{"NonnegativeConeT": n}]
            dim = r
        P = dict(m=dim, n=dim, colptr=[0] * (dim + 1), rowval=[], nzval=[])
        return dict(P=P, q=q, A=A, b=b, cones=cones, settings=settings(
            self.cfg.get("solver_tol", "1e-30"), int(self.cfg.get("max_iter", 400)))), scales

    def separation(self, y, ray=False, count=64):
        combined = {} if ray else dict(self.cost)
        for yr, row in zip(y, self.rows):
            combined = addfun(combined, row, -yr)
        features = {}
        poles = {}
        for spec, weight in combined.items():
            if spec[0] == "pole":
                poles[spec[1]] = weight
            else:
                for degree, coeff in enumerate(self.polys[spec]):
                    key = (sum(spec), degree)
                    features[key] = features.get(key, ZERO) + weight * coeff
        degree = max((key[1] for key in features), default=0)
        worst = []
        diverse = set()
        sampled_spin_min = {}
        spin_step = max(1,self.ns//64)
        minimum, minimum_i = None, None
        for p in range(self.nmu):
            energy_minimum,energy_index=None,None
            poly = [ZERO] * (degree + 1)
            for (power_, d), coeff in features.items():
                poly[d] += coeff * self.zpowers[power_][p]
            for j, L in enumerate(self.L):
                idx = p * self.ns + j
                v = horner(poly, M(L))
                for ti, coeff in poles.items():
                    v += coeff * self.pole_values[ti][idx]
                v *= self.colscale[p]
                if energy_minimum is None or v<energy_minimum:
                    energy_minimum,energy_index=v,idx
                if j%spin_step==0 or j==self.ns-1:
                    if j not in sampled_spin_min or v<sampled_spin_min[j][0]:sampled_spin_min[j]=(v,idx)
                if minimum is None or v < minimum:
                    minimum, minimum_i = v, idx
                if v < 0:
                    candidate = (-v, idx)
                    if len(worst) < count:
                        heapq.heappush(worst, candidate)
                    elif candidate > worst[0]:
                        heapq.heapreplace(worst, candidate)
            if energy_minimum<0:diverse.add(energy_index)
        for k, (mu, L) in enumerate(zip(self.ext_mu, self.ext_L)):
            idx = self.Nmain + k
            if k % self.ext_nb == 0:
                energy_minimum, energy_index = None, None
                poly = [ZERO] * (degree + 1)
                for (power_, d), coeff in features.items():
                    poly[d] += coeff / mu ** power_
            v = horner(poly, L)
            for ti, coeff in poles.items():
                v += coeff * self.ext_pole[ti][k]
            v *= self.ext_scale[k]
            if energy_minimum is None or v < energy_minimum:
                energy_minimum, energy_index = v, idx
            if minimum is None or v < minimum:
                minimum, minimum_i = v, idx
            if v < 0:
                candidate = (-v, idx)
                if len(worst) < count:
                    heapq.heappush(worst, candidate)
                elif candidate > worst[0]:
                    heapq.heapreplace(worst, candidate)
            if k % self.ext_nb == self.ext_nb - 1 and energy_minimum < 0:
                diverse.add(energy_index)
        diverse.update(i for v,i in sampled_spin_min.values() if v<0)
        return minimum, minimum_i, sorted(diverse|{i for _,i in worst})

    def audit(self, result, indices, scales):
        y = [M(x) for x in result["x"]]
        rho = [M(z) / h for z, h in zip(result["z"], scales)]
        lhs = [sum((self.column(i)[k] * w for i, w in zip(indices, rho)), ZERO)
               for k in range(len(self.rows))]
        residual = max((abs(a - b) for a, b in zip(lhs, self.b)), default=ZERO)
        normalized = max((abs(a - b) / max(ONE, abs(b)) for a, b in zip(lhs, self.b)), default=ZERO)
        pobj = sum((self.costval(i) * w for i, w in zip(indices, rho)), ZERO)
        dobj = sum((b * yy for b, yy in zip(self.b, y)), ZERO)
        values = {"c%d%d" % (m, n): sum((self.fnval(wilson(m, n), *divmod(i, self.ns)) * w
                  for i, w in zip(indices, rho)), ZERO)
                  for m, n in [(0, 0), (1, 0), (2, 0)]}
        records = []
        for i, weight in zip(indices, rho):
            if weight > M("1e-35"):
                p, j = divmod(i, self.ns)
                if i >= self.Nmain:
                    k = i - self.Nmain
                    records.append(dict(index=i, mu_index=-1, spin=self.ext_spin[k], mu=dec(self.ext_mu[k]),
                                        rho=dec(weight * self.ext_scale[k])))
                    continue
                # Physical weight of the state (the LP variable is rho / s(mu)).
                records.append(dict(index=i, mu_index=p, spin=self.spins[j], mu=dec(self.mu[p]),
                                    rho=dec(weight * self.colscale[p])))
        return y, dict(primal_objective=dec(pobj), dual_objective=dec(dobj),
                       gap=dec(abs(pobj - dobj)), equality_residual=dec(residual),
                       normalized_equality_residual=dec(normalized),
                       nonnegativity_violation=dec(max(ZERO, -min(rho, default=ZERO))),
                       wilson={k: dec(v) for k, v in values.items()}, spectrum=records)


def call_solver(binary, problem, path, threads, primal=False, bits=256):
    write_json(path / "input.json", problem)
    args = [str(binary), str(path / "input.json"), "--precision", str(bits),
            "--threads", str(threads), "--output", str(path / "point.json")]
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               SDPX_RECEIPT=str(path / "receipt.json"))
    tic = time.time()
    with (path / "stdout.log").open("w") as stdout, (path / "stderr.log").open("w") as stderr:
        run = subprocess.run(args, env=env, stdout=stdout, stderr=stderr, timeout=1950)
    point = json.loads((path / "point.json").read_text()) if (path / "point.json").exists() else {}
    write_json(path / "run.json", dict(command=args, exit_code=run.returncode,
                                      elapsed_seconds=time.time() - tic,
                                      input_sha256=sha(path / "input.json"),
                                      solver_sha256=sha(binary)))
    return point


def run_case(cfg, binary, output, threads=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    write_json(output / "config.json", cfg)
    tic = time.time()
    model = Model(cfg)
    indices = model.initial()
    resume_path = Path(cfg['_resume_round_directory']) if cfg.get('_resume_round_directory') else None
    if resume_path:
        previous_mapping=json.loads((resume_path/'mapping.json').read_text())
        indices=previous_mapping['indices']
    outcome = dict(case=cfg.get("name", output.name), config=cfg, accepted=False,
                   classification="unresolved", precompute_bits=512, precision_bits=int(model.cfg.get("solver_bits", 256)),
                   solver_sha256=sha(binary), n_spectral_columns=model.N,
                   n_equalities=len(model.rows), rounds=[])
    acceptance = M(cfg.get("audit_tol", "1e-25"))
    for iteration in range(int(cfg.get("max_exchange", 32))):
        path = output / ("round_%02d" % iteration)
        path.mkdir(exist_ok=True)
        problem, scales = model.problem(indices)
        write_json(path / "mapping.json", dict(indices=indices, scales=[dec(h) for h in scales], row_names=model.names))
        if iteration == 0 and resume_path and not cfg.get('_resolve_saved_master',False):
            import shutil
            for name in ['point.json','receipt.json','run.json','stdout.log','stderr.log','input.json']:
                if (resume_path/name).exists(): shutil.copy2(resume_path/name,path/name)
            point=json.loads((path/'point.json').read_text())
            write_json(path/'adopted_solver_output.json',dict(source_directory=str(resume_path),
                       point_sha256=sha(path/'point.json'),reason='continue after coefficient-reporting exception or bounded exchange limit'))
        else:
            # Block columns at mu ~ 1e16 carry pole entries ~ mu^2: the audit's absolute slack
            # tolerance then needs a master solved beyond 256 bits.
            point = call_solver(binary, problem, path, threads, bits=int(model.cfg.get("solver_bits", 256)))
        status = point.get("status", "NoResult")
        entry = dict(round=iteration, master_columns=len(indices), status=status,
                     native_seconds=point.get("native_seconds"), iterations=point.get("iterations"))
        outcome["rounds"].append(entry)
        if status == "Solved":
            y, audit = model.audit(point, indices, scales)
            minimum, worst_i, worst = model.separation(y)
            entry.update(minimum_full_grid_slack=dec(minimum), worst_index=worst_i)
            write_json(path / "audit.json", audit)
            good = (M(audit["normalized_equality_residual"]) <= acceptance and
                    M(audit["nonnegativity_violation"]) <= acceptance and
                    M(audit["gap"]) <= acceptance * max(ONE, abs(M(audit["primal_objective"]))))
            if good and not any(model.cost.values()):
                # For a feasibility LP the exact zero dual is feasible on
                # every column. The independently audited positive primal
                # suffices; numerical noise in the native dual cannot change
                # this zero-objective certificate.
                minimum, worst_i, worst = ZERO, 0, []
                entry['minimum_full_grid_slack']='0'
                outcome['zero_objective_certificate_y']=['0']*len(model.rows)
            if minimum >= -acceptance and good:
                outcome.update(accepted=True, classification="feasible", status=status,
                               audit=audit, minimum_full_grid_slack=dec(minimum),
                               final_round=iteration, master_columns=len(indices))
                if "fix_c00" in model.names and minimum < 0:
                    repaired = list(y)
                    repaired[model.names.index("fix_c00")] += minimum / 2
                    fixed_min, _, _ = model.separation(repaired)
                    outcome["repaired_dual_objective"] = dec(sum((b * yy for b, yy in zip(model.b, repaired)), ZERO))
                    outcome["repaired_minimum_full_grid_slack"] = dec(fixed_min)
                break
            if not good and minimum >= -acceptance:
                outcome.update(status=status, failure="external_audit_failed", audit=audit)
                break
        elif status == "DualInfeasible":
            # The solved program is the physical LP's dual. Its recession ray
            # is an infeasibility certificate for the physical primal only
            # after separation checks all declared spectral columns.
            y = [M(x) for x in point.get("x", [])]
            normal = sum((b * yy for b, yy in zip(model.b, y)), ZERO)
            if len(y) != len(model.rows) or normal <= 0:
                outcome.update(status=status, failure="invalid_infeasibility_ray")
                break
            y = [yy / normal for yy in y]
            minimum, worst_i, worst = model.separation(y, ray=True)
            entry.update(minimum_full_grid_ray_slack=dec(minimum), worst_index=worst_i)
            if minimum >= -acceptance:
                outcome.update(accepted=True, classification="infeasible", status=status,
                               minimum_full_grid_ray_slack=dec(minimum), final_round=iteration,
                               certificate_y=[dec(v) for v in y], certificate_b_dot_y="1")
                break
        elif status in ["AlmostDualInfeasible", "AlmostSolved", "MaxIterations",
                        "InsufficientProgress", "NumericalError"] and len(point.get('x',[])) == len(model.rows):
            # Incomplete iterates can guide column generation, but cannot
            # certify either feasibility or exclusion. Keep their statuses.
            y=[M(v) for v in point['x']]
            normal=sum((b*v for b,v in zip(model.b,y)),ZERO)
            ray=(status=='AlmostDualInfeasible' and normal>0)
            if ray: y=[v/normal for v in y]
            minimum,worst_i,worst=model.separation(y,ray=ray)
            entry.update(heuristic_only=True,minimum_heuristic_slack=dec(minimum),worst_index=worst_i)
            if minimum>=-acceptance:
                outcome.update(status=status,failure="solver_incomplete_or_wrong_direction")
                break
        else:
            outcome.update(status=status, failure="solver_incomplete_or_wrong_direction")
            break
        new = set(worst) - set(indices)
        if iteration == 0 and resume_path:
            new.update(set(model.initial())-set(indices))
        # Adjacent spins and energy nodes help resolve a moving support point.
        for i in list(new):
            if i >= model.Nmain:
                new.update(model.ext_neighbours(i))
                continue
            p, j = divmod(i, model.ns)
            for pp, jj in [(p - 1, j), (p + 1, j), (p, j - 1), (p, j + 1)]:
                if 0 <= pp < model.nmu and 0 <= jj < model.ns:
                    new.add(pp * model.ns + jj)
        if not new:
            outcome.update(status=status, failure="exchange_stalled")
            break
        indices = sorted(set(indices) | new)
        write_json(output / "progress.json", outcome)
        print(json.dumps(dict(case=outcome["case"], round=iteration, status=status,
                              columns=len(indices), minimum_slack=dec(minimum))), flush=True)
    else:
        outcome["failure"] = "exchange_limit"
    outcome["elapsed_seconds"] = time.time() - tic
    outcome["max_rss_kb"] = __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss
    write_json(summary_path, outcome)
    print(json.dumps(dict(case=outcome["case"], classification=outcome["classification"],
                          accepted=outcome["accepted"], seconds=outcome["elapsed_seconds"])), flush=True)
    return outcome


def algebra_gate(output):
    assert qpoly(0, 1) == (F(-1), F(1, 2))
    assert qpoly(0, 2) == (F(-1), F(-13, 20), F(1, 20))
    assert qpoly(2, 0) == (F(2),)
    # Unit logarithmic atoms reproduce the correctly normalized forward sums.
    tail = sum((M(1) / M(n)**3 for n in range(1, 2001)), ZERO)
    assert abs(2 * tail - 2 * g.zeta(3)) < M("3e-7")
    write_json(output, dict(passed=True, rational_polynomial_checks=3,
                            precompute_bits=512, forward_string_tail_check=True))


def scipy_gate(cfg, output):
    import numpy as np
    from scipy.optimize import linprog
    model = Model(cfg)
    indices = list(range(model.N))
    cols = [model.column(i) for i in indices]
    cost = [model.costval(i) for i in indices]
    scales = [max([ONE, abs(f)] + [abs(v) for v in col]) for col, f in zip(cols, cost)]
    A = np.array([[float(v / h) for v, h in zip(row, scales)] for row in zip(*cols)])
    c = np.array([float(v / h) for v, h in zip(cost, scales)])
    b = np.array([float(v) for v in model.b])
    result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    dual = linprog(-b, A_ub=A.T, b_ub=c, bounds=[(None, None)] * len(b), method="highs")
    write_json(output, dict(success=bool(result.success), status=int(result.status),
                            objective=float(result.fun) if result.success else None,
                            dual_success=bool(dual.success),
                            dual_objective=float(-dual.fun) if dual.success else None,
                            primal_dual_gap=float(abs(result.fun + dual.fun)) if result.success and dual.success else None,
                            message=result.message, comparison_precision="binary64",
                            original_equality_residual=float(np.max(np.abs(A @ result.x - b))) if result.success else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--binary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gate", choices=["algebra", "scipy"])
    args = parser.parse_args()
    if args.gate == "algebra":
        algebra_gate(args.output)
    elif args.gate == "scipy":
        scipy_gate(json.loads(Path(args.config).read_text()), args.output)
    else:
        run_case(json.loads(Path(args.config).read_text()), args.binary, args.output, args.threads)


if __name__ == "__main__":
    main()
