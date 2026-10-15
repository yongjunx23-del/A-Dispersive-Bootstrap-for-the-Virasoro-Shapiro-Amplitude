#!/usr/bin/env python3
"""Time SDPB and SDPX on identical discrete LPs taken from this campaign.

Each case is an SDPX input.json (Clarabel form: minimize q.x subject to
A x + s = b, s >= 0).  The same LP is written as an SDPB polynomial matrix
program with one 1x1, degree-0 positive block per spectral column,
    y0 * b_i - sum_r A_ir x_r >= 0,   normalization y0 = 1,   objective -q.x,
so both solvers receive the same scaled decimal data.  Both run at 256-bit
working precision with gap/feasibility thresholds 1e-30 and the same number
of cores (SDPX threads = SDPB MPI ranks).
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import gmpy2 as g

g.get_context().precision = 512
M = g.mpfr


def to_pmp(problem):
    A, b, q = problem['A'], [M(x) for x in problem['b']], [M(x) for x in problem['q']]
    n_cols, r = A['m'], A['n']
    rows = [[M(0)] * r for _ in range(n_cols)]
    for k in range(r):
        for idx in range(A['colptr'][k], A['colptr'][k + 1]):
            rows[A['rowval'][idx]][k] = M(A['nzval'][idx])
    f = lambda x: format(x, '.150g')
    blocks = [{'polynomials': [[[[f(b[i])]] + [[f(-v)] for v in rows[i]]]]} for i in range(n_cols)]
    return dict(objective=['0'] + [f(-x) for x in q], normalization=['1'] + ['0'] * r,
                PositiveMatrixWithPrefactorArray=blocks)


def run(cmd, log, env=None):
    tic = time.time()
    with open(log, 'w') as out:
        code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
    return code, time.time() - tic


def sdpb_case(name, problem, work, procs, sdpb_env):
    d = work / name / 'sdpb'
    assert not d.exists(), "preserve existing timing output"
    d.mkdir(parents=True)
    (d / 'pmp.json').write_text(json.dumps(to_pmp(problem)))
    mpirun = ['mpirun', '--bind-to', 'none', '-n', str(procs)]
    code1, t_convert = run(mpirun + ['pmp2sdp', '--precision', '256', '-i', str(d / 'pmp.json'), '-o', str(d / 'sdp')],
                           d / 'pmp2sdp.log', sdpb_env)
    params = ['--precision=256', '--dualityGapThreshold=1e-30', '--primalErrorThreshold=1e-30',
              '--dualErrorThreshold=1e-30', '--maxIterations=500', '--maxRuntime=7200',
              '--noFinalCheckpoint', '--writeSolution=y']
    code2, t_solve = run(mpirun + ['sdpb', '-s', str(d / 'sdp'), '-o', str(d / 'out')] + params,
                         d / 'sdpb.log', sdpb_env)
    text = (d / 'sdpb.log').read_text(errors='replace')
    out = (d / 'out' / 'out.txt').read_text() if (d / 'out' / 'out.txt').exists() else ''
    grab = lambda key: (re.search(key + r'\s*=\s*"?([^";\n]+)', out) or [None, None])[1]
    iterations = len(re.findall(r'^\s*\d+\s+[\d:]+\s', text, re.M))
    return dict(convert_exit=code1, convert_seconds=t_convert, solve_exit=code2, solve_seconds=t_solve,
                terminate_reason=grab('terminateReason'), primal_objective=grab('primalObjective'),
                dual_objective=grab('dualObjective'), duality_gap=grab('dualityGap'),
                solver_seconds=grab('Solver runtime'), iterations=iterations, blocks=problem['A']['m'])


def sdpx_case(name, problem, work, threads, binary):
    d = work / name / 'sdpx'
    assert not d.exists(), "preserve existing timing output"
    d.mkdir(parents=True)
    problem = json.loads(json.dumps(problem))
    problem['settings'].update(max_iter=500, time_limit=7200)
    (d / 'input.json').write_text(json.dumps(problem))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', SDPX_RECEIPT=str(d / 'receipt.json'))
    code, seconds = run([binary, str(d / 'input.json'), '--precision', '256', '--threads', str(threads),
                         '--output', str(d / 'point.json')], d / 'sdpx.log', env)
    point = json.loads((d / 'point.json').read_text()) if (d / 'point.json').exists() else {}
    q = [M(x) for x in problem['q']]
    obj = sum((a * M(x) for a, x in zip(q, point.get('x', []))), M(0)) if point.get('x') else None
    return dict(exit=code, wall_seconds=seconds, status=point.get('status'), iterations=point.get('iterations'),
                native_seconds=point.get('native_seconds'),
                objective_minus_q_dot_x=format(-obj, '.40g') if obj is not None else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', required=True, help='JSON list of {name, input}')
    p.add_argument('--work', required=True)
    p.add_argument('--cores', type=int, default=8)
    p.add_argument('--sdpx', required=True)
    p.add_argument('--sdpb-env', required=True)
    a = p.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    # The PBS script sources the SDPB runtime; never dump the environment.
    sdpb_env = dict(os.environ)
    sdpb_env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    results = []
    out_path = work / 'sdpb_vs_sdpx.json'
    for case in json.loads(Path(a.cases).read_text()):
        problem = json.loads(Path(case['input']).read_text())
        entry = dict(name=case['name'], source=case['input'], columns=problem['A']['m'], rows=problem['A']['n'],
                     cores=a.cores, solver_bits=256, kernel_bits=512, tolerance='1e-30')
        entry['sdpx'] = sdpx_case(case['name'], problem, work, a.cores, a.sdpx)
        entry['sdpb'] = sdpb_case(case['name'], problem, work, a.cores, sdpb_env)
        results.append(entry)
        out_path.write_text(json.dumps(results, indent=1) + '\n')
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
