#!/usr/bin/env python3
"""Bounded node-local process queue; every case has separate immutable output."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from bootstrap_sdpx import run_case, write_json


def physical_cpus(count):
    allowed = os.sched_getaffinity(0)
    seen, selected = set(), []
    for cpu in sorted(allowed):
        p = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        identity = ((p/'physical_package_id').read_text().strip(), (p/'core_id').read_text().strip())
        if identity not in seen:
            seen.add(identity); selected.append(cpu)
        if len(selected) == count:
            return selected
    raise RuntimeError('Not enough physical cores visible')


def worker(worker_id, tasks, binary, root, cpus, threads):
    os.sched_setaffinity(0, set(cpus))
    for cfg in tasks:
        target = root / 'results' / cfg['name']
        target.mkdir(parents=True, exist_ok=True)
        try:
            with (target/'worker.log').open('a', buffering=1) as log:
                import contextlib
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    run_case(cfg, binary, target, threads=threads)
        except Exception as exc:
            write_json(target/'exception.json', dict(type=type(exc).__name__, message=str(exc), trace=traceback.format_exc()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--binary', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    root = Path(args.root)
    tasks = json.loads(Path(args.manifest).read_text())['cases']
    cores = physical_cpus(args.workers * args.threads)
    write_json(root/'results/batch_resource_record.json', dict(workers=args.workers, threads_per_solver=args.threads,
                physical_cpus=cores, pbs_job_id=os.environ.get('PBS_JOBID'), number_of_cases=len(tasks)))
    tic = time.time()
    children = []
    for w in range(args.workers):
        child = mp.Process(target=worker, args=(w, tasks[w::args.workers], args.binary, root,
                          cores[w*args.threads:(w+1)*args.threads], args.threads))
        child.start(); children.append(child)
    for child in children:
        child.join()
    summaries = []
    for cfg in tasks:
        path = root/'results'/cfg['name']/'summary.json'
        if path.exists():
            d = json.loads(path.read_text())
            d.pop('audit', None)
            summaries.append(d)
        else:
            summaries.append(dict(case=cfg['name'], accepted=False, classification='unresolved', failure='missing_summary'))
    write_json(root/'results/batch_completion.json', dict(elapsed_seconds=time.time()-tic,
                accepted=sum(bool(x.get('accepted')) for x in summaries), number_of_cases=len(tasks),
                child_exit_codes=[child.exitcode for child in children], summaries=summaries))


if __name__ == '__main__':
    main()
