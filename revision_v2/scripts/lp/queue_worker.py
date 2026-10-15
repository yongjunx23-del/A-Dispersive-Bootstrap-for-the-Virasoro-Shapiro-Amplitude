#!/usr/bin/env python3
"""Shared-filesystem queue worker: SDPX solve (512-bit inputs), 768-bit full-grid audit.

Tasks are files queue/tasks/<priority>__<signature>__<name>.json written by
planner.py.  A worker claims a task by creating queue/claims/<name> (atomic
mkdir), solves it with run_case, re-audits it with verify_case at 768 bits,
and marks it done.  Workers prefer tasks with the grid signature of their previous
task, whose active columns seed the next restricted master (WarmedModel).
"""
import argparse
import contextlib
import gc
import json
import re
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import socket
import threading
import time
import traceback

import bootstrap_sdpx_v5 as core
from verify_results_v9 import verify_case
from plan import signature, memory_gb

SOLVE_BITS, AUDIT_BITS = 512, 768
HOST = socket.gethostname()
WARM = {}
BaseModel = core.Model


def warm_key(cfg):
    return (cfg['energy_grid'], cfg['n_mu'], cfg['j_max'], bool(cfg['gravity']), int(cfg.get('nl', 0)))


class WarmedModel(BaseModel):
    """Seed the restricted master with the active columns of the previous solve on this grid."""
    def initial(self):
        out = set(super().initial())
        out.update(i for i in WARM.get(warm_key(self.cfg), ()) if 0 <= i < self.N)
        return sorted(out)


def absorb(cfg, outcome, folder):
    if outcome.get('classification') != 'feasible' or 'final_round' not in outcome:
        return
    p = folder / ('round_%02d' % int(outcome['final_round']))
    try:
        mapping = json.loads((p / 'mapping.json').read_text())
        point = json.loads((p / 'point.json').read_text())
    except (OSError, ValueError):
        return
    z = point.get('z', [])
    if len(z) != len(mapping['indices']):
        return
    order = sorted(range(len(z)), key=lambda j: core.M(z[j]), reverse=True)
    keep = max(100, 4 * len(mapping['row_names']))
    WARM[warm_key(cfg)] = {mapping['indices'][j] for j in order[:keep]}


def host_memory(queue):
    total = 0.0
    for name in os.listdir(queue / 'claims'):
        folder = queue / 'claims' / name
        if (folder / 'done').exists():
            continue
        try:
            info = json.loads((folder / 'owner.json').read_text())
        except (OSError, ValueError):
            continue
        if info.get('host') == HOST:
            total += float(info.get('memory_gb', 0))
    return total


def commit_headroom_gb():
    """Strict overcommit (vm.overcommit_memory=2) on these nodes: allocations fail beyond CommitLimit."""
    info = {}
    with open('/proc/meminfo') as fh:
        for line in fh:
            key, value = line.split(':', 1)
            info[key] = int(value.split()[0])
    return (info['CommitLimit'] - info['Committed_AS']) / 1048576


ONLY = ''
TASKS = 'tasks'


def claim(queue, current_sig, budget, wid):
    claimed = set(os.listdir(queue / 'claims'))
    entries = []
    for f in os.listdir(queue / TASKS):
        if not f.endswith('.json'):
            continue
        prio, sig, name = f[:-5].split('__', 2)
        if ONLY and not re.search(ONLY, name):
            continue
        if name not in claimed:
            entries.append((sig != current_sig, int(prio), name, sig, f))
    if not entries:
        return None
    entries.sort()
    used = host_memory(queue)
    for _, _, name, sig, f in entries:
        cfg = json.loads((queue / TASKS / f).read_text())
        need = memory_gb(cfg)
        if used > 0 and used + need > budget:
            continue
        # Fork and audit roughly double a task's commit charge; our running tasks may still be growing.
        if commit_headroom_gb() < 2 * need + 12:
            continue
        attempts = queue.parent / 'results' / name / 'attempts.json'
        if attempts.exists() and any(a.get('host') == HOST for a in json.loads(attempts.read_text())):
            continue
        try:
            os.mkdir(queue / 'claims' / name)
        except FileExistsError:
            continue
        (queue / 'claims' / name / 'owner.json').write_text(json.dumps(dict(
            host=HOST, pid=os.getpid(), worker=wid, memory_gb=need, start=time.time(),
            pbs_job_id=os.environ.get('PBS_JOBID'))))
        return cfg
    return None


def heartbeat(path, stop):
    while not stop.wait(60):
        try:
            path.touch()
        except OSError:
            pass


def run_task(cfg, root, binary, threads, warm):
    """Child process: one solve and one audit, so all kernel memory is returned on exit."""
    WARM.clear()
    WARM.update(warm)
    target = root / 'results' / cfg['name']
    target.mkdir(parents=True, exist_ok=True)
    try:
        core.Model = WarmedModel
        core.g.get_context().precision = SOLVE_BITS
        with (target / 'worker.log').open('a', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                outcome = core.run_case(cfg, binary, target, threads=threads)
                core.write_json(target / 'warm.json', dict(final_round=outcome.get('final_round'),
                                                           classification=outcome.get('classification')))
                # Hold one precision of gravity kernels at a time (memory_gb assumes this).
                core.POLE_CACHE.clear()
                gc.collect()
                core.Model = BaseModel
                core.g.get_context().precision = AUDIT_BITS
                d = verify_case(cfg, root)
                if d is not None:
                    d['audit_bits'] = AUDIT_BITS
                    # verify_case labels are fixed strings; the arithmetic ran at the context precision.
                    d['verification'] = d.get('verification', '').replace('384-bit', '%d-bit' % AUDIT_BITS)
                    core.write_json(target / 'verified_summary_v5.json', d)
    except Exception as exc:
        core.write_json(target / 'exception.json',
                        dict(type=type(exc).__name__, message=str(exc), trace=traceback.format_exc()))
        os._exit(3)
    os._exit(0)


def worker(wid, root, binary, threads, budget, cpus):
    if cpus:
        os.sched_setaffinity(0, set(cpus))
    queue = root / 'queue'
    current = None
    # Stagger the first claims: a claim counts only the memory of tasks already claimed on this host.
    time.sleep(5 * wid)
    while True:
        cfg = claim(queue, current, budget, wid)
        if cfg is None:
            if (queue / 'planner_done').exists():
                return
            time.sleep(30)
            continue
        current = signature(cfg)
        folder = queue / 'claims' / cfg['name']
        stop = threading.Event()
        beat = threading.Thread(target=heartbeat, args=(folder / 'heartbeat', stop), daemon=True)
        (folder / 'heartbeat').touch()
        beat.start()
        tic = time.time()
        child = mp.Process(target=run_task, args=(cfg, root, binary, threads, dict(WARM)))
        child.start()
        child.join()
        stop.set()
        target = root / 'results' / cfg['name']
        try:
            info = json.loads((target / 'warm.json').read_text())
            absorb(cfg, info, target)
        except (OSError, ValueError):
            pass
        status = 'ok' if child.exitcode == 0 else 'exit_%s' % child.exitcode
        attempts_path = target / 'attempts.json'
        attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        if status != 'ok':
            exc = target / 'exception.json'
            reason = json.loads(exc.read_text()).get('type') if exc.exists() else None
            attempts.append(dict(status=status, reason=reason, host=HOST, seconds=time.time() - tic))
            attempts_path.write_text(json.dumps(attempts, indent=1))
            # Preserve a failed task for diagnosis; the planner records it as unresolved.
            # A scoped retry is prepared only after its failure has been inspected.
        (folder / 'done').write_text(json.dumps(dict(status=status, seconds=time.time() - tic, host=HOST,
                                                     attempts=len(attempts) + (status == 'ok'))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--binary', required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--memory-budget', type=float, default=170)
    p.add_argument('--allocated-cores', type=int, default=50)
    p.add_argument('--tasks-subdir', default='tasks')
    p.add_argument('--only', default='', help='claim only tasks whose name matches this regular expression')
    a = p.parse_args()
    global TASKS, ONLY
    TASKS, ONLY = a.tasks_subdir, a.only
    root = Path(a.root)
    mp.set_start_method('fork')
    allowed = sorted(os.sched_getaffinity(0))
    # Pin only when PBS has confined the job to its own cpuset.
    pin = len(allowed) <= a.allocated_cores
    children = []
    for w in range(a.workers):
        cpus = allowed[w * a.threads:(w + 1) * a.threads] if pin else None
        child = mp.Process(target=worker, args=(w, root, a.binary, a.threads, a.memory_budget, cpus))
        child.start()
        children.append(child)
    core.write_json(root / 'logs' / ('workers_%s.json' % HOST), dict(
        host=HOST, pbs_job_id=os.environ.get('PBS_JOBID'), workers=a.workers, threads=a.threads,
        affinity=allowed, pinned=pin, memory_budget_gb=a.memory_budget, start=time.time()))
    for child in children:
        child.join()


if __name__ == '__main__':
    main()
