"""Add NL6 endpoint tests and certified transition refinements to the live queue.

This brief planner does no numerical optimization. It preserves the frozen
model, all existing tasks, and failed outputs. Its task names hash the fixed
coefficients, so changing a frontier cannot overwrite a previous case.
"""
import argparse
import json
from pathlib import Path
from collect_refinement import collect
from plan import D, M, NL6_G, base, mathematical_key, signature

TARGET = M('0.005')
MAX_STEPS = 24
MAX_NEW_TASKS = 3000


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root)
    seed, fresh, aliases = collect(root)
    rows = seed + aliases + fresh
    completed = {mathematical_key(r['canonical_config']): r for r in rows
                 if r.get('accepted')}
    tasks = {mathematical_key(c): c for c in
             (json.loads(f.read_text()) for f in (root / 'queue/tasks').glob('*.json'))}
    targets = {D(gamma) for gamma in NL6_G}
    slices = {}
    for r in rows:
        c = r['canonical_config']
        if (c['kind'] in ('gnl3_slice', 'gnl3_end') and c.get('variant') == 'base'
                and r.get('accepted') and r['classification'] == 'feasible'):
            key = D(c['c00'])
            if key in targets:
                slices.setdefault(key, {})['min' if M(c['objective']['c10']) > 0 else 'max'] = r
    added = 0

    def add(gamma, x, level, reason):
        nonlocal added
        config = base(True, kind='gnl6_scan', nl=6, fix_c00=True, fix_c10=True,
                      c00=D(gamma), c10=D(x), objective={}, gamma=D(gamma))
        key = mathematical_key(config)
        if key in completed or key in tasks:
            return
        assert sum(c.get('supplement', False) for c in tasks.values()) < MAX_NEW_TASKS
        config.update(name='ref_edge_' + key[:24], priority=0, supplement=True,
                      supplement_step=level, supplement_reason=reason)
        path = root / 'queue/tasks' / ('0__%s__%s.json' % (signature(config), config['name']))
        assert not path.exists()
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(config, sort_keys=True, indent=1) + '\n')
        temporary.rename(path)
        tasks[key] = config
        added += 1

    for gamma, pair in sorted(slices.items(), key=lambda q: M(q[0])):
        if set(pair) != {'min', 'max'}:
            continue
        lo, hi = (M(pair[label]['audit']['wilson']['c10']) for label in ('min', 'max'))
        add(gamma, lo, 0, 'NL3 lower endpoint tested at NL6')
        add(gamma, hi, 0, 'NL3 upper endpoint tested at NL6')
        points = {}
        for key, c in tasks.items():
            if not (c.get('gravity') and c.get('nl') == 6 and c.get('fix_c10')
                    and D(c['c00']) == gamma and c.get('objective') == {}):
                continue
            # Enforce the baseline mathematical problem, including energy-grid identity.
            probe = base(True, nl=6, fix_c00=True, fix_c10=True, c00=gamma,
                         c10=c['c10'], objective={})
            if mathematical_key(probe) != key:
                continue
            r = completed.get(key)
            points[M(c['c10'])] = (r['classification'] if r else 'pending',
                                  c.get('supplement_step', 0))
        for key, r in completed.items():
            c = r['canonical_config']
            if not (c.get('gravity') and c.get('nl') == 6 and c.get('fix_c10')
                    and D(c['c00']) == gamma and c.get('objective') == {}):
                continue
            probe = base(True, nl=6, fix_c00=True, fix_c10=True, c00=gamma,
                         c10=c['c10'], objective={})
            if mathematical_key(probe) == key:
                x = M(c['c10'])
                level = max(points.get(x, ('', 0))[1], c.get('supplement_step', 0))
                points[x] = (r['classification'], level)
        feasible = [x for x, (cls, _) in points.items() if cls == 'feasible']
        if not feasible:
            continue
        width = max(feasible) - min(feasible)
        target = TARGET * width if width > 0 else M('1e-12') * max(abs(feasible[0]), M('1e-8'))
        ordered = sorted(points.items())
        for (x, (ca, la)), (y, (cb, lb)) in zip(ordered, ordered[1:]):
            level = max(la, lb) + 1
            if {ca, cb} == {'feasible', 'infeasible'} and y - x > target and level <= MAX_STEPS:
                add(gamma, (x + y) / 2, level, 'verified feasible/excluded transition')
    print(json.dumps(dict(new_endpoint_or_boundary_tasks=added,
                          total_supplement_tasks=sum(c.get('supplement', False) for c in tasks.values()))))


if __name__ == '__main__':
    main()
