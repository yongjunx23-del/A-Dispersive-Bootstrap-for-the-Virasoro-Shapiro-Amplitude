"""Refine each already checked pole-subtracted NL6 slice interval fourfold."""
import argparse
import json
from pathlib import Path
from collect_refinement import collect
from plan import C0, D, M, base, mathematical_key, signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    root = Path(parser.parse_args().root)
    seed, fresh, aliases = collect(root)
    known = {mathematical_key(r['canonical_config']) for r in seed + fresh + aliases if r.get('accepted')}
    tasks = [json.loads(p.read_text()) for p in (root / 'queue/tasks').glob('*.json')]
    known.update(mathematical_key(c) for c in tasks)
    slices = {}
    # Refine the frozen predecessor intervals, so repeated calls cannot refine indefinitely.
    for row in seed:
        c = row['canonical_config']
        if (c['kind'] == 'pband' and c.get('variant') == 'base' and row.get('accepted')
                and row['classification'] == 'feasible'):
            slices.setdefault(M(c['scan_x']), set()).add('min' if M(c['objective']['c20']) > 0 else 'max')
    xs = sorted(x for x, signs in slices.items() if signs == {'min', 'max'})
    added = 0
    for a, b in zip(xs, xs[1:]):
        if b - a > M('0.00003'):
            continue
        for j in (1, 2, 3):
            x = a + (b - a) * j / 4
            for sign in ('1', '-1'):
                c = base(False, kind='pband', nl=6, fix_c00=True, fix_c10=True,
                         c00=D(C0), c10=D(C0 * x), scan_x=D(x), objective={'c20': sign})
                key = mathematical_key(c)
                if key in known:
                    continue
                c.update(name='ref_pband_' + key[:24], priority=2, presentation_band_refinement=4)
                path = root / 'queue/tasks' / ('2__%s__%s.json' % (signature(c), c['name']))
                assert not path.exists()
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(c, sort_keys=True, indent=1) + '\n')
                temporary.rename(path)
                known.add(key)
                added += 1
    assert added <= 240
    print(json.dumps(dict(new_pole_band_tasks=added)))


if __name__ == '__main__':
    main()
