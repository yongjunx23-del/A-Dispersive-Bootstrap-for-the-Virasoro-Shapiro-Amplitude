"""Bounded diagnostic of the baseline exclusions at small alpha.

Keep the same pole samples and null rows, and append the established exact
Gegenbauer high-energy/high-spin columns. These are separate finite problems;
their records must never be merged into the baseline shaded region.
"""
import argparse
import hashlib
import json
from pathlib import Path
from plan import C0, C1, D, M, base, signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    root = Path(parser.parse_args().root)
    added = 0
    for alpha in ('0.1','0.2','0.3','0.35','0.4'):
        a = M(alpha)
        for reach in (6, 8, 12):
            for nl in (3, 6):
                config = base(True, kind='presentation_smallalpha_tail', nl=nl,
                              fix_c00=True, fix_c10=True, c00=D(C0*a**3), c10=D(C1*a**5),
                              reference_alpha=alpha, variant='tail_E%d'%reach, objective={},
                              column_scale='mu2', eikonal_mu_max='1e%d'%reach,
                              eikonal_per_decade=8, eikonal_b_max=200, eikonal_b_step='0.25')
                key = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
                config.update(name='ref_smalltail_'+key[:24], priority=0)
                path = root/'queue/tasks'/('0__%s__%s.json'%(signature(config),config['name']))
                if path.exists():
                    assert json.loads(path.read_text()) == config
                    continue
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(config,sort_keys=True,indent=1)+'\n')
                temporary.rename(path)
                added += 1
    print(json.dumps(dict(new_smallalpha_tail_tasks=added,maximum_tasks=30)))


if __name__ == '__main__':
    main()
