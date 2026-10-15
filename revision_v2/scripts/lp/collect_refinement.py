"""Merge completed audits, exact-problem aliases, and the frozen predecessor data."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def collect(root):
    seed = json.loads((root / 'data/seed_records.json').read_text())
    source = {r['case']: r for r in seed}
    fresh = []
    for task in sorted((root / 'queue/tasks').glob('*.json')):
        config = json.loads(task.read_text())
        name = config['name']
        done = root / 'queue/claims' / name / 'done'
        folder = root / 'results' / name
        if not done.exists():
            continue
        row = dict(case=name, canonical_config=config, accepted=False,
                   classification='unresolved', provenance='new solve')
        row['worker_seconds'] = json.loads(done.read_text()).get('seconds')
        audit_file = folder / 'verified_summary_v5.json'
        if audit_file.exists():
            audit = json.loads(audit_file.read_text())
            for key in ['accepted', 'classification', 'failure', 'audit_bits', 'verification',
                        'final_round', 'master_columns', 'n_equalities', 'n_spectral_columns',
                        'elapsed_seconds', 'minimum_full_grid_slack',
                        'minimum_full_grid_ray_slack', 'verified_gap', 'status', 'max_rss_kb']:
                if key in audit:
                    row[key] = audit[key]
            if 'audit' in audit:
                row['audit'] = dict(audit['audit'])
                row['audit'].pop('spectrum', None)
        elif (folder / 'exception.json').exists():
            row['failure'] = json.loads((folder / 'exception.json').read_text())
        fresh.append(row)
    aliases = []
    for name, alias in json.loads((root / 'data/aliases.json').read_text()).items():
        row = copy.deepcopy(source[alias['source_case']])
        row.update(case=name, canonical_config=alias['config'],
                   provenance='reused identical finite problem', source_case=alias['source_case'])
        row.get('audit', {}).pop('spectrum', None)
        aliases.append(row)
    return seed, fresh, aliases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root)
    seed, fresh, aliases = collect(root)
    combined = seed + aliases + fresh
    for name, rows in [('refinement_records.json', aliases + fresh),
                       ('combined_records.json', combined)]:
        target = root / 'data' / name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(rows, indent=0) + '\n')
        temporary.replace(target)
    counts = dict(Counter(r['classification'] for r in fresh))
    print(json.dumps(dict(completed_new=len(fresh), aliases=len(aliases),
                          total_records=len(combined), classifications=counts)))


if __name__ == '__main__':
    main()
