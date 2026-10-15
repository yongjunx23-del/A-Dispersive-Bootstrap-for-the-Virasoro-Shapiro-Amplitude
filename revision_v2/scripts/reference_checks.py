"""Small-alpha checks of the string projection on the declared finite LPs."""
import mpmath as mp

ALPHAS = ['0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45','0.50',
          '0.55','0.60','0.65','0.70','0.75','0.80','0.85','0.90',
          '0.925','0.95','0.97','0.98','0.99','1']


def reference_checks(records):
    z3, z5, z7 = 2*mp.zeta(3), -2*mp.zeta(5), 2*mp.zeta(7)
    rows = []
    for alpha in ALPHAS:
        a = mp.mpf(alpha)
        gamma, xref, yref = z3*a**3, z5*a**5, z7*a**7
        gravity, pole = [], []
        for d in records:
            c = d['canonical_config']
            if (c.get('energy_grid') != 'chebyshev_open' or c.get('n_mu') != 512
                    or c.get('t_grid') or c.get('column_scale') or c.get('eikonal_mu_max')
                    or not c.get('fix_c00', True)):
                continue
            c00 = mp.mpf(c.get('c00', z3*mp.mpf(c.get('alpha','1'))**3))
            if abs(c00-gamma) > mp.mpf('1e-35'):
                continue
            if c.get('gravity'):
                if c.get('j_max') == 10000 and c.get('n_t') == 6 and c.get('max_null_power') == 10:
                    gravity.append(d)
            elif c.get('j_max') == 2000 and c.get('max_null_power') == 12:
                pole.append(d)
        pair = {}
        nl3_excluded = False
        nl3_string_states = []
        tests = {}
        string_states = []
        used = set()
        for d in gravity:
            c = d['canonical_config']
            cls = d['classification'] if d.get('accepted') else 'unresolved'
            if c.get('nl') == 3 and not c.get('fix_c10', False) and cls == 'infeasible':
                nl3_excluded = True
                used.add(d['case'])
            if c.get('nl') == 3 and c.get('fix_c10') and c.get('objective') == {}:
                x=mp.mpf(c.get('c10', z5*mp.mpf(c.get('alpha','1'))**5))
                if abs(x-xref)<mp.mpf('1e-35'):
                    nl3_string_states.append(cls)
                    used.add(d['case'])
            if (c.get('nl') == 3 and not c.get('fix_c10', False)
                    and set(c.get('objective', {})) == {'c10'} and cls == 'feasible'):
                pair['min' if mp.mpf(c['objective']['c10']) > 0 else 'max'] = d
                used.add(d['case'])
            if c.get('nl') == 6 and c.get('fix_c10') and c.get('objective') == {}:
                x = mp.mpf(c.get('c10', z5*mp.mpf(c.get('alpha','1'))**5))
                key = mp.nstr(x, 45)
                previous = tests.get(key)
                if previous in ('feasible','infeasible') and cls in ('feasible','infeasible') and previous != cls:
                    raise ValueError('Conflicting audited string-slice tests')
                if previous is None or cls in ('feasible','infeasible'):
                    tests[key] = cls
                if abs(x-xref) < mp.mpf('1e-35'):
                    string_states.append(cls)
                used.add(d['case'])
        runs, run = [], []
        for key, cls in sorted(tests.items(), key=lambda q: mp.mpf(q[0])):
            if cls == 'feasible':
                run.append(mp.mpf(key))
            elif run:
                runs.append((min(run),max(run)))
                run = []
        if run:
            runs.append((min(run),max(run)))
        ppair = {}
        for d in pole:
            c = d['canonical_config']
            if not (c.get('nl') == 6 and c.get('fix_c10') and d.get('accepted')
                    and d['classification'] == 'feasible' and set(c.get('objective', {})) == {'c20'}):
                continue
            c10 = mp.mpf(c.get('c10', z5*mp.mpf(c.get('alpha','1'))**5))
            if abs(c10-xref) < mp.mpf('1e-35'):
                ppair['min' if mp.mpf(c['objective']['c20']) > 0 else 'max'] = d
                used.add(d['case'])
        n3 = tuple(mp.mpf(pair[label]['audit']['wilson']['c10']) for label in ('min','max')) if len(pair)==2 else None
        p6 = tuple(mp.mpf(ppair[label]['audit']['wilson']['c20']) for label in ('min','max')) if len(ppair)==2 else None
        state = next((s for s in ('feasible','infeasible','unresolved') if s in string_states), 'pending')
        if state=='pending' and nl3_excluded:state='excluded NL3 slice'
        n3state=next((s for s in ('feasible','infeasible','unresolved') if s in nl3_string_states),'pending')
        if n3state=='pending':
            if nl3_excluded:n3state='infeasible'
            elif n3 and n3[0]<=xref<=n3[1]:n3state='feasible'
            elif n3 and (xref<n3[0]-mp.mpf('1e-23') or xref>n3[1]+mp.mpf('1e-23')):n3state='infeasible'
        rows.append(dict(alpha=alpha, gamma=gamma, string_c10=xref, string_c20=yref,
                         gravity_nl3=n3, gravity_nl6=runs, gravity_string_status=state,
                         gravity_nl3_string_status=n3state,
                         pole_nl6=p6, pole_string_inside=(p6[0]<=yref<=p6[1]) if p6 else None,
                         source_cases=sorted(used)))
    return rows


def tail_checks(records, references=None):
    references=reference_checks(records) if references is None else references
    lookup={mp.mpf(r['alpha']):r for r in references}
    rows=[]
    for alpha in ('0.1','0.2','0.3','0.35','0.4'):
        ref=lookup[mp.mpf(alpha)]
        for nl in (3,6):
            baseline=(ref['gravity_nl3_string_status'] if nl==3 else ref['gravity_string_status'])
            if baseline=='excluded NL3 slice':baseline='infeasible'
            statuses={'base':baseline}
            sources=set(ref['source_cases'])
            for reach in (6,8,12):
                variant='tail_E%d'%reach
                found=[d for d in records if d['canonical_config']['kind']=='presentation_smallalpha_tail'
                       and mp.mpf(d['canonical_config']['reference_alpha'])==mp.mpf(alpha)
                       and d['canonical_config']['nl']==nl and d['canonical_config']['variant']==variant]
                statuses[variant]=next((d['classification'] for d in found if d.get('accepted')),
                                       'unresolved' if found else 'pending')
                sources.update(d['case'] for d in found)
            rows.append(dict(alpha=alpha,nl=nl,statuses=statuses,source_cases=sorted(sources)))
    return rows
