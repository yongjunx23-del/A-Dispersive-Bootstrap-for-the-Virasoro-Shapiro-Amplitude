#!/usr/bin/env python3
"""Retain raw solver results and write separate guard-precision verification."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

from bootstrap_sdpx_v5 import Model, M, ZERO, ONE, dec, write_json
from polish_certificate_v5 import reconstruct
from audit_unbounded import check as check_unbounded
from batch_runner import physical_cpus


def negative_interior(model, root):
    """Check an auxiliary E^T y <= -2 on this target's complete grid.

    The source functional only supplies a candidate. No source-grid status
    is used as a substitute for checking the target grid.
    """
    if not model.nt or model.cfg.get('nl', 0) or 'fix_c00' in model.names:
        return None, None
    candidates=[]
    for power in [2,3,4,5,6,8,10]:
        folder=root/'results'/('gr_linear_c00_N300_J3000_t6_p%d_max'%power)
        summary=folder/('continued_summary.json' if (folder/'continued_summary.json').exists() else 'summary.json')
        if not summary.exists():continue
        raw=json.loads(summary.read_text())
        if not raw.get('rounds'):continue
        source=root/'results'/raw['continued_directory'] if raw.get('continued_directory') else folder
        final=int(raw.get('final_round',raw['rounds'][-1]['round']))
        path=source/('round_%02d'%final)
        if not (path/'point.json').exists():continue
        point=json.loads((path/'point.json').read_text())
        if point.get('status')!='Solved':continue
        names=json.loads((path/'mapping.json').read_text())['row_names']
        if not set(names).issubset(model.names):continue
        vectors=[]
        for version in ['v3','v2']:
            p=folder/('polished_certificate_%s.json'%version)
            if p.exists():
                d=json.loads(p.read_text())
                if d.get('passed') and d.get('polished_y'):vectors.append(d['polished_y'])
        vectors.append(point['x'])
        for vector in vectors:
            by_name=dict(zip(names,vector))
            candidates.append(([M(by_name.get(name,'0')) for name in model.names],folder.name))
    saved=model.cost
    model.cost={(0,0):-ONE}
    try:
        for y,source in candidates:
            minimum,_,_=model.separation(y)
            margin=M('1e-35')
            denominator=-2-minimum+margin
            if minimum<0 and denominator<0:y=[(-2/denominator)*v for v in y]
            y=[M(dec(v)) for v in y]
            minimum,_,_=model.separation(y)
            if minimum>=-M('1e-80'):
                return y,dict(source_case=source,minimum_auxiliary_slack=dec(minimum),
                              checked_target_columns=model.N,auxiliary_y=[dec(v) for v in y])
    finally:
        model.cost=saved
    return None,None


def verify_case(cfg, root):
    target = root/'results'/cfg['name']
    destination = target/'verified_summary_v5.json'
    if destination.exists():
        return json.loads(destination.read_text())
    previous=target/'verified_summary_v4.json'
    if previous.exists():
        earlier=json.loads(previous.read_text())
        if earlier.get('accepted'):
            earlier['verification_reused_from_version']=8
            write_json(destination,earlier);return earlier
    raw_path=target/('continued_summary.json' if (target/'continued_summary.json').exists() else 'summary.json')
    if not raw_path.exists():
        return None
    raw = json.loads(raw_path.read_text())
    d = dict(raw)
    d['raw_accepted'] = raw['accepted']
    d['verification_version']=9
    if not raw.get('rounds'):
        if raw.get('classification') == 'infeasible' and raw.get('certificate_y'):
            model=Model(cfg)
            y=[M(v) for v in raw['certificate_y']]
            normal=sum((b*v for b,v in zip(model.b,y)),ZERO)
            minimum,_,_=model.separation(y,ray=True)
            d.update(accepted=(normal>0 and minimum>=-M('1e-80')),
                     verification='independent 384-bit complete-grid check of reused ray',
                     certificate_b_dot_y=dec(normal),minimum_full_grid_ray_slack=dec(minimum))
            if not d['accepted']: d.update(classification='unresolved',failure='reused_ray_audit_failed')
        write_json(destination, d); return d
    final = int(raw.get('final_round', raw['rounds'][-1]['round']))
    source=root/'results'/raw['continued_directory'] if raw.get('continued_directory') else target
    path = source/('round_%02d' % final)
    point = json.loads((path/'point.json').read_text()) if (path/'point.json').exists() else {}
    if point.get('status')=='PrimalInfeasible':
        model=Model(cfg)
        mapping=json.loads((path/'mapping.json').read_text())
        recession=check_unbounded(model,point,mapping,root)
        write_json(target/'primal_recession_certificate.json',recession)
        if recession['passed']:
            d.update(accepted=True,classification='unbounded',
                     verification='384-bit positive recession-ray audit and independent feasible base point',
                     unbounded_audit={k:v for k,v in recession.items()if k not in ['ray_indices','ray_weights']})
            d.pop('failure',None)
        write_json(destination,d);return d
    if point.get('status') not in ['Solved', 'DualInfeasible']:
        write_json(destination, d); return d
    model = Model(cfg)
    tolerance = M(cfg.get('audit_tol', '1e-25'))
    if point['status'] == 'Solved':
        mapping = json.loads((path/'mapping.json').read_text())
        scales = [M(x) for x in mapping['scales']]
        _, audit = model.audit(point, mapping['indices'], scales)
        y=[M(v) for v in point['x']]
        minimum,_,_=model.separation(y)
        good_primal = (M(audit['normalized_equality_residual']) <= tolerance
                       and M(audit['nonnegativity_violation']) <= tolerance)
        good_native = (good_primal and minimum >= -tolerance
                       and M(audit['gap']) <= tolerance * max(ONE,abs(M(audit['primal_objective']))))
        d.update(accepted=good_native,classification='feasible' if good_native else 'unresolved',
                 audit=audit,minimum_full_grid_slack=dec(minimum),
                 verification='384-bit original equations and complete-grid native dual check')
        if good_native: d.pop('failure',None)
        if not any(model.cost.values()) and good_primal:
            # Exact zero is a full-grid dual certificate for a feasibility LP.
            d.update(accepted=True,classification='feasible',verified_gap='0',
                     minimum_full_grid_slack='0',repaired_dual_objective='0',
                     verification='384-bit positive-primal audit and exact zero-objective dual')
            d.pop('failure',None)
            write_json(destination,d);return d
        try:
            interior,auxiliary=negative_interior(model,root) if set(model.cost)!={(0,0)} else (None,None)
            polished = reconstruct(model, point, mapping, interior_y=interior)
            if auxiliary:polished['auxiliary_interior_certificate']=auxiliary
            write_json(target/'polished_certificate_v5.json', polished)
            if polished['passed'] and good_primal:
                d.update(accepted=True, classification='feasible', audit=audit,
                         verification='384-bit active-column dual reconstruction and complete-grid check',
                         minimum_full_grid_slack=polished.get('repaired_minimum_full_grid_slack', polished['minimum_full_grid_slack']),
                         repaired_dual_objective=polished.get('repaired_dual_objective', polished['dual_objective']),
                         verified_gap=polished.get('repaired_gap', polished['gap']))
                d.pop('failure', None)
        except ArithmeticError as exc:
            d['polish_note'] = str(exc)
    else:
        y = [M(x) for x in point['x']]
        normal = sum((b*v for b,v in zip(model.b,y)), ZERO)
        if normal > 0:
            y = [v/normal for v in y]
            minimum, _, _ = model.separation(y, ray=True)
            if minimum >= -tolerance:
                if minimum < 0 and 'fix_c00' in model.names:
                    y[model.names.index('fix_c00')] += minimum/2 - M('1e-35')
                normal = sum((b*v for b,v in zip(model.b,y)), ZERO)
                if normal > 0:
                    y = [v/normal for v in y]
                    minimum, _, _ = model.separation(y, ray=True)
                    if minimum >= -M('1e-80'):
                        d.update(accepted=True, classification='infeasible',
                                 verification='384-bit complete-grid recession-ray check',
                                 certificate_y=[dec(v) for v in y], certificate_b_dot_y='1',
                                 minimum_full_grid_ray_slack=dec(minimum))
                        d.pop('failure', None)
    write_json(destination, d)
    return d


def verify_task(task):
    cfg,root=task
    try:
        d=verify_case(cfg,root)
        if d:
            return dict(case=d['case'],accepted=d['accepted'],classification=d['classification'],failure=d.get('failure'))
        return dict(case=cfg['name'],accepted=False,classification='unresolved',failure='missing_raw_summary')
    except Exception as exc:
        target=root/'results'/cfg['name']
        if target.exists():write_json(target/'verification_exception_v9.json',dict(message=str(exc),trace=traceback.format_exc()))
        return dict(case=cfg['name'],accepted=False,classification='unresolved',verification_error=str(exc))


def pin_worker(cores):
    index=(mp.current_process()._identity[0]-1)%len(cores)
    os.sched_setaffinity(0,{cores[index]})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--tag',default='all')
    args=p.parse_args()
    root=Path(args.root)
    cases=json.loads(Path(args.manifest).read_text())['cases']
    outcomes=[]
    if args.workers==1:
        iterator=map(verify_task,[(cfg,root) for cfg in cases])
        for outcome in iterator:
            outcomes.append(outcome);print(json.dumps(outcome),flush=True)
    else:
        cores=physical_cpus(args.workers)
        with mp.Pool(args.workers,initializer=pin_worker,initargs=(cores,)) as pool:
            for outcome in pool.imap_unordered(verify_task,[(cfg,root) for cfg in cases],chunksize=1):
                outcomes.append(outcome);print(json.dumps(outcome),flush=True)
    write_json(root/('results/verification_'+args.tag+'_completion.json'),dict(number_of_cases=len(cases),checked=len(outcomes),
               accepted=sum(r['accepted'] for r in outcomes),outcomes=outcomes))


if __name__=='__main__':
    main()
