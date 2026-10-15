#!/usr/bin/env python3
"""Reconstruct a dual certificate from the active columns at guard precision.

This is an external audit of the same finite problem, not a second solver.
It never promotes a certificate before checking the complete declared grid.
"""
import argparse
import json
from pathlib import Path

from bootstrap_sdpx import M, ZERO, ONE, Model, dec, write_json


def solve_square(A, b):
    n = len(b)
    A = [list(row) + [rhs] for row, rhs in zip(A, b)]
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[pivot][k]) < M('1e-100'):
            raise ArithmeticError('Singular active-column system')
        A[k], A[pivot] = A[pivot], A[k]
        fac = A[k][k]
        for j in range(k, n + 1):
            A[k][j] /= fac
        for i in range(k + 1, n):
            fac = A[i][k]
            if fac:
                for j in range(k, n + 1):
                    A[i][j] -= fac * A[k][j]
    x = [ZERO] * n
    for k in reversed(range(n)):
        x[k] = A[k][n] - sum((A[k][j] * x[j] for j in range(k + 1, n)), ZERO)
    return x


def reconstruct_full_basis(model, point, mapping):
    indices = mapping['indices']
    scales = [M(x) for x in mapping['scales']]
    weights = [M(x) for x in point['z']]
    ranked = sorted(range(len(indices)), key=lambda j: weights[j], reverse=True)
    echelon, selected = [], []
    for j in ranked:
        v = [a / scales[j] for a in model.column(indices[j])]
        for pivot, vector in echelon:
            factor = v[pivot]
            v = [a - factor * b for a, b in zip(v, vector)]
        pivot = max(range(len(v)), key=lambda k: abs(v[k]))
        if abs(v[pivot]) < M('1e-80'):
            continue
        divisor = v[pivot]
        echelon.append((pivot, [a / divisor for a in v]))
        selected.append(j)
        if len(selected) == len(model.rows):
            break
    if len(selected) != len(model.rows):
        raise ArithmeticError('Active support does not span all equality rows')
    A = [[a / scales[j] for a in model.column(indices[j])] for j in selected]
    cost = [model.costval(indices[j]) / scales[j] for j in selected]
    y = solve_square(A, cost)
    minimum, worst, _ = model.separation(y)
    rho = [M(z) / h for z, h in zip(point['z'], scales)]
    primal = sum((model.costval(i) * w for i, w in zip(indices, rho)), ZERO)
    dual = sum((b * yy for b, yy in zip(model.b, y)), ZERO)
    result = dict(precompute_bits=384, polished_y=[dec(v) for v in y],
                  selected_indices=[indices[j] for j in selected],
                  minimum_full_grid_slack=dec(minimum), worst_index=worst,
                  primal_objective=dec(primal), dual_objective=dec(dual), gap=dec(abs(primal-dual)))
    if 'fix_c00' in model.names and minimum < 0:
        y[model.names.index('fix_c00')] += minimum / 2 - M('1e-90')
        repaired, worst, _ = model.separation(y)
        dual = sum((b * yy for b, yy in zip(model.b, y)), ZERO)
        result.update(repaired_y=[dec(v) for v in y], repaired_minimum_full_grid_slack=dec(repaired),
                      repaired_dual_objective=dec(dual), repaired_gap=dec(abs(primal-dual)))
    result['passed'] = (M(result.get('repaired_minimum_full_grid_slack', result['minimum_full_grid_slack'])) >= -M('1e-80')
                        and M(result.get('repaired_gap', result['gap'])) < M('1e-25'))
    return result


def refine_native(model, point, mapping, relative_threshold):
    """Keep native free dual components when physical active columns have low rank.

    A degenerate optimizer need not determine all dual components. Forcing
    tiny barrier weights to be active can select a very different functional.
    This correction imposes only independent significant active equations,
    then checks every original grid inequality and the objective gap.
    """
    indices=mapping['indices'];scales=[M(v) for v in mapping['scales']]
    physical=[M(z)/h for z,h in zip(point['z'],scales)]
    largest=max(physical,default=ZERO)
    ranked=sorted(range(len(indices)),key=lambda j:physical[j],reverse=True)
    native=[M(v) for v in point['x']]
    echelon=[];selected=[]
    for j in ranked:
        if physical[j] <= largest*M(relative_threshold):continue
        v=[a/scales[j] for a in model.column(indices[j])]
        for pivot,previous in echelon:
            factor=v[pivot];v=[a-factor*b for a,b in zip(v,previous)]
        pivot=max(range(len(v)),key=lambda k:abs(v[k]))
        if abs(v[pivot])<M('1e-85'):continue
        divisor=v[pivot];echelon.append((pivot,[a/divisor for a in v]));selected.append(j)
        if len(selected)==len(model.rows):break
    rows=[]
    for j in selected:
        a=[v/scales[j] for v in model.column(indices[j])]
        residual=model.costval(indices[j])/scales[j]-sum((v*y for v,y in zip(a,native)),ZERO)
        rows.append(a+[residual])
    pivots=[];m=len(rows);n=len(native)
    for k in range(m):
        available=[j for j in range(n) if j not in pivots]
        i,pivot=max(((i,j) for i in range(k,m) for j in available),key=lambda ij:abs(rows[ij[0]][ij[1]]))
        if abs(rows[i][pivot])<M('1e-100'):raise ArithmeticError('Rank loss in native-dual correction')
        rows[k],rows[i]=rows[i],rows[k];factor=rows[k][pivot]
        rows[k]=[v/factor for v in rows[k]]
        for i in range(m):
            if i==k:continue
            factor=rows[i][pivot]
            if factor:rows[i]=[a-factor*b for a,b in zip(rows[i],rows[k])]
        pivots.append(pivot)
    correction=[ZERO]*n
    for k,pivot in enumerate(pivots):correction[pivot]=rows[k][-1]
    return [a+b for a,b in zip(native,correction)],selected


def evaluate_candidate(model, point, mapping, y, method, interior_y=None):
    # Audit the serialized precision, not an unrecorded higher-precision vector.
    y=[M(dec(v)) for v in y]
    minimum,worst,_=model.separation(y)
    repaired=False
    if minimum<0:
        margin=M('1e-35')
        if 'fix_c00' in model.names:
            y[model.names.index('fix_c00')]+=minimum/2-margin
            repaired=True
        else:
            cost={k:v for k,v in model.cost.items() if v!=0}
            if set(cost)=={(0,0)}:
                f0=2*cost[(0,0)];denominator=f0-minimum+margin
                if f0*denominator>0:
                    factor=f0/denominator
                    y=[factor*v for v in y];repaired=True
            elif interior_y is not None:
                # A checked negative functional E^T y_g <= -2 gives a strict
                # feasible dual for any of these forward-moment objectives.
                fmin=min(model.costval(p*model.ns) for p in range(model.nmu))
                scale=max(ZERO,-fmin/2)+ONE
                strict=[scale*v for v in interior_y]
                strict_min,_,_=model.separation(strict)
                if strict_min>=ONE:
                    delta=-minimum
                    mix=(delta+margin)/(strict_min+delta+margin)
                    y=[(ONE-mix)*a+mix*b for a,b in zip(y,strict)]
                    repaired=True
        if repaired:
            y=[M(dec(v)) for v in y];minimum,worst,_=model.separation(y)
    scales=[M(v) for v in mapping['scales']]
    rho=[M(z)/h for z,h in zip(point['z'],scales)]
    primal=sum((model.costval(i)*w for i,w in zip(mapping['indices'],rho)),ZERO)
    dual=sum((b*v for b,v in zip(model.b,y)),ZERO)
    return dict(precompute_bits=384,polished_y=[dec(v) for v in y],
                method=method,uniform_correction_applied=repaired,
                minimum_full_grid_slack=dec(minimum),worst_index=worst,
                primal_objective=dec(primal),dual_objective=dec(dual),gap=dec(abs(primal-dual)),
                passed=(minimum>=-M('1e-80') and abs(primal-dual)<M('1e-25')))


def high_spin_null_correction(model, vector):
    """Propose a tiny homogeneous-row shift for ill-scaled high-spin slacks.

    Its RHS is zero, so it preserves b^T y. This proposal is never itself an
    acceptance condition: the serialized vector, all grid inequalities, and
    the original objective gap still pass the usual independent audit.
    """
    if not model.nt or 'fix_c00' in model.names or model.cfg.get('nl',0):
        return None,[]
    name='null_0_%d'%model.cfg['max_null_power']
    if name not in model.names:return None,[]
    row=model.names.index(name)
    if model.b[row]!=0:return None,[]
    y=list(vector);steps=[];total=ZERO
    for iteration in range(12):
        minimum,worst,_=model.separation(y)
        if minimum>=-M('1e-30'):return y,steps
        spin=model.spins[worst%model.ns]
        coefficient=model.column(worst)[row]
        if spin<64 or coefficient<=0:return (y,steps)if steps else(None,[])
        shift=(-minimum+M('1e-35'))/coefficient
        total+=shift
        if total>M('1e-20'):return None,steps
        y[row]-=shift
        steps.append(dict(row=name,index=worst,spin=spin,shift=dec(shift),
                          preceding_minimum_full_grid_slack=dec(minimum)))
    return y,steps


def reconstruct(model, point, mapping, interior_y=None):
    try:
        full=reconstruct_full_basis(model,point,mapping)
        if 'polished_y' in full:
            serialized=evaluate_candidate(model,point,mapping,
                [M(v) for v in full.get('repaired_y',full['polished_y'])],
                '384-bit active-column reconstruction, serialized-vector complete-grid check',interior_y)
            serialized['selected_indices']=full['selected_indices']
            if serialized['passed']:return serialized
            repaired,steps=high_spin_null_correction(model,
                [M(v)for v in full.get('repaired_y',full['polished_y'])])
            if repaired is not None and steps:
                corrected=evaluate_candidate(model,point,mapping,repaired,
                    'homogeneous null-row correction, serialized complete-grid and objective-gap check',interior_y)
                corrected.update(homogeneous_null_corrections=steps,selected_indices=full['selected_indices'])
                if corrected['passed']:return corrected
                full['homogeneous_correction_audit']={k:v for k,v in corrected.items()if k!='polished_y'}
    except ArithmeticError as exc:
        full=dict(passed=False,full_basis_note=str(exc))
    candidates=[]
    if 'fix_c00' in model.names:
        y=[ZERO]*len(model.rows)
        y[model.names.index('fix_c00')]=min(model.costval(p*model.ns) for p in range(model.nmu))/2
        simple=evaluate_candidate(model,point,mapping,y,'forward-moment minimum with checked complete-grid certificate')
        candidates.append(simple)
        if simple['passed']:return simple
    native=evaluate_candidate(model,point,mapping,[M(v) for v in point['x']],
                              'native dual with checked uniform correction',interior_y)
    candidates.append(native)
    if native['passed']:return native
    for threshold in ['1e-12','1e-24','1e-40','1e-60']:
        try:
            y,selected=refine_native(model,point,mapping,threshold)
            candidate=evaluate_candidate(model,point,mapping,y,
                       '384-bit native-dual correction on independent physical active columns',interior_y)
            candidate.update(relative_active_weight_threshold=threshold,
                             selected_indices=[mapping['indices'][j] for j in selected])
            candidates.append(candidate)
            if candidate['passed']:return candidate
        except ArithmeticError:
            continue
    full['additional_candidate_audits']=[{k:v for k,v in d.items() if k!='polished_y'} for d in candidates]
    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--point', required=True)
    p.add_argument('--mapping', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    model = Model(json.loads(Path(args.config).read_text()))
    result = reconstruct(model, json.loads(Path(args.point).read_text()), json.loads(Path(args.mapping).read_text()))
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['polished_y', 'repaired_y', 'selected_indices']}))


if __name__ == '__main__':
    main()
