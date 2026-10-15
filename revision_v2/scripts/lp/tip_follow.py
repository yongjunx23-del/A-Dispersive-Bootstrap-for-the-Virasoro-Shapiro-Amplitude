"""One bounded continuation scan near the last observed gravitational NL6 window."""
import argparse
import json
from pathlib import Path
from collect_refinement import collect
from plan import M,D,base,signature,mathematical_key


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    root=Path(parser.parse_args().root)
    receipt=root/'data/tip_follow_plan.json'
    if receipt.exists():
        print('tip continuation already enqueued');return
    seed,fresh,aliases=collect(root)
    rows=seed+aliases+fresh
    slices={}
    for r in rows:
        c=r['canonical_config']
        if (c['kind'] in ('gnl6_scan','gnl6_tip') and c.get('variant')=='base'
            and r.get('accepted') and r['classification']=='feasible'):
            slices.setdefault(M(D(c['c00'])),[]).append(M(c['c10']))
    gamma=max(slices);previous=max(g for g in slices if g<gamma)
    center=(min(slices[gamma])+max(slices[gamma]))/2
    old_center=(min(slices[previous])+max(slices[previous]))/2
    slope=(center-old_center)/(gamma-previous)
    width=max(slices[gamma])-min(slices[gamma])
    assert width>0
    upper=M('2.4043')
    assert gamma<upper
    known={mathematical_key(r['canonical_config']) for r in rows if r.get('accepted')}
    known.update(mathematical_key(json.loads(p.read_text())) for p in (root/'queue/tasks').glob('*.json'))
    configs=[]
    # Eight slices: the complete three-level midpoint mesh plus the upper endpoint.
    for i in range(1,9):
        g=gamma+(upper-gamma)*i/8
        predicted=center+slope*(g-gamma)
        for j in range(-10,11):
            x=predicted+M(j)*M('0.15')*width
            c=base(True,kind='gnl6_tip',nl=6,fix_c00=True,fix_c10=True,
                   c00=D(g),c10=D(x),objective={},tip_follow=True)
            key=mathematical_key(c)
            if key in known:continue
            c.update(name='ref_tip_follow_'+key[:24],priority=0)
            p=root/'queue/tasks'/('0__%s__%s.json'%(signature(c),c['name']))
            temporary=p.with_suffix('.tmp');temporary.write_text(json.dumps(c,sort_keys=True,indent=1)+'\n')
            temporary.rename(p);known.add(key);configs.append(c)
    assert len(configs)<=168
    receipt.write_text(json.dumps(dict(anchor_gamma=D(gamma),upper_gamma=D(upper),
                      anchor_width=D(width),center_slope=D(slope),cases=configs),indent=1)+'\n')
    print(json.dumps(dict(new_tip_continuation_tasks=len(configs),anchor_gamma=D(gamma))))


if __name__=='__main__':main()
