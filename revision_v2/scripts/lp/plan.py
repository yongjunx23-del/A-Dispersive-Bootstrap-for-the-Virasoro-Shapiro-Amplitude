"""Finite, deterministic boundary-refinement plan. Kernels and finite spectral problems are unchanged."""
import hashlib
import json
from pathlib import Path
import gmpy2 as g

g.get_context().precision=512
M=g.mpfr
C0,C1,C2=2*g.zeta(3),-2*g.zeta(5),2*g.zeta(7)
ROOT=Path(__file__).resolve().parents[1]
SEED=json.loads((ROOT/'data/seed_records.json').read_text())
SEED=[r for r in SEED if r.get('accepted') and not r['canonical_config'].get('t_grid')]

def D(x):return format(M(x),'.150g')

def mathematical_key(c):
    fixed0=c.get('fix_c00',True);fixed1=c.get('fix_c10',False)
    q=dict(energy_grid=c['energy_grid'],n_mu=c['n_mu'],j_max=c['j_max'],
           n_t=c.get('n_t',0) if c.get('gravity') else 0,gravity=bool(c.get('gravity')),
           max_null_power=c['max_null_power'],nl=c.get('nl',0),fix_c00=fixed0,fix_c10=fixed1,
           t_grid=c.get('t_grid','chebyshev'),column_scale=c.get('column_scale'),
           eikonal_mu_max=c.get('eikonal_mu_max'),
           objective={k:D(v) for k,v in c.get('objective',{'c20':'1'}).items() if M(v)!=0})
    if fixed0:q['c00']=D(c.get('c00',C0*M(c.get('alpha','1'))**3))
    if fixed1:q['c10']=D(c.get('c10',C1*M(c.get('alpha','1'))**5))
    return hashlib.sha256(json.dumps(q,sort_keys=True).encode()).hexdigest()

BY_KEY={mathematical_key(r['canonical_config']):r for r in SEED}

def signature(c):
    return '%s_co%d_J%d_t%d'%('G' if c['gravity'] else 'P',c['n_mu'],c['j_max'],c.get('n_t',0))

def memory_gb(c):
    return 2.0 if not c['gravity'] else 1.5+c['n_mu']*(c['j_max']//2+1)*c['n_t']*140/1e9

def base(gravity,**kw):
    c=dict(alpha='1',audit_tol='1e-25',full_master=False,seed_energies=32,
           campaign='presentation_refinement_20261014_v1',energy_grid='chebyshev_open',
           n_mu=512,j_max=10000 if gravity else 2000,n_t=6 if gravity else 0,
           max_null_power=10 if gravity else 12,gravity=gravity,
           solver_tol='1e-36' if gravity else '1e-30',max_exchange=64 if gravity else 48,
           variant='base',stage='presentation_refinement_20261014_v1')
    c.update(kw);return c

ALPHAS=[M(v) for v in ['0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45',
                       '0.50','0.55','0.60','0.65','0.70','0.75','0.80','0.85',
                       '0.90','0.925','0.95','0.97','0.98','0.99','1']]
OLD_G=[M(v) for v in ['0.2','0.5','0.8','1.1','1.4','1.7','2','2.25','2.35','2.4']]
TIP_G=[C0+M(k)*M('0.00005') for k in range(-10,8)]
NL3_G=sorted(set([C0*a**3 for a in ALPHAS]+OLD_G+[(a+b)/2 for a,b in zip(OLD_G,OLD_G[1:])]+
                   TIP_G+[C0+M(k)*M('0.0002') for k in range(2,23)]))
NL6_G=sorted(set([C0*a**3 for a in ALPHAS]+OLD_G+TIP_G))
SCAN_POINTS=17
EDGE_RELATIVE_TARGET=M('0.005')  # bracket width <=0.5% of sampled feasible span
MAX_EDGE_STEPS=24

class Plan:
    def __init__(self,lookup):
        self.lookup=lookup;self.tasks={};self.aliases={}
    def add(self,name,cfg,priority):
        c=dict(cfg,name=name,priority=priority)
        seed=BY_KEY.get(mathematical_key(c))
        if seed is not None:
            self.aliases[name]=dict(config=c,source_case=seed['case'])
            return seed_result(seed)
        self.tasks[name]=c
        return self.lookup(name)
    def pair(self,i,gamma):
        a=self.add('ref_gnl3_%03d_min'%i,base(True,kind='gnl3_slice',nl=3,fix_c00=True,
                   fix_c10=False,c00=D(gamma),objective={'c10':'1'},gamma=D(gamma)),1)
        b=self.add('ref_gnl3_%03d_max'%i,base(True,kind='gnl3_slice',nl=3,fix_c00=True,
                   fix_c10=False,c00=D(gamma),objective={'c10':'-1'},gamma=D(gamma)),1)
        return (a,b)
    def test(self,name,gamma,x,priority=2):
        return self.add(name,base(True,kind='gnl6_scan',nl=6,fix_c00=True,fix_c10=True,
                        c00=D(gamma),c10=D(x),objective={},gamma=D(gamma)),priority)
    def boundary(self,i,gamma,span):
        lo,hi=span
        # Existing accepted tests on this exact slice are legitimate starting brackets.
        points={}
        for r in SEED:
            c=r['canonical_config']
            if (c.get('gravity') and c.get('nl')==6 and c.get('fix_c00') and c.get('fix_c10')
                and c.get('n_mu')==512 and c.get('j_max')==10000 and c.get('n_t')==6
                and c.get('max_null_power')==10 and c['energy_grid']=='chebyshev_open'
                and not c.get('eikonal_mu_max') and not c.get('column_scale')
                and M(c['c00'])==M(D(gamma)) and c.get('objective')=={}):
                points[M(c['c10'])]=r['classification']
        if len(points)<SCAN_POINTS:
            for k in range(SCAN_POINTS):
                x=lo+(M(k)+M('0.5'))*(hi-lo)/SCAN_POINTS
                r=self.test('ref_gnl6_%03d_u%02d'%(i,k),gamma,x)
                points[x]=r['cls'] if r else None
        # Always test the exact string reference for alpha<=1; above it this is a trial point.
        xref=C1*(gamma/C0)**(M(5)/3)
        r=self.test('ref_gnl6_%03d_string'%i,gamma,xref,1)
        points[xref]=r['cls'] if r else None
        if any(c is None for c in points.values()):return
        valid=sorted((x,c) for x,c in points.items() if c in ('feasible','infeasible'))
        fs=[x for x,c in valid if c=='feasible']
        if not fs:return
        explored=list(fs)
        for edge,((a,ca),(b,cb)) in enumerate(zip(valid,valid[1:])):
            if {ca,cb}!={'feasible','infeasible'}:continue
            for step in range(1,MAX_EDGE_STEPS+1):
                width=max(explored)-min(explored)
                target=EDGE_RELATIVE_TARGET*width if width>0 else M('1e-12')*max(abs(xref),M('1e-8'))
                if b-a<=target:break
                x=(a+b)/2
                r=self.test('ref_gnl6_%03d_e%02d_%02d'%(i,edge,step),gamma,x)
                if r is None or r['cls'] not in ('feasible','infeasible'):break
                if r['cls']=='feasible':explored.append(x)
                if r['cls']==ca:a=x
                else:b=x
    def build(self):
        # Nested uniform direction grids, with exact cardinal directions.
        for gravity,nl,n in [(True,0,192),(False,0,192),(False,3,384)]:
            for j in range(n):
                if j%(n//4)==0:
                    u,v=[('-1','0'),('0','-1'),('1','0'),('0','1')][j//(n//4)]
                else:
                    t=2*g.const_pi()*j/n;u,v=D(-g.cos(t)),D(-g.sin(t))
                fields=dict(kind='glin_direction' if gravity else 'pdir',nl=nl,fix_c00=not gravity,
                            fix_c10=False,direction_index=j,direction_count=n)
                if gravity:fields['objective']={'c00':u,'c10':v}
                else:fields.update(c00=D(C0),objective={'c10':u,'c20':v})
                self.add('ref_%s_nl%d_dir%03d'%('g' if gravity else 'p',nl,j),base(gravity,**fields),3)
        pairs={}
        for i,gamma in enumerate(NL3_G):pairs[D(gamma)]=self.pair(i,gamma)
        for i,gamma in enumerate(NL6_G):
            a,b=pairs[D(gamma)]
            if a and b and a['cls']==b['cls']=='feasible':
                self.boundary(i,gamma,(a['c10'],b['c10']))
        for alpha in [M(v) for v in ['0.1','0.2','0.4','0.6','0.8','1']]:
            for sign,label in [('1','min'),('-1','max')]:
                self.add('ref_p_string_a%s_%s'%(D(alpha).replace('.','p'),label),
                         base(False,kind='presentation_pstring',nl=6,fix_c00=True,fix_c10=True,
                              c00=D(C0*alpha**3),c10=D(C1*alpha**5),reference_alpha=D(alpha),
                              reference_c20=D(C2*alpha**7),objective={'c20':sign}),1)
        return self

def seed_result(r):
    out={'cls':r['classification']}
    for k,v in r.get('audit',{}).get('wilson',{}).items():out[k]=M(v)
    return out

def plan(lookup):return Plan(lookup).build()
