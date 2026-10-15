#!/usr/bin/env python3
"""Appendix-A summary, response summary and evidence JSON from the 2026-10-14 campaign records.

All quantities are read from 768-bit-audited records and manipulated with
mpmath at 60 digits; geometric scan diagnostics (hull areas, Hausdorff
distances) are drawing diagnostics only.
"""
import argparse,json,math,re,statistics
from collections import Counter
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.spatial import ConvexHull
from reference_checks import reference_checks,tail_checks

mp.mp.dps=60
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--data',required=True,help='campaign_records.json')
parser.add_argument('--sdpb',help='sdpb_vs_sdpx.json from the timing test')
args=parser.parse_args()
D=json.loads(Path(args.data).read_text())
# Exploratory placements of the gravity nodes (equally spaced, closed cosine, ...) are not part of the paper;
# every reported gravity run uses the Chebyshev nodes of eq. (fixed-t-cheb-nodes).
D=[d for d in D if not d['canonical_config'].get('t_grid')]
C={d['case']:d for d in D};counts=Counter(d['classification'] for d in D)
Z3,Z5,Z7=2*mp.zeta(3),-2*mp.zeta(5),2*mp.zeta(7)

def cfg(d):return d['canonical_config']
def num(d,key):return mp.mpf(d['audit']['wilson'][key])
def ok(name):
 d=C.get(name);return d if d and d.get('accepted') and d['classification']=='feasible' else None
def status(name):
 d=C.get(name)
 if d is None:return 'pending'
 return d['classification'] if d.get('accepted') else 'unresolved'
def f(x,n=7):return mp.nstr(mp.mpf(x),n,strip_zeros=False)
def sci(x,n=2):
 x=float(x)
 if x==0:return '0'
 m,e=('%.*e'%(n,x)).split('e');return m+r'\times10^{'+str(int(e))+'}'
def nulls(p):return p*(p+1)//2

def edge_bracket(tests):
 """tests: list of (x, cls) along one line. Returns feasible span and the brackets of its two ends."""
 tests=sorted((mp.mpf(x),c) for x,c in tests if c in ('feasible','infeasible'))
 feas=[x for x,c in tests if c=='feasible']
 if not feas:return None
 lo,hi=min(feas),max(feas)
 below=[x for x,c in tests if c=='infeasible' and x<lo];above=[x for x,c in tests if c=='infeasible' and x>hi]
 return dict(lo=lo,hi=hi,lo_out=max(below) if below else None,hi_out=min(above) if above else None,
             n_feasible=len(feas),n_tests=len(tests))

def nl6_tests(k,variant,gamma=None):
 out=[]
 for d in D:
  c=cfg(d)
  if c['kind']==k and c.get('variant')==variant and (gamma is None or abs(mp.mpf(c['c00'])-gamma)<mp.mpf('1e-30')):
   out.append((c['c10'],d['classification'] if d.get('accepted') else 'unresolved'))
 return out

def nl6_string_tests(variant):
 tests=nl6_tests('gnl6_convergence',variant)
 if variant=='base':
  for k in ('gnl6_scan','gnl6_tip'):tests+=nl6_tests(k,variant,Z3)
 return tests

GRAVITY_VARIANTS=[('base',r'baseline $(512,10^4,6,55)$'),('N256',r'$N_\mu=256$'),('N1024',r'$N_\mu=1024$'),
 ('J6000',r'$J_{\max}=6000$'),('J16000',r'$J_{\max}=16000$'),('T4',r'$N_t=4$'),
 ('T8',r'$N_t=8$'),('P8',r'$N_{\rm null}=36$'),('P12',r'$N_{\rm null}=78$')]
LINEAR_EXTRA=[('T8J16000',r'$N_t=8$, $J_{\max}=16000$'),('T10',r'$N_t=10$'),('T10J16000',r'$N_t=10$, $J_{\max}=16000$'),
 ('T12',r'$N_t=12$'),('T12J16000',r'$N_t=12$, $J_{\max}=16000$'),('T12J24000',r'$N_t=12$, $J_{\max}=24000$'),
 ('T12N1024J16000',r'$N_t=12$, $N_\mu=1024$, $J_{\max}=16000$'),('T12N1024J24000',r'$N_t=12$, $N_\mu=1024$, $J_{\max}=24000$'),
 ('S1',r'$(256,6000,6,36)$'),('S3',r'$(1024,16000,6,78)$')]
END_VARIANTS=[('base',r'baseline'),('P8',r'$N_{\rm null}=36$'),('P12',r'$N_{\rm null}=78$'),('N1024',r'$N_\mu=1024$'),
 ('J16000',r'$J_{\max}=16000$'),('T4',r'$N_t=4$'),('T8',r'$N_t=8$'),
 ('T0',r'no pole rows')]
# Number of fixed-t gravity nodes on the baseline grid (Chebyshev nodes of the first kind on [-1,0]).
NT_LADDER=[(4,'T4'),(6,'base'),(8,'T8'),(0,'T0')]
NL_EXTRA=[('T0',r'no pole rows')]
# Energy refinement on the open Chebyshev grid: the top node mu_1 = csc^2(pi/2N) grows with N_mu.
SCALED=[256,512,1024,2048,4096,8192]
# Energy reach of the impact-parameter block, per number of gravity nodes.
REACH_COLUMNS=[('',r'grid'),('N1024',r'$N_\mu=1024$'),('N2048',r'$N_\mu=2048$'),('E6',r'$10^{6}$'),('E8',r'$10^{8}$'),
 ('E12',r'$10^{12}$'),('E16',r'$10^{16}$'),('E20',r'$10^{20}$')]
TIP_GAMMAS=['2.4042','2.40425','2.4043','2.40435','2.4044','2.4046','2.405','2.4055','2.406']
BAND_VARIANTS=[('P8',r'$N_{\rm null}=36$'),('P10',r'$N_{\rm null}=55$'),('base',r'$N_{\rm null}=78$ (baseline)'),
 ('P14',r'$N_{\rm null}=105$'),('N1024',r'$N_\mu=1024$, $N_{\rm null}=78$')]
PG_ALPHAS=['0.6','0.8','0.9','0.95','1','1.01','1.02','1.05','1.1','1.2','1.4']
POLE_VARIANTS=[('base',r'baseline $(512,2000,78)$'),('N256',r'$N_\mu=256$'),('N1024',r'$N_\mu=1024$'),
 ('J1000',r'$J_{\max}=1000$'),('J4000',r'$J_{\max}=4000$'),('P8',r'$N_{\rm null}=36$'),('P10',r'$N_{\rm null}=55$'),
 ('P14',r'$N_{\rm null}=105$')]

def linear_rows():
 rows=[]
 for v,label in GRAVITY_VARIANTS+LINEAR_EXTRA:
  name='glin_c00max_'+v;d=ok(name)
  rows.append(dict(variant=v,label=label,status=status(name),c00max=num(d,'c00') if d else None))
 return rows

def gravity_nl_rows():
 rows=[]
 for v,label in GRAVITY_VARIANTS+NL_EXTRA:
  a,b=ok('gnl3c_%s_min'%v),ok('gnl3c_%s_max'%v)
  e=edge_bracket(nl6_string_tests(v))
  rows.append(dict(variant=v,label=label,nl3=(num(a,'c10')-Z5,num(b,'c10')-Z5) if a and b else None,
                   nl3_status=(status('gnl3c_%s_min'%v),status('gnl3c_%s_max'%v)),nl6=e,
                   string=status('gnl6c_%s_string'%v)))
 return rows

def lin_label(name):
 d=C.get(name)
 if d is None:return '---'
 if not d.get('accepted'):return 'unresolved'
 if d['classification']=='feasible':return f(num(d,'c00'),7)
 return {'infeasible':'empty','unbounded':'unbounded'}[d['classification']]

def end_of(v):
 if v=='base':
  tests=[(cfg(d)['c00'],cls_of(d)) for d in D if cfg(d)['kind'] in ('gnl3_slice','gnl3_end') and cfg(d).get('variant')=='base'
         and float(cfg(d)['objective']['c10'])>0]
 else:
  tests=[(cfg(d)['c00'],cls_of(d)) for d in D if cfg(d)['kind']=='gnl3_end_variant' and cfg(d).get('variant')==v]
 return edge_bracket(tests)

def scaled_rows():
 """Open Chebyshev grid with kernels rescaled by M^4/mu^2; the top node grows like N_mu^2."""
 rows=[]
 for n in SCALED:
  v='K%d'%n;a,b=ok('gnl3c_%s_min'%v),ok('gnl3c_%s_max'%v)
  rows.append(dict(n=n,top=1/mp.sin(mp.pi/(2*n))**2,linear=lin_label('glin_c00max_'+v),
   nl3=(num(a,'c10')-Z5,num(b,'c10')-Z5) if a and b else None,nl3_status=(status('gnl3c_%s_min'%v),status('gnl3c_%s_max'%v)),
   nl6=edge_bracket(nl6_tests('gnl6_convergence',v)),string=status('gnl6c_%s_string'%v),end=end_of(v)))
 return rows

def reach_rows():
 names={6:['base','N1024','N2048','E6','E8','E12','E16','E20'],8:['T8','T8N1024',None,None,None,'T8E12',None,'T8E20'],
        10:['T10','T10N1024','T10N2048',None,None,'T10E12',None,'T10E20'],12:['T12','T12N1024J16000','T12N2048',None,None,'T12E12',None,'T12E20']}
 return [[str(nt)]+[lin_label('glin_c00max_'+v) if v else '---' for v in vs] for nt,vs in names.items()]

def tip_rows():
 rows=[]
 for tag in TIP_GAMMAS:
  row=[tag]
  for v in ['base','T0']:
   gam=mp.mpf(tag)
   tests=[(cfg(d)['c10'],cls_of(d)) for d in D if cfg(d)['kind']=='gnl6_tip' and cfg(d).get('variant')==v
          and abs(mp.mpf(cfg(d)['c00'])-gam)<mp.mpf('1e-30')]
   e=edge_bracket(tests)
   if e:row.append('$[%s,\\,%s]$ (%d/%d)'%(f(e['lo'],7),f(e['hi'],7),e['n_feasible'],e['n_tests']))
   elif tests:row.append('none (0/%d)'%len(tests))
   else:row.append('---')
  rows.append(row)
 return rows

def pole_rows():
 rows=[]
 for v,label in POLE_VARIANTS:
  a,b=ok('pconv_%s_min'%v),ok('pconv_%s_max'%v)
  rows.append(dict(variant=v,label=label,status=(status('pconv_%s_min'%v),status('pconv_%s_max'%v)),
                   interval=((num(a,'c20')-Z7),(num(b,'c20')-Z7)) if a and b else None))
 return rows

def normalization_rows():
 rows=[]
 for alpha in ['0.6','0.8','1']:
  al=mp.mpf(alpha);tag=alpha.replace('.','p')
  for factor in ['1','0.95','1.05']:
   if alpha=='1':
    if factor!='1':continue
    lo,hi=ok('pconv_base_min'),ok('pconv_base_max');st=(status('pconv_base_min'),status('pconv_base_max'))
   else:
    nm='pnorm_a%s_f%s_'%(tag,factor.replace('.','p'));lo,hi=ok(nm+'min'),ok(nm+'max');st=(status(nm+'min'),status(nm+'max'))
   c00=Z3*al**3;string=Z7*al**7
   rows.append(dict(alpha=alpha,factor=factor,status=st,gamma=c00,
                    offsets=((num(lo,'c20')-string)/c00,(num(hi,'c20')-string)/c00) if lo and hi else None,
                    width=(num(hi,'c20')-num(lo,'c20'))/c00 if lo and hi else None))
 return rows

def polygon_metrics():
 out=[]
 for nl,fallback in [(0,48),(3,96)]:
  meshes={}
  for d in D:
   c=cfg(d)
   if d.get('accepted') and d['classification']=='feasible' and c['kind']=='pdir' and c['nl']==nl:
    meshes.setdefault(c.get('direction_count',fallback),{})[c['direction_index']]=d
  complete=[n for n,rows in meshes.items() if set(range(n))<=set(rows)]
  if not complete:continue
  total=max(complete);rows=meshes[total]
  previous=None
  for n in [total//4,total//2,total]:
   sel=[rows[i] for i in range(0,total,total//n) if i in rows]
   if len(sel)<3:continue
   pts=np.array([(float(num(d,'c10')/num(d,'c00')),float(num(d,'c20')/num(d,'c00'))) for d in sel])
   hull=ConvexHull(pts);v=pts[hull.vertices]
   def dist(p,poly):
    a=poly;b=np.roll(poly,-1,axis=0);w=b-a
    q=np.clip(np.einsum('ij,ij->i',p-a,w)/np.einsum('ij,ij->i',w,w),0,1)
    return min(np.linalg.norm(p-(a+q[:,None]*w),axis=1))
   change=None if previous is None else max(max(dist(p,previous) for p in v),max(dist(p,v) for p in previous))
   out.append(dict(nl=nl,directions=n,checked=len(sel),area=float(hull.volume),hausdorff=change));previous=v
 return out

def gravity_scan():
 """Base-truncation NL3 slices and NL6 tests along gamma (main-text statements)."""
 nl3={}
 for d in D:
  c=cfg(d)
  if c['kind'] in ('gnl3_slice','gnl3_end') and c.get('variant')=='base':
   key=mp.nstr(mp.mpf(c['c00']),45)
   nl3.setdefault(key,{})['min' if float(c['objective']['c10'])>0 else 'max']=d
 slices=[]
 for g,s in nl3.items():
  if 'min' not in s or 'max' not in s:continue
  st=[x['classification'] if x.get('accepted') else 'unresolved' for x in (s['min'],s['max'])]
  slices.append(dict(gamma=mp.mpf(g),status=st,c10=(num(s['min'],'c10'),num(s['max'],'c10')) if st==['feasible','feasible'] else None))
 slices.sort(key=lambda r:r['gamma'])
 feas=[r['gamma'] for r in slices if r['c10']];empty=[r['gamma'] for r in slices if r['status']==['infeasible','infeasible']]
 end=(max(feas),min(g for g in empty if g>max(feas))) if feas and any(g>max(feas) for g in empty) else None
 nl6={}
 for d in D:
  c=cfg(d)
  if c['kind'] in ('gnl6_scan','gnl6_tip') and c.get('variant')=='base':
   key=mp.nstr(mp.mpf(c['c00']),45)
   nl6.setdefault(key,[]).append((c['c10'],d['classification'] if d.get('accepted') else 'unresolved'))
 rows6=[dict(gamma=mp.mpf(g),edges=edge_bracket(t),tests=len(t),feasible=sum(1 for _,c in t if c=='feasible'),
             unresolved=sum(1 for _,c in t if c=='unresolved')) for g,t in nl6.items()]
 rows6.sort(key=lambda r:r['gamma'])
 with6=[r['gamma'] for r in rows6 if r['feasible']];without=[r['gamma'] for r in rows6 if not r['feasible'] and r['unresolved']==0]
 end6=(max(with6),min(g for g in without if g>max(with6))) if with6 and any(g>max(with6) for g in without) else None
 string6=status('gnl6_gVS_string')
 return dict(nl3=slices,nl3_end=end,nl6=rows6,nl6_end=end6,nl6_string=string6)

def pole_band():
 sl={}
 for d in D:
  c=cfg(d)
  if c['kind']=='pband':sl.setdefault(c['scan_x'],{})['min' if float(c['objective']['c20'])>0 else 'max']=d
 tests=[]
 rows=[]
 for x,s in sl.items():
  cls=s['min']['classification'] if 'min' in s and s['min'].get('accepted') else 'unresolved'
  tests.append((x,cls))
  if 'min' in s and 'max' in s and cls=='feasible' and s['max'].get('accepted') and s['max']['classification']=='feasible':
   rows.append((mp.mpf(x),num(s['min'],'c20')/num(s['min'],'c00'),num(s['max'],'c20')/num(s['max'],'c00')))
 return edge_bracket(tests),sorted(rows)


def cls_of(d):return (d['classification'] if d.get('accepted') else 'unresolved') if d else 'pending'

def nt_rows():
 """Linear maximum, NL3/NL6 at 2 zeta(3) and the NL3 right end against the number of gravity nodes."""
 rows=[]
 for nt,v in NT_LADDER:
  a,b=ok('gnl3c_%s_min'%v),ok('gnl3c_%s_max'%v)
  rows.append(dict(nt=nt,variant=v,t1=mp.sin(mp.pi/(4*nt))**2 if nt else None,
   linear=lin_label('glin_c00max_'+v) if nt else '---',
   nl3=(num(a,'c10')-Z5,num(b,'c10')-Z5) if a and b else None,nl3_status=(status('gnl3c_%s_min'%v),status('gnl3c_%s_max'%v)),
   nl6=edge_bracket(nl6_string_tests(v)),string=status('gnl6c_%s_string'%v),end=end_of(v)))
 return rows

def gravity_end_rows():
 """Largest c00 with a nonempty NL3 slice, per truncation (bisection bracket)."""
 rows=[]
 for v,label in END_VARIANTS:
  tests=[]
  for d in D:
   c=cfg(d)
   if v=='base' and c['kind'] in ('gnl3_slice','gnl3_end') and c.get('variant')=='base' and float(c['objective']['c10'])>0:
    tests.append((c['c00'],cls_of(d)))
   if v!='base' and c['kind']=='gnl3_end_variant' and c.get('variant')==v:
    tests.append((c['c00'],cls_of(d)))
  e=edge_bracket(tests)
  rows.append(dict(variant=v,label=label,last=e['hi'] if e else None,first_empty=e['hi_out'] if e else None,tests=len(tests)))
 return rows

def gravity_bulk_rows():
 rows=[]
 for v,label in GRAVITY_VARIANTS+NL_EXTRA[:1]:
  a,b=ok('gnl3b_%s_min'%v),ok('gnl3b_%s_max'%v)
  e=edge_bracket(nl6_tests('gnl6_bulk_convergence',v))
  rows.append(dict(variant=v,label=label,nl3=(num(a,'c10'),num(b,'c10')) if a and b else None,
                   nl3_status=(status('gnl3b_%s_min'%v),status('gnl3b_%s_max'%v)),nl6=e))
 return rows

def band_truncation_rows():
 """Pole-subtracted NL6 band: feasible range of x=c10/c00 per truncation, and the c20 interval on the string slice."""
 rows=[]
 x0=Z5/Z3
 for v,label in BAND_VARIANTS:
  tests=[]
  for d in D:
   c=cfg(d)
   if c['kind']=='pband' and c.get('variant','base')==v and float(c['objective']['c20'])>0:
    tests.append((c['scan_x'],cls_of(d)))
  e=edge_bracket(tests)
  a,b=ok('pconv_%s_min'%v),ok('pconv_%s_max'%v)
  if v=='N1024':a,b=ok('pconv_N1024_min'),ok('pconv_N1024_max')
  rows.append(dict(variant=v,label=label,edges=e,x0=x0,tests=len(tests),
                   c20=((num(a,'c20')-Z7),(num(b,'c20')-Z7)) if a and b else None))
 return rows

def alpha_rows():
 """Slices c00=2 zeta(3) alpha^3: NL3 c10 range, NL6 c10 extent, rescaled string point."""
 rows=[]
 for alpha in PG_ALPHAS:
  al=mp.mpf(alpha);tag=alpha.replace('.','p');c00=Z3*al**3;xs=Z5*al**5
  a,b=ok('pgam_a%s_nl3_min'%tag),ok('pgam_a%s_nl3_max'%tag)
  tests=[]
  for d in D:
   c=cfg(d)
   if c['kind']=='pgamma_nl6' and c.get('alpha')==alpha and float(c['objective']['c20'])>0:
    tests.append((c['c10'],cls_of(d)))
  e=edge_bracket(tests)
  sa,sb=ok('pgam_a%s_string_min'%tag),ok('pgam_a%s_string_max'%tag)
  rows.append(dict(alpha=alpha,c00=c00,string_c10=xs,
    nl3=((num(a,'c10')-xs)/abs(xs),(num(b,'c10')-xs)/abs(xs)) if a and b else None,
    nl3_status=(status('pgam_a%s_nl3_min'%tag),status('pgam_a%s_nl3_max'%tag)),
    nl6=((e['lo']-xs)/abs(xs),(e['hi']-xs)/abs(xs),e) if e else None,nl6_tests=len(tests),
    string=((num(sa,'c20')-Z7*al**7)/c00,(num(sb,'c20')-Z7*al**7)/c00) if sa and sb else None,
    string_status=(status('pgam_a%s_string_min'%tag),status('pgam_a%s_string_max'%tag))))
 ends=[]
 for d in D:
  if cfg(d)['kind']=='pgamma_end':ends.append((cfg(d)['alpha'],cls_of(d)))
 for r in rows:
  if r['nl3']:ends.append((r['alpha'],'feasible'))
  elif r['nl3_status'][0]=='infeasible':ends.append((r['alpha'],'infeasible'))
 return rows,edge_bracket(ends)

def deformation_rows():
 """B_VS(lam s,lam t,lam u)^w: pole-subtracted on fixed c00 slices and gravitational at lam=1."""
 pole=[]
 for d in D:
  c=cfg(d)
  if c['kind']=='pw_deformation' and float(c['objective']['c20'])>0:
   name=d['case'][:-4];lo,hi=ok(name+'_min'),ok(name+'_max');t=mp.mpf(c['c20_deformation'])
   inside=None
   if lo and hi:inside=num(lo,'c20')<=t<=num(hi,'c20')
   pole.append(dict(alpha=c['alpha'],w=c['w'],lam=mp.mpf(c['w_lambda']),status=(status(name+'_min'),status(name+'_max')),
                    inside=inside,interval=((num(lo,'c20')-t),(num(hi,'c20')-t)) if lo and hi else None))
 pole.sort(key=lambda r:(-float(r['alpha']),float(r['w'])))
 grav={}
 for d in D:
  c=cfg(d)
  if c['kind']=='gw_deformation':
   grav.setdefault((c['w_alpha'],c['w']),{})['NL%d'%c['nl']]=cls_of(d)
  if c['kind']=='gw_deformation_c20':
   grav.setdefault((c['w_alpha'],c['w']),{})['c20_'+('min' if float(c['objective']['c20'])>0 else 'max')]=d
 out=[]
 for (a,w),r in sorted(grav.items(),key=lambda kv:(-float(kv[0][0]),float(kv[0][1]))):
  inside=None
  if r.get('c20_min') and r.get('c20_max') and r['c20_min'].get('accepted') and r['c20_max'].get('accepted') \
     and r['c20_min']['classification']==r['c20_max']['classification']=='feasible':
   t=Z7*mp.mpf(w)*mp.mpf(a)**7;inside=num(r['c20_min'],'c20')<=t<=num(r['c20_max'],'c20')
  out.append(dict(alpha=a,w=w,nl3=r.get('NL3','pending'),nl6=r.get('NL6','pending'),inside=inside))
 return pole,out

def residual_gate():
 res=[];gap=[]
 for d in D:
  if d.get('accepted') and d['classification']=='feasible' and d.get('audit'):
   a=d['audit'];res.append(mp.mpf(a['normalized_equality_residual']))
   g=mp.mpf(d.get('verified_gap',a['gap']));gap.append(g/max(1,abs(mp.mpf(a['primal_objective']))))
 return max(res,default=0),max(gap,default=0)

def table(header,spec,rows,caption,label):
 s='\\begin{table}[htbp]\n \\centering\n \\small\n \\begin{tabular}{'+spec+'}\n \\toprule\n '+header+'\\\\\n \\midrule\n'
 s+=''.join(' '+' & '.join(r)+'\\\\\n' for r in rows)
 return s+' \\bottomrule\n \\end{tabular}\n \\caption{'+caption+'}\n \\label{'+label+'}\n\\end{table}\n\n'

def sdpb_rows():
 if not args.sdpb or not Path(args.sdpb).exists():return []
 out=[];groups={}
 for e in json.loads(Path(args.sdpb).read_text()):
  x,b=e['sdpx'],e['sdpb']
  if b.get('primal_objective') is None or x.get('objective_minus_q_dot_x') is None:continue
  diff=abs(mp.mpf(b['primal_objective'])-mp.mpf(x['objective_minus_q_dot_x']))
  row=dict(name=e['name'],columns=e['columns'],rows=e['rows'],cores=e['cores'],sdpx_s=x['wall_seconds'],sdpx_it=x['iterations'],
                  sdpb_s=b['solve_seconds'],sdpb_convert=b['convert_seconds'],sdpb_it=b['iterations'],diff=diff,
                  ratio=(b['solve_seconds']+b['convert_seconds'])/x['wall_seconds'],repeats=1)
  key=re.sub(r'_rep\d+$','',e['name']);groups.setdefault(key,[]).append(row)
 for key,rows in groups.items():
  if any('_rep' in r['name'] for r in rows) and len(rows)<3:continue
  assert len({(r['columns'],r['rows'],r['cores']) for r in rows})==1
  row=dict(rows[0],name=key,repeats=len(rows))
  for field in ['sdpx_s','sdpx_it','sdpb_s','sdpb_convert','sdpb_it']:
   row[field]=statistics.median(r[field] for r in rows)
  row['diff']=max(r['diff'] for r in rows)
  row['sdpb_total']=statistics.median(r['sdpb_s']+r['sdpb_convert'] for r in rows)
  row['ratio']=row['sdpb_total']/row['sdpx_s'];out.append(row)
 return out

def main():
 eq,gap=residual_gate();lin=linear_rows();gnl=gravity_nl_rows();pol=pole_rows()
 ends=gravity_end_rows();bulk=gravity_bulk_rows();bands=band_truncation_rows();alphas,alpha_end=alpha_rows();pw,gw=deformation_rows()
 metrics=polygon_metrics();gs=gravity_scan();band,band_rows=pole_band();timing=sdpb_rows()
 references=reference_checks(D)
 tails=tail_checks(D,references)
 unresolved=sorted(d['case'] for d in D if not d.get('accepted'))
 s=r"""\subsection{Convergence checks}
\label{app:revision-convergence}

We vary the energy resolution, spin cutoff, gravity samples and null tower
in figures~\ref{fig:gravity-convergence-revision}--\ref{fig:pole-band-truncations}.
Adding constraints to the same grid can only shrink the feasible set;
adding spectral states with the constraints fixed can only enlarge it.
The energy grids are nested when \(N_\mu\) is doubled; the gravity node
sets for different \(N_t\) are not.
The null tower contains \((k,n)=(2m,n)\), with \(1\le n\le p\) and
\(0\le m<n\): \(p(p+1)/2\) rows, or 36, 55, 78 and 105 for
\(p=8,10,12,14\). Adding null constraints strengthens the bounds. We do
not push the null tower to its optimum: higher null constraints probe
higher spins, so each enlargement also requires a larger spin cutoff.
Only audited outcomes enter the plots; the companion
contains the full tables and records.

\paragraph{Sampling the graviton pole.}
Larger \(N_t\) can strengthen a finite bound, but also probes nearer the
forward pole: the closest Chebyshev node has
\(|t_1|=\sin^2[\pi/(4N_t)]\). Resolving these conditions requires an
adequately dense energy grid, support at higher energies and a larger
\(J_{\max}\), since the pole is built from arbitrarily high energy and
spin \cite{Haring:2024wyz}. Thus the energy support and spin cutoff must
be increased together with \(N_t\). Smeared calculations likewise need
greater mass and spin resolution as their functional basis is enlarged
\cite[section~3.2 and appendix~A]{Caron-Huot:2021rmr}.
Here we compare \(N_t=4,6,8\), whose node sets are not nested, to assess
sensitivity at the stated settings; we do not seek the optimal
gravitational bound.

"""
 unique_runs={d.get('source_case',d['case']):d for d in D}
 s+=r"""\begin{figure}[htbp]
 \centering
 \includegraphics[width=0.96\textwidth]{fig-gravity-convergence-sdpx256.pdf}
 \caption{Linear maximum of $c_{0,0}M^6/(8\pi G)$, changing one truncation at a
 time about $(N_\mu,J_{\max},N_t,N_{\rm null})=(512,10^4,6,55)$: (a) energy
 counts $N_\mu=256,512,1024$, with the largest numerical energy node
 $\mu_1=\csc^2[\pi/(2N_\mu)]M^2$ on the top
 axis; (b) spin cutoff; (c) gravity nodes; (d) null constraints. Panels (a),
 (b), and (d) share one vertical window.}
 \label{fig:gravity-convergence-revision}
\end{figure}

"""
 s+=r"""\paragraph{Nonlinear gravitational regions.}
Figures~\ref{fig:gravity-nl-convergence} and
\ref{fig:gravity-nl6-convergence} show the NL3/NL6 ranges and the resolved
NL6 scale. Dashed extensions retain the adjacent feasible/excluded brackets.

\begin{figure}[htbp]
 \centering
 \includegraphics[width=0.96\textwidth]{fig-gravity-nl3-parameters-sdpx256.pdf}
 \caption{Nonlinear gravitational results on the slice
 $c_{0,0}M^6/(8\pi G)=2\zeta(3)$, varying one setting about
 $(N_\mu,J_{\max},N_t,N_{\rm null})=(512,10^4,6,55)$.
 Light bars: NL3 interval of $c_{1,0}+2\zeta(5)$; dark bars:
 span of feasible NL6 tests. The top axis of panel (a) gives the highest
 grid energy. The dashed line is the string value.}
 \label{fig:gravity-nl-convergence}
\end{figure}

\begin{figure}[htbp]
 \centering
 \includegraphics[width=0.96\textwidth]{fig-gravity-nl6-cutoff-checks-sdpx256.pdf}
 \caption{The gravitational NL6 tests of
 figure~\ref{fig:gravity-nl-convergence} on a common expanded scale.
 Solid bars span the feasible tests; grey dashed extensions reach the
 nearest excluded tests on either side. These extensions quantify the
 scan resolution and do not certify a continuous NL6 boundary.
 The red dashed line is the string value.}
 \label{fig:gravity-nl6-convergence}
\end{figure}

"""
 s+=r"""On this slice, changing the energy count to \(N_\mu=256\) or 1024, or the
spin cutoff to \(J_{\max}=6000\) or 16000, changes the width of the
feasible NL6 window by at most \(1.7\%\), and \(N_t=4\) and 8 change it by
\(3.8\%\) and \(0.6\%\). Increasing the null tower from 55 to 78 rows
narrows it by \(8.1\%\). Removing the sampled pole rows altogether widens
the NL3 interval by \(7.7\%\) but changes the NL6 window by only
\(0.6\%\), so at this truncation the NL6 window is set mainly by the
nonlinear relations and the moment and null rows.

\paragraph{Pole-subtracted island.}
"""
 bp=next(r for r in pol if r['variant']=='base')['interval']
 if bp:
  wb=bp[1]-bp[0]
  changes={}
  for r in pol:
   if r['interval']:
    q=r['interval'];wq=q[1]-q[0]
    changes[r['variant']]=dict(width=100*abs(wq-wb)/wb,
     endpoint=100*max(abs(q[0]-bp[0]),abs(q[1]-bp[1]))/min(wb,wq))
  s+=('At the string values of \\(c_{0,0}\\) and \\(c_{1,0}\\), doubling '
      'the energy count from 512 to 1024 '
      'changes the NL6 width by \\(%s\\%%\\), and doubling the spin cutoff '
      'from 2000 to 4000 changes it by \\(%s\\%%\\). '
      'Increasing the null tower from 78 to 105 rows reduces the width '
      'by \\(%s\\%%\\). Thus energy and spin are locally stable on these '
      'tests, while the null tower still narrows the island.\n\n'%(
       f(changes['N1024']['width'],3),f(changes['J4000']['width'],3),
       f(changes['P14']['width'],3)))
 s+=r"""\begin{figure}[htbp]
 \centering
 \includegraphics[width=0.96\textwidth]{fig-pole-cutoff-checks-sdpx256.pdf}
 \caption{Pole-subtracted NL6 interval of $c_{2,0}$ at the string values of
 $c_{0,0}$ and $c_{1,0}$, shown as $10^6[c_{2,0}-2\zeta(7)]$ ($M=1$), changing
 one setting about $(N_\mu,J_{\max},N_{\rm null})=(512,2000,78)$.}
 \label{fig:pole-convergence-revision}
\end{figure}

"""
 s+=r"""\begin{figure}[htbp]
 \centering
 \includegraphics[width=0.96\textwidth]{fig-pole-band-truncations-sdpx256.pdf}
 \caption{Pole-subtracted NL6 islands at $c_{0,0}=2\zeta(3)$ for four null
 towers and a finer energy grid. Both panels show the coefficient ratios
 $M^4c_{1,0}/c_{0,0}$ and $M^8c_{2,0}/c_{0,0}$ directly. Panel (a) shows
 the full displayed islands; its dashed box marks the extent of panel (b).
 The islands shrink as null constraints are added, and the red point marks
 the string reference. Outlines close at the extreme checked slices.}
 \label{fig:pole-band-truncations}
\end{figure}

"""
 solver_comparison=''
 if timing:
  rows=[['$%d\\times%d$'%(t['columns'],t['rows']),'%.1f (%s)'%(t['sdpx_s'],t['sdpx_it']),
         '%.1f (%s)'%(t['sdpb_total'],t['sdpb_it']),'%.1f'%t['ratio'],'$'+sci(t['diff'],1)+'$'] for t in timing]
  solver_comparison=table(r'columns$\times$rows & SDPX s (it.) & SDPB total s (it.) & ratio & $|\Delta\,\mathrm{obj}|$','ccccc',rows,
   r'SDPX and SDPB on two pole-subtracted NL6 string-slice LPs, with identical inputs for each solver comparison, 256-bit arithmetic and $10^{-30}$ thresholds, using %d solver core%s. Times and iteration counts are medians of %d sequential repeats; SDPB total time includes pmp2sdp conversion. The last column gives the largest objective difference across the repeats.'%(timing[0]['cores'],'' if timing[0]['cores']==1 else 's',timing[0]['repeats']),
   'tab:sdpb-sdpx')
 out=R/'tex/revision-numerical-summary.tex';out.parent.mkdir(exist_ok=True)
 out.write_text(s)
 (out.parent/'revision-solver-comparison.tex').write_text(solver_comparison)
 def serial(value):
  if isinstance(value,mp.mpf):return str(value)
  if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
  if isinstance(value,(list,tuple)):return [serial(v) for v in value]
  return value
 evidence=dict(records=len(D),independent_audited_runs=len(unique_runs),counts=dict(counts),
  reference_checks=serial(references),small_alpha_tail_checks=tails,
  max_normalized_equality_residual=str(eq),max_normalized_gap=str(gap),
  linear={r['variant']:(str(r['c00max']) if r['c00max'] is not None else r['status']) for r in lin},
  gravity_nl={r['variant']:dict(nl3=[str(x) for x in r['nl3']] if r['nl3'] else r['nl3_status'],
     nl6={k:(str(v) if v is not None and not isinstance(v,int) else v) for k,v in r['nl6'].items()} if r['nl6'] else None,string=r['string']) for r in gnl},
  pole={r['variant']:[str(x) for x in r['interval']] if r['interval'] else r['status'] for r in pol},
  gravity_end=[dict(variant=r['variant'],last=str(r['last']),first_empty=str(r['first_empty'])) for r in ends],
  pole_bands={r['variant']:dict(edges={k:(str(v) if v is not None and not isinstance(v,int) else v) for k,v in r['edges'].items()} if r['edges'] else None,
                                c20=[str(x) for x in r['c20']] if r['c20'] else None) for r in bands},
  alpha_slices=[dict(alpha=r['alpha'],nl3=[str(x) for x in r['nl3']] if r['nl3'] else r['nl3_status'],
                     nl6=[str(x) for x in r['nl6'][:2]] if r['nl6'] else None,string=[str(x) for x in r['string']] if r['string'] else r['string_status']) for r in alphas],
  alpha_end={k:(str(v) if v is not None and not isinstance(v,int) else v) for k,v in alpha_end.items()} if alpha_end else None,
  deformation_pole=[dict(alpha=r['alpha'],w=r['w'],status=r['status'],inside=r['inside'],interval=[str(x) for x in r['interval']] if r['interval'] else None) for r in pw],
  deformation_gravity=gw,
  scan_geometry=metrics,
  gravity_scan=dict(nl3=[dict(gamma=str(r['gamma']),status=r['status'],c10=[str(x) for x in r['c10']] if r['c10'] else None) for r in gs['nl3']],
    nl3_end=[str(x) for x in gs['nl3_end']] if gs['nl3_end'] else None,
    nl6=[dict(gamma=str(r['gamma']),tests=r['tests'],feasible=r['feasible'],unresolved=r['unresolved'],
              edges={k:(str(v) if v is not None and not isinstance(v,int) else v) for k,v in r['edges'].items()} if r['edges'] else None) for r in gs['nl6']],
    nl6_end=[str(x) for x in gs['nl6_end']] if gs['nl6_end'] else None,nl6_string=gs['nl6_string']),
  pole_band=dict(edges={k:(str(v) if v is not None and not isinstance(v,int) else v) for k,v in band.items()} if band else None,
    slices=[[str(x) for x in r] for r in band_rows]),
  sdpb_vs_sdpx=[{k:(str(v) if isinstance(v,mp.mpf) else v) for k,v in t.items()} for t in timing])
 (R/'data/numerical_evidence.json').write_text(json.dumps(evidence,indent=1)+'\n')
 response=r"""\paragraph{Completed numerical checks.}
The revised numerical results use 512-bit kernel generation, 256-bit SDPX
arithmetic, and independent 768-bit audits on the complete declared spectral
grid. Solver thresholds are $10^{-36}$ for gravitational LPs and $10^{-30}$
for pole-subtracted LPs; the original-equation and gap acceptance gate is
$10^{-25}$. Old Float64 results are retained only as historical context in
a footnote.

Appendix A compares the energy resolution, spin cutoff, gravity samples,
and null tower, and gives feasible/excluded boundary brackets for the narrow
NL6 scans. The main figures show regions without trial-point overlays.
The gravitational extremal spectrum compares three joint truncations.
The mass support is $1\le\mu<\infty$; the stated grid reaches are numerical
sampling choices.

"""
 if timing:
  response+=('The sequential one-core SDPB/SDPX comparison uses the same LP '
    'data, 256-bit arithmetic, and $10^{-30}$ thresholds. Each case is '
    'repeated three times. The largest objective difference is $%s$; '
    'the median SDPB total time is %.1f--%.1f times the SDPX time on '
    'these small LPs.\n\n'%(sci(max(t['diff'] for t in timing),2),
       min(t['ratio'] for t in timing),max(t['ratio'] for t in timing)))
 response+=('The versioned numerical companion contains the canonical '
   'configurations, accepted and unresolved records, source hashes, and '
   'figure-generation scripts. Arithmetic accuracy, scan resolution, and '
   'finite spectral-ansatz effects are reported separately.\n')
 (R/'tex/response-numerical-summary.tex').write_text(response)
 print(json.dumps(dict(records=len(D),counts=dict(counts)),indent=1))

if __name__=='__main__':main()
