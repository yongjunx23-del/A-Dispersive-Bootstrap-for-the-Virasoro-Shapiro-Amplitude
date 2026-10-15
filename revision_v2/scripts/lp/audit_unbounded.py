"""Check a positive primal recession ray and a feasible base point."""
import json
from bootstrap_sdpx_v5 import M,ZERO,ONE,dec

def terminal(root,case):
 folder=root/'results'/case
 p=folder/('continued_summary.json'if(folder/'continued_summary.json').exists()else'summary.json')
 if not p.exists():return None
 d=json.loads(p.read_text())
 if not d.get('rounds'):return None
 source=root/'results'/d['continued_directory']if d.get('continued_directory')else folder
 final=int(d.get('final_round',d['rounds'][-1]['round']))
 path=source/('round_%02d'%final)
 return d,json.loads((path/'point.json').read_text()),json.loads((path/'mapping.json').read_text())

def check(model,point,mapping,root):
 tolerance=M(model.cfg.get('audit_tol','1e-25'))
 rho=[M(z)/M(h)for z,h in zip(point['z'],mapping['scales'])]
 cost=sum((model.costval(i)*w for i,w in zip(mapping['indices'],rho)),ZERO)
 if cost>=0:return dict(passed=False,note='No negative objective recession direction')
 rho=[w/(-cost)for w in rho]
 sums=[ZERO]*len(model.rows)
 for i,w in zip(mapping['indices'],rho):
  for j,e in enumerate(model.column(i)):sums[j]+=e*w
 residual=max(map(abs,sums),default=ZERO)
 violation=max(ZERO,-min(rho,default=ZERO))
 result=dict(passed=False,precompute_bits=384,ray_cost='-1',ray_equations=[dec(v)for v in sums],
             maximum_ray_equality_residual=dec(residual),ray_nonnegativity_violation=dec(violation),
             ray_indices=mapping['indices'],ray_weights=[dec(w)for w in rho])
 if residual>tolerance or violation>tolerance:return result
 # The p=3 solve on exactly this grid supplies a feasible point for p=2.
 source=model.cfg.get('unbounded_base_case') or 'gr_g0_N%d_J%d_t%d_p3'%(model.nmu,model.cfg['j_max'],model.nt)
 terminal_data=terminal(root,source)
 if not terminal_data:return dict(result,note='No independent feasible base point available')
 d,feasible,mp=terminal_data;c=d['config']
 keys=['n_mu','j_max','n_t','gravity','nl','fix_c00','fix_c10','energy_grid','max_null_power','t_grid','column_scale']
 if feasible.get('status')!='Solved'or not all(c.get(k)==model.cfg.get(k)for k in keys):
  return dict(result,note='Base point has different physical settings')
 weights=[M(z)/M(h)for z,h in zip(feasible['z'],mp['scales'])]
 sums=[ZERO]*len(model.rows)
 for i,w in zip(mp['indices'],weights):
  for j,e in enumerate(model.column(i)):sums[j]+=e*w
 normalized=max((abs(v-b)/max(ONE,abs(b))for v,b in zip(sums,model.b)),default=ZERO)
 nonnegative=max(ZERO,-min(weights,default=ZERO))
 result.update(base_point_source=source,base_point_normalized_residual=dec(normalized),
               base_point_nonnegativity_violation=dec(nonnegative),
               passed=(normalized<=tolerance and nonnegative<=tolerance))
 return result
