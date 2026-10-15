#!/usr/bin/env python3
"""Publication figures from the 768-bit-audited SDPX results of the 2026-10-14 campaign.

Every plotted coordinate is an audited multiprecision value (or the exact fixed
coordinate of a feasibility test); binary64 is used only by matplotlib to draw.
"""
import argparse,json,math,os,logging,hashlib
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/virasoro-revision-matplotlib-20261013')
logging.getLogger('fontTools').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Rectangle
from scipy.spatial import ConvexHull
import mpmath
from reference_checks import reference_checks,tail_checks

R=Path(__file__).resolve().parents[1]
OUT=R/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'cm',
 'font.size':9,'axes.labelsize':10,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'axes.linewidth':.65,'lines.linewidth':1.05,'pdf.fonttype':42,
 'savefig.facecolor':'white','figure.facecolor':'white','axes.spines.top':True,'axes.spines.right':True})
COL={0:'#356b9c',3:'#c68b32',6:'#8e557e','string':'#ad3446'}
REGION={0:'#333333',3:'#777777',6:'#087f74','string':'#b82724'}
POLE_REGION={0:'#444444',3:'#168c7c',6:'#c56f09'}
mpmath.mp.dps=60
C0,C1,C2=(float(2*mpmath.zeta(3)),float(-2*mpmath.zeta(5)),float(2*mpmath.zeta(7)))
VS=(C1/C0,C2/C0)
parser=argparse.ArgumentParser()
parser.add_argument('--data',required=True,help='campaign_records.json written by collect_campaign.py')
parser.add_argument('--only',nargs='+',choices=['gravity-convergence','nl3-convergence','nl6-convergence',
                    'gravity-linear','gravity-nl3-region','gravity-nl6-region','pole-region','pole-band-truncations',
                    'pole-cutoff'],
                    help='Rebuild only selected figures, preserving all other figure bytes')
args=parser.parse_args()
DATA=json.loads(Path(args.data).read_text())
DATA=[d for d in DATA if not d['canonical_config'].get('t_grid')]
GOOD=[d for d in DATA if d.get('accepted') and d['classification']=='feasible' and 'audit' in d]
EXCLUDED=[d for d in DATA if d.get('accepted') and d['classification']=='infeasible']
MANIFEST={}

def cfg(d):return d['canonical_config']
def val(d,key):return float(d['audit']['wilson'][key])
def kind(items,k,variant='base'):return [d for d in items if cfg(d)['kind']==k and cfg(d).get('variant','base')==variant]
def prettify(ax):
 ax.grid(color='#dedede',lw=.45,alpha=.6);ax.set_axisbelow(True)
 for axis in ('x','y'):
  try:ax.ticklabel_format(axis=axis,style='plain',useOffset=False)
  except AttributeError:pass       # categorical tick labels
 ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True,length=3,width=.6)

def export(fig,name,sources):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.04)
 plt.close(fig);MANIFEST[name]=sorted({d['case'] for d in sources})

def points(items,ratio=False):
 return np.array([(val(d,'c10')/val(d,'c00'),val(d,'c20')/val(d,'c00')) if ratio else
                  (val(d,'c00'),val(d,'c10')) for d in items])

def polygon(ax,items,nl,ratio=False,alpha=.16,color=None):
 if len(items)<3:return
 pts=np.unique(points(items,ratio),axis=0)
 if len(pts)<3:return
 order=ConvexHull(pts).vertices;color=color or REGION[nl]
 ax.fill(pts[order,0],pts[order,1],facecolor=color,alpha=alpha,edgecolor='none',zorder=2)
 ax.plot(np.r_[pts[order,0],pts[order[0],0]],np.r_[pts[order,1],pts[order[0],1]],color=color,lw=1.1,zorder=3)

def gravity_reference(ax):
 alpha=np.linspace(.01,1,500);ax.plot(C0*alpha**3,C1*alpha**5,'--',color=REGION['string'],lw=1.2,zorder=4)

# The Virasoro-Shapiro reference point (alpha=1) is marked identically in every region plot.
VS_KW=dict(marker='o',s=14,color=REGION['string'],edgecolors='white',linewidths=.35,zorder=10)
def vs_marker(ax,xy):ax.scatter([xy[0]],[xy[1]],**VS_KW)
def vs_handle():return Line2D([],[],marker='o',ls='none',ms=3.8,color=REGION['string'],markeredgecolor='white',markeredgewidth=.35)

def zoom_guide(ax,limits,label,offset=(12,14)):
 """Locate a detail panel in its parent's actual plotting coordinates."""
 xlo,xhi,ylo,yhi=limits
 ax.add_patch(Rectangle((xlo,ylo),xhi-xlo,yhi-ylo,fill=False,edgecolor='#555555',
                        lw=.75,ls=(0,(3,2)),zorder=8))
 ax.annotate(label,xy=(xhi,yhi),xytext=offset,textcoords='offset points',
             fontsize=8,color='#555555',arrowprops=dict(arrowstyle='-',lw=.65,color='#555555'),
             bbox=dict(facecolor='white',edgecolor='none',pad=.5),zorder=9)

def direct_axes(ax,origin=(0.,0.),scale=1.):
 """Print coefficient values as decimals, with no offset or power-of-ten factor.

 Figure11 retains its internal drawing transform; only its labels invert it.
 """
 for axis,limits,center in [(ax.xaxis,ax.get_xlim(),origin[0]),(ax.yaxis,ax.get_ylim(),origin[1])]:
  lo,hi=(center+v/scale for v in limits)
  for bins in (3,4,6):
   ticks=matplotlib.ticker.MaxNLocator(nbins=bins,steps=[1,2,2.5,5,10]).tick_values(lo,hi)
   ticks=ticks[(ticks>=lo-1e-12)&(ticks<=hi+1e-12)]
   if len(ticks)>=3:break
  step=min(np.diff(ticks)) if len(ticks)>1 else hi-lo
  digits=next((n for n in range(11) if np.all(np.abs(np.round(ticks,n)-ticks)<step*1e-6)),10)
  axis.set_major_locator(matplotlib.ticker.FixedLocator((ticks-center)*scale))
  def fmt(v,pos,c=center,n=digits):return format(c+v/scale,'.%df'%n)
  axis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))

def sample_handle(color,marker='o',ms=3):return Line2D([],[],marker=marker,ls='none',ms=ms,color=color)
def bracket(ax,a,b):ax.axvspan(a,b,color='#9a9a9a',alpha=.35,lw=0,zorder=1)
def bracket_handle():return Patch(facecolor='#9a9a9a',alpha=.35,edgecolor='none')

def slice_pairs(items,coord='c00',objective='c10'):
 out={}
 for d in items:
  key=float(cfg(d)[coord]);sign=float(cfg(d)['objective'][objective])
  out.setdefault(key,{})['min' if sign>0 else 'max']=d
 return out

def sampled_band(ax,x,lo,hi,color,alpha=.2,lw=1,zorder=3,fill=True,ls='-'):
 """Outline one displayed window, closing only at its existing sampled ends.

 Callers retain all excluded/unresolved splits. The vertical end segments
 delimit the displayed slice span; they do not locate an unsampled boundary.
 """
 x,lo,hi=(np.asarray(v,dtype=float) for v in (x,lo,hi))
 if len(x)<2:return
 if (x.ndim!=1 or lo.shape!=x.shape or hi.shape!=x.shape
     or not all(np.isfinite(v).all() for v in (x,lo,hi))
     or np.any(np.diff(x)<=0) or np.any(lo>hi)):
  raise ValueError('A sampled band needs finite, ordered slice intervals')
 if fill:
  ax.fill_between(x,lo,hi,color=color,alpha=alpha,edgecolor='none',linewidth=0,zorder=zorder)
 # Traverse the same lower/upper curves and add the two end segments.
 # No coordinate is extrapolated and no distinct window is connected.
 return ax.plot(np.r_[x,x[::-1],x[0]],np.r_[lo,hi[::-1],lo[0]],
                color=color,lw=lw,ls=ls,zorder=zorder+1)[0]

def nl3_slices():
 """Feasible (gamma, min, max) and the bracket [last feasible, first empty] of the right end."""
 feasible=slice_pairs(kind(GOOD,'gnl3_slice')+kind(GOOD,'gnl3_end'))
 pairs=[(x,val(s['min'],'c10'),val(s['max'],'c10')) for x,s in sorted(feasible.items()) if 'min' in s and 'max' in s]
 empty=sorted({float(cfg(d)['c00']) for d in kind(EXCLUDED,'gnl3_slice')+kind(EXCLUDED,'gnl3_end')})
 end=None
 if pairs and empty:
  last=max(p[0] for p in pairs);later=[g for g in empty if g>last]
  if later:end=(last,min(later))
 return pairs,end

def nl6_tests(variant='base',k='gnl6_scan'):
 ok=[(float(cfg(d)['c00']),float(cfg(d)['c10'])) for d in kind(GOOD,k,variant)]
 bad=[(float(cfg(d)['c00']),float(cfg(d)['c10'])) for d in kind(EXCLUDED,k,variant)]
 return ok,bad

def nl6_string_records(variant='base'):
 """All accepted tests of the same finite problem on the string slice."""
 records=kind(GOOD,'gnl6_convergence',variant)+kind(EXCLUDED,'gnl6_convergence',variant)
 if variant=='base':
  records += [d for d in DATA if cfg(d)['kind'] in ('gnl6_scan','gnl6_tip')
              and cfg(d).get('variant')=='base' and d.get('accepted')
              and abs(mpmath.mpf(cfg(d)['c00'])-2*mpmath.zeta(3))<mpmath.mpf('1e-30')]
 return records

def nl6_windows():
 """Keep separate feasible runs; never shade across an excluded or unresolved test."""
 grouped={}
 for d in DATA:
  c=cfg(d)
  if c['kind'] not in ('gnl6_scan','gnl6_tip') or c.get('variant')!='base':continue
  gx=mpmath.nstr(mpmath.mpf(c['c00']),45)
  cx=mpmath.nstr(mpmath.mpf(c['c10']),45)
  cls=d['classification'] if d.get('accepted') else 'unresolved'
  previous=grouped.setdefault(gx,{}).get(cx)
  if previous in ('feasible','infeasible') and cls in ('feasible','infeasible') and cls!=previous:
   raise ValueError('Contradictory accepted NL6 classifications at '+gx+', '+cx)
  if previous is None or cls in ('feasible','infeasible'):grouped[gx][cx]=cls
 slices=[]
 for gx,tests in sorted(grouped.items(),key=lambda q:mpmath.mpf(q[0])):
  runs=[];run=[]
  for cx,cls in sorted(tests.items(),key=lambda q:mpmath.mpf(q[0])):
   if cls=='feasible':run.append(float(cx))
   elif run:runs.append((min(run),max(run)));run=[]
  if run:runs.append((min(run),max(run)))
  slices.append((float(gx),runs))
 return slices

def draw_nl6_windows(ax,slices):
 groups=[];group=[]
 for x,runs in slices:
  if not runs or (group and len(runs)!=len(group[-1][1])):
   if group:groups.append(group)
   group=[]
  if runs:group.append((x,runs))
 if group:groups.append(group)
 for group in groups:
  if len(group)<2:continue
  xs=[x for x,_ in group]
  for i in range(len(group[0][1])):
   lo=[runs[i][0] for _,runs in group];hi=[runs[i][1] for _,runs in group]
   sampled_band(ax,xs,lo,hi,REGION[6],alpha=.3,zorder=5)

def gravity_linear():
 """Allowed region of the linear problem in the style of the original figure: region, string family, alpha=1."""
 items=kind(GOOD,'glin_direction')
 if len(items)<3:return
 fig,ax=plt.subplots(figsize=(4.9,3.9));fig.subplots_adjust(top=.97,bottom=.15,left=.17,right=.97)
 pts=np.unique(points(items),axis=0);order=ConvexHull(pts).vertices
 ax.fill(pts[order,0],pts[order,1],facecolor='#e0e0e0',edgecolor='none',zorder=1)
 ax.plot(np.r_[pts[order,0],pts[order[0],0]],np.r_[pts[order,1],pts[order[0],1]],color='#222222',lw=1.3,zorder=2)
 alpha=np.linspace(0,1,500);ax.plot(C0*alpha**3,C1*alpha**5,'--',color='#b22222',lw=1.3,zorder=3)
 vs_marker(ax,(C0,C1))
 ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True)
 ax.set(xlabel=r'$c_{0,0}M^6/(8\pi G)$',ylabel=r'$c_{1,0}M^{10}/(8\pi G)$')
 ax.legend([Patch(facecolor='#e0e0e0',edgecolor='none'),Line2D([],[],color='#b22222',ls='--',lw=1.3),
            vs_handle()],
           ['allowed','Virasoro-Shapiro',r'$\alpha=1$'],loc='upper right',frameon=False)
 export(fig,'fig-gpole-region-sdpx256',items)

def zoom_window(pairs,end,x0=None):
 x0=C0-.0012 if x0 is None else x0
 x1=(end[0] if end else max(p[0] for p in pairs))+.00012
 near=[p for p in pairs if x0-1e-9<=p[0]<=x1]
 if not near:near=[min(pairs,key=lambda p:abs(p[0]-C0))]
 ys=[p[1] for p in near]+[p[2] for p in near]+[C1]
 pad=.12*(max(ys)-min(ys))
 return (x0,x1),(min(ys)-pad,max(ys)+pad)

def gravity_nl3():
 pairs,end=nl3_slices()
 if not pairs:return
 items=kind(GOOD,'gnl3_slice')+kind(GOOD,'gnl3_end')+kind(EXCLUDED,'gnl3_slice')+kind(EXCLUDED,'gnl3_end')
 xlim,ylim=zoom_window(pairs,end)
 for zoom,name in [(False,'fig-nl3-region-sdpx256'),(True,'fig-nl3-region-zoom-sdpx256')]:
  fig,ax=plt.subplots(figsize=(3.75,3.35));fig.subplots_adjust(top=.85,bottom=.18,left=.20,right=.97)
  x=np.array([p[0] for p in pairs]);lo=np.array([p[1] for p in pairs]);hi=np.array([p[2] for p in pairs])
  sampled_band(ax,x,lo,hi,REGION[3],alpha=.19,zorder=2)
  gravity_reference(ax);vs_marker(ax,(C0,C1));prettify(ax);ax.grid(False)
  ax.set(xlabel=r'$c_{0,0}M^6/(8\pi G)$',ylabel=r'$c_{1,0}M^{10}/(8\pi G)$')
  if zoom:
   ax.set(xlim=xlim,ylim=ylim)
   direct_axes(ax)
  else:
   ax.set(xlim=(0,2.45),ylim=(-2.3,.02))
  handles=[Patch(facecolor=REGION[3],alpha=.25),Line2D([],[],color=REGION['string'],ls='--'),vs_handle()]
  labels=['NL3','Virasoro-Shapiro',r'$\alpha=1$']
  fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=8,columnspacing=1.0,handlelength=1.6)
  export(fig,name,items)

def gravity_nl6():
 pairs,end=nl3_slices();ok,bad=nl6_tests();tok,tbad=nl6_tests(k='gnl6_tip')
 if not pairs:return
 # The finer scans near the tip of the NL6 window are part of the same feasibility map.
 ok,bad=ok+tok,bad+tbad
 items=(kind(GOOD,'gnl3_slice')+kind(GOOD,'gnl3_end')+kind(GOOD,'gnl6_scan')+kind(EXCLUDED,'gnl6_scan')
        +kind(GOOD,'gnl6_tip','base')+kind(EXCLUDED,'gnl6_tip','base'))
 windows=nl6_windows()
 near=[(x,a,b) for x,runs in windows for a,b in runs if C0-.00065<=x<=C0+.00035]
 if near:
  yvals=[v for _,a,b in near for v in (a,b)]+[C1]
  pad=.14*(max(yvals)-min(yvals))
  xlim=(C0-.00065,min(C0+.00035,max(x for x,_,_ in near)+.000075))
  ylim=(min(yvals)-pad,max(yvals)+pad)
 else:xlim,ylim=zoom_window(pairs,end)
 for zoom,name in [(False,'fig-nl3-nl6-region-sdpx256'),(True,'fig-nl3-nl6-zoom-sdpx256')]:
  fig,ax=plt.subplots(figsize=(3.75,3.35));fig.subplots_adjust(top=.83,bottom=.18,left=.20,right=.97)
  sampled_band(ax,[p[0] for p in pairs],[p[1] for p in pairs],[p[2] for p in pairs],
               REGION[3],alpha=.19,lw=.9)
  draw_nl6_windows(ax,windows)
  gravity_reference(ax);vs_marker(ax,(C0,C1));prettify(ax);ax.grid(False)
  ax.set(xlabel=r'$c_{0,0}M^6/(8\pi G)$',ylabel=r'$c_{1,0}M^{10}/(8\pi G)$')
  if zoom:
   ax.set(xlim=xlim,ylim=ylim)
   direct_axes(ax)
  else:
   ax.set(xlim=(0,2.45),ylim=(-2.3,.02))
  handles=[Patch(facecolor=REGION[3],alpha=.25),Patch(facecolor=REGION[6],alpha=.4),
           Line2D([],[],color=REGION['string'],ls='--'),vs_handle()]
  labels=['NL3','NL6','Virasoro-Shapiro',r'$\alpha=1$']
  fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,columnspacing=1.7,handlelength=1.6)
  export(fig,name,items)

def box(pts,pad=.12,extra=()):
 pts=np.array(list(pts)+list(extra))
 lo,hi=pts.min(axis=0),pts.max(axis=0);span=np.maximum(hi-lo,1e-12)
 return (lo[0]-pad*span[0],hi[0]+pad*span[0],lo[1]-pad*span[1],hi[1]+pad*span[1])

def pole_band():
 slices=slice_pairs(kind(GOOD,'pband'),'c10','c20')
 return [(float(cfg(s['min'])['scan_x']),val(s['min'],'c20')/C0,val(s['max'],'c20')/C0)
         for x,s in sorted(slices.items()) if 'min' in s and 'max' in s]

def pole_region():
 dirs={nl:[d for d in kind(GOOD,'pdir') if cfg(d)['nl']==nl] for nl in [0,3]}
 paired=sorted(pole_band())
 if not dirs[0] or not dirs[3]:return
 fig=plt.figure(figsize=(7.0,5.25));gs=fig.add_gridspec(2,2,left=.11,right=.98,bottom=.105,top=.87,wspace=.4,hspace=.52)
 axes=[fig.add_subplot(gs[:,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
 nl3=points(dirs[3],ratio=True)
 band_pts=[(p[0],p[1]) for p in paired]+[(p[0],p[2]) for p in paired]
 ranges=[(-1.025,.025,-.02,1.03),box(nl3,.08),box(band_pts,.18,[VS]) if band_pts else box(nl3,.02)]
 for ax,rng,title in zip(axes,ranges,['(a) Overview','(b) NL3 zoom','(c) NL6 zoom']):
  levels=[0,3]if ax is axes[0]else[3]if ax is axes[1]else[]
  for nl in levels:
   polygon(ax,dirs[nl],nl,ratio=True,alpha=.12 if nl==0 else .22,color=POLE_REGION[nl])
  vs_marker(ax,VS)
  if paired and ax is not axes[1]:
   xs=np.array([p[0]for p in paired]);lo=np.array([p[1]for p in paired]);hi=np.array([p[2]for p in paired])
   # Keep the existing checked-slice groups; end segments stay at their extrema.
   groups=np.split(np.arange(len(xs)),np.flatnonzero(np.diff(xs)>3e-5)+1)
   for group in groups:
    if len(group)<2:continue
    sampled_band(ax,xs[group],lo[group],hi[group],POLE_REGION[6],alpha=.27,zorder=4)
  ax.set(xlim=rng[:2],ylim=rng[2:],title=title,xlabel=r'$M^4c_{1,0}/c_{0,0}$',ylabel=r'$M^8c_{2,0}/c_{0,0}$');prettify(ax)
  ax.grid(False)
  if ax is not axes[0]:direct_axes(ax)
 for parent,target,label in [(axes[0],ranges[1],'(b)'),(axes[1],ranges[2],'(c)')]:
  zoom_guide(parent,target,label)
 fig.legend([Patch(facecolor=POLE_REGION[0],alpha=.18),Patch(facecolor=POLE_REGION[3],alpha=.28),
             Patch(facecolor=POLE_REGION[6],alpha=.35),vs_handle()],
  ['NL0','NL3','NL6','Virasoro-Shapiro'],loc='upper center',ncol=4,frameon=False,columnspacing=1.4,handlelength=1.6)
 export(fig,'fig-subtracted-regions-sdpx256',dirs[0]+dirs[3]+kind(GOOD,'pband'))
 MANIFEST['_pole_reference_display']=dict(coordinates=list(VS),marker='red point',marker_area_pt2=14,
      full_displayed_regions_visible=True,zoom_boxes_match_panel_limits=True,reference_point_shifted=False,
      zoom_axes_reference_relative=False,zoom_axes_show_direct_coefficient_ratios=True)
 return ranges

def band_of(variant):
 slices=slice_pairs([d for d in GOOD if cfg(d)['kind']=='pband' and cfg(d).get('variant','base')==variant],'c10','c20')
 return sorted((float(cfg(s['min'])['scan_x']),val(s['min'],'c20')/C0,val(s['max'],'c20')/C0)
               for x,s in slices.items() if 'min' in s and 'max' in s)

def pole_band_truncations():
 """NL6 islands at c00=2 zeta(3), labeled with direct coefficient ratios."""
 series=[('P8',r'$N_{\rm null}=36$','#9ecae1'),('P10',r'$N_{\rm null}=55$','#4292c6'),('base',r'$N_{\rm null}=78$','#08519c'),
         ('P14',r'$N_{\rm null}=105$','#08306b'),('N1024',r'$N_{\rm null}=78$, $N_\mu=1024$','#c56f09')]
 x0,y0=VS;fig,axes=plt.subplots(1,2,figsize=(7.4,4.15))
 fig.subplots_adjust(left=.095,right=.98,bottom=.18,top=.79,wspace=.30)
 used=[];handles=[];labels=[]
 for v,label,color in series:
  b=band_of(v)
  if len(b)<2:continue
  xs=(np.array([r[0] for r in b])-x0)*1e5;lo=(np.array([r[1] for r in b])-y0)*1e5;hi=(np.array([r[2] for r in b])-y0)*1e5
  groups=np.split(np.arange(len(xs)),np.flatnonzero(np.diff(xs)>3)+1)
  for ax in axes:
   for group in groups:
    if len(group)<2:continue
    sampled_band(ax,xs[group],lo[group],hi[group],color,alpha=.10 if v=='base' else .055,
                 lw=1.1 if v in ('base','N1024','P14') else .85,
                 zorder=5 if v=='N1024' else 3,
                 fill=v!='N1024',ls='--' if v=='N1024' else '-')
  used+=[d for d in GOOD if cfg(d)['kind']=='pband' and cfg(d).get('variant','base')==v]
  handles.append(Line2D([],[],color=color,lw=2 if v!='N1024' else 1,ls='-' if v!='N1024' else '--'));labels.append(label)
 for ax in axes:vs_marker(ax,(0,0))
 handles.append(vs_handle());labels.append('Virasoro-Shapiro')
 if not used:plt.close(fig);return
 allx=np.concatenate([[r[0] for r in band_of(v)] for v,_,_ in series if band_of(v)])
 axes[0].set(xlim=((allx.min()-x0)*1e5-1.5,(allx.max()-x0)*1e5+1.5))
 ys=np.concatenate([[r[1] for r in band_of(v)]+[r[2] for r in band_of(v)] for v,_,_ in series if band_of(v)])
 axes[0].set(ylim=((ys.min()-y0)*1e5-.4,(ys.max()-y0)*1e5+.4),title='(a) Full regions')
 zoom=(-1.2,1.2,-2.5,2.5)
 axes[1].set(xlim=zoom[:2],ylim=zoom[2:],title='(b) Boxed detail')
 zoom_guide(axes[0],zoom,'(b)',offset=(10,14))
 for ax in axes:
  prettify(ax);direct_axes(ax,VS,1e5)
 fig.supxlabel(r'$M^4c_{1,0}/c_{0,0}$',y=.035,fontsize=10)
 fig.supylabel(r'$M^8c_{2,0}/c_{0,0}$',x=.005,fontsize=10)
 fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=7.4,columnspacing=1.35)
 MANIFEST['_pole_band_zoom']=dict(limits=list(zoom),rectangle_matches_detail=True,
                                 coordinates='direct coefficient-ratio labels in both panels',
                                 internal_drawing_transform=dict(origin=list(VS),scale=1e5),
                                 deformation_curve_displayed=False)
 export(fig,'fig-pole-band-truncations-sdpx256',used)

# Interior-point solutions carry weights at the numerical floor (rho/rho_max ~ 1e-35 to 1e-24 for the
# 1e-30 and 1e-36 gap thresholds); every weight above it is plotted.
RHO_FLOOR=1e-22

def spectrum_comparison(items,prefix,jlabel,zoom_window,joint=False):
 """Every nonzero weight of the extremal measures, in the style of the original figures.
 The few weights at the top of the grid (spins of order 10^3-10^4) lie outside both windows."""
 colors=['#e07b39','#4daf4a','#2b2b2b'];sizes=[46,22,7]   # decreasing sizes keep coincident points visible
 if len(items)==1:colors,sizes=['#303030'],[20]
 for zoom,suffix in [(False,''),(True,'-zoom')]:
  fig,ax=plt.subplots(figsize=(3.5,2.75));fig.subplots_adjust(left=.17,right=.97,bottom=.17,top=.97)
  for d,color,size in zip(items,colors,sizes):
   spec=d['audit']['spectrum'];top=max(float(x['rho']) for x in spec)
   spec=[x for x in spec if float(x['rho'])>RHO_FLOOR*top];c=cfg(d);p=c['max_null_power']
   label=({'S1':'coarser','base':'baseline','S3':'finer'}[c['variant']] if joint
          else r'$N_\mu=%d$'%c['n_mu'])
   ax.scatter([x['spin']for x in spec],[float(x['mu'])for x in spec],s=size,color=color,lw=0,label=label,zorder=4)
  if zoom:ax.set(xlim=zoom_window[:2],ylim=zoom_window[2:])
  else:ax.set(xlim=(-3,125),ylim=(.8,400),yscale='log')
  ax.minorticks_on();ax.tick_params(which='both',direction='in',top=True,right=True)
  ax.set(xlabel=r'$\ell$',ylabel=r'$\mu/M^2$')
  if len(items)>1:ax.legend(loc='upper left' if not zoom else 'upper right',frameon=False,fontsize=7.5,handletextpad=.2,borderaxespad=.4)
  export(fig,prefix.replace('-sdpx256',suffix+'-sdpx256'),items)

def linear_case(variant):
 found=[d for d in DATA if cfg(d)['kind'] in ('glin_c00max','glin_c00max_scaled') and cfg(d)['variant']==variant]
 return found[0] if found else None

def mark(ax,x,d,color):
 if d is None:return
 if d.get('accepted') and d['classification']=='feasible':ax.plot([x],[val(d,'c00')],'o',color=color,ms=3.4,zorder=5)
 elif d.get('accepted') and d['classification']=='infeasible':
  ax.annotate('empty',xy=(x,.06),xycoords=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=6.8,color='#ad3446')
 else:ax.annotate('unres.',xy=(x,.06),xycoords=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=6.8,color='#777777')

def series_values(series,getter):
 """(x, value) for accepted feasible records, and the list of records used."""
 rows=[(x,getter(v)) for v,x in series];used=[d for _,d in rows if d is not None]
 return rows,used

def nl3_pair_of(v):
 lo=next((d for d in GOOD if d['case']=='gnl3c_%s_min'%v),None);hi=next((d for d in GOOD if d['case']=='gnl3c_%s_max'%v),None)
 return (lo,hi) if lo and hi else None

# Display the requested local-refinement range. Larger-energy and other Nt
# records remain in the unchanged audited input and the reproduction archive.
ENERGY_LADDER=[('K%d'%n,n) for n in (256,512,1024)]
NT_CHEB=[('T4',4),('base',6),('T8',8)]

def top_energy_axis(ax,xs):
 sec=ax.secondary_xaxis('top');sec.set_xticks(xs);sec.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
 sec.set_xticklabels([r'$%.0f\times10^{%d}$'%(m/10**int(math.floor(math.log10(m))),int(math.floor(math.log10(m))))
                      for m in (1/math.sin(math.pi/(2*x))**2 for x in xs)],fontsize=6.3)
 sec.tick_params(direction='in',length=2.5);sec.set_xlabel(r'largest energy node $\mu_1/M^2$',fontsize=7.5)

def gravity_convergence():
 if linear_case('base') is None:return
 fig,axes=plt.subplots(2,2,figsize=(7,6.0));fig.subplots_adjust(left=.11,right=.98,bottom=.09,top=.89,wspace=.38,hspace=.62)
 used=[];blue,orange,purple='#356b9c','#c68b32','#8e557e'
 ylab=r'$c_{0,0}^{\max}M^6/(8\pi G)$'
 # (a) Energy nodes: the open Chebyshev grid reaches mu_1 = csc^2(pi/2N), so refining also raises the top energy.
 ax=axes[0,0];rows=[(x,d) for x,d in ((x,linear_case(v)) for v,x in ENERGY_LADDER) if d is None or d.get('accepted')]
 if not any(d for _,d in rows):rows=[(x,linear_case(v)) for v,x in [('N256',256),('base',512),('N1024',1024)]]
 used+=[d for _,d in rows if d];ok=[(x,val(d,'c00')) for x,d in rows if d and d.get('accepted') and d['classification']=='feasible']
 if ok:ax.plot(*zip(*ok),'-o',color=blue,ms=3.4,lw=.9)
 for x,d in rows:
  if d is None or not (d.get('accepted') and d['classification']=='feasible'):mark(ax,x,d,blue)
 ax.set_xscale('log',base=2);xs=[x for x,d in rows if d is not None];ax.set_xticks(xs);ax.set_xticklabels([str(x) for x in xs])
 ax.set(title=r'(a) Energy nodes ($J_{\max}=10^4$, $N_t=6$, 55 nulls)',xlabel=r'$N_\mu$',ylabel=ylab);prettify(ax);ax.minorticks_off()
 top_energy_axis(ax,xs)
 # (b) Spin cutoff, (d) null constraints.
 for ax,series,xlabel,title in [(axes[0,1],[('J6000',6000),('base',10000),('J16000',16000)],r'$J_{\max}$',r'(b) Spin cutoff ($N_\mu=512$, 55 nulls)'),
         (axes[1,1],[('P8',36),('base',55),('P12',78)],r'$N_{\rm null}$',r'(d) Null constraints ($N_\mu=512$, $J_{\max}=10^4$)')]:
  rows=[(x,linear_case(v)) for v,x in series];used+=[d for _,d in rows if d]
  ok=[(x,val(d,'c00')) for x,d in rows if d and d.get('accepted') and d['classification']=='feasible']
  if ok:ax.plot(*zip(*ok),'-o',color=blue,ms=3.4,lw=.9)
  ax.set(title=title,xlabel=xlabel,ylabel=ylab,xticks=[x for _,x in series]);prettify(ax)
 # (c) Gravity nodes: Chebyshev nodes crowd toward t=0 like 1/N_t^2, equally spaced ones like 1/N_t.
 ax=axes[1,0]
 for series,color,marker,label in [(NT_CHEB,blue,'o',r'$t_i=\frac12[\cos\frac{(2i-1)\pi}{2N_t}-1]$')]:
  rows=[(x,linear_case(v)) for v,x in series];used+=[d for _,d in rows if d]
  ok=[(x,val(d,'c00')) for x,d in rows if d and d.get('accepted') and d['classification']=='feasible']
  if ok:ax.plot(*zip(*ok),'-',marker=marker,color=color,ms=3.2,lw=.9,label=label)
  for x,d in rows:
   if d is not None and d.get('accepted') and d['classification']=='infeasible':
    ax.plot([x],[0.04],marker='x',color=color,ms=5,transform=ax.get_xaxis_transform(),clip_on=False)
   elif d is not None and d['classification']=='unbounded':
    ax.plot([x],[0.96],marker='^',color=color,ms=4.5,transform=ax.get_xaxis_transform(),clip_on=False)
 ax.set(title=r'(c) Gravity nodes ($N_\mu=512$, $J_{\max}=10^4$)',xlabel=r'$N_t$',ylabel=ylab,
        xticks=[x for _,x in NT_CHEB]);prettify(ax)
 # Panels (a), (b), (d) on one window, so that their variations can be compared directly.
 ys=[v for ax in (axes[0,0],axes[0,1],axes[1,1]) for line in ax.get_lines() for v in line.get_ydata()]
 if ys:
  mid=.5*(max(ys)+min(ys));half=max(.006,.6*(max(ys)-min(ys)))
  for ax in (axes[0,0],axes[0,1],axes[1,1]):ax.set_ylim(mid-half,mid+half)
 export(fig,'fig-gravity-convergence-sdpx256',used)

def nl3_convergence():
 """NL3 interval of c10 at c00 = 2 zeta(3) against the same four settings."""
 if nl3_pair_of('base') is None:return
 fig,axes=plt.subplots(2,2,figsize=(7,6.0));fig.subplots_adjust(left=.12,right=.98,bottom=.09,top=.89,wspace=.4,hspace=.62)
 used=[];ylab=r'$[c_{1,0}+2\zeta(5)]M^{10}/(8\pi G)$'
 def draw(ax,series,color,dx=0.0,log=False,label=None):
  pts=[]
  for v,x in series:
   pr=nl3_pair_of(v)
   if not pr:continue
   used.extend(pr);xx=x*(2**dx) if log else x+dx
   lo,hi=val(pr[0],'c10')-C1,val(pr[1],'c10')-C1;pts.append((xx,lo,hi))
   ax.plot([xx,xx],[lo,hi],color=color,lw=4,alpha=.35,solid_capstyle='butt')
   # Span of the feasible NL6 tests of the same truncation on the same slice.
   six=[val(d,'c10')-C1 for d in nl6_string_records(v) if d['classification']=='feasible']
   if six:
    used.extend(nl6_string_records(v))
    ax.plot([xx,xx],[min(six),max(six)],color=REGION[6],lw=4,solid_capstyle='butt',zorder=6)
  if pts:
   ax.plot([p[0] for p in pts],[p[1] for p in pts],'-',color=color,lw=.7);ax.plot([p[0] for p in pts],[p[2] for p in pts],'-',color=color,lw=.7,label=label)
 blue,orange='#356b9c','#c68b32'
 draw(axes[0,0],ENERGY_LADDER,blue,log=True);axes[0,0].set_xscale('log',base=2)
 xs=[x for v,x in ENERGY_LADDER if nl3_pair_of(v)];axes[0,0].set_xticks(xs);axes[0,0].set_xticklabels([str(x) for x in xs]);axes[0,0].minorticks_off()
 top_energy_axis(axes[0,0],xs)
 draw(axes[0,1],[('J6000',6000),('base',10000),('J16000',16000)],blue)
 draw(axes[1,0],NT_CHEB,blue);axes[1,0].set_xticks([x for _,x in NT_CHEB])
 draw(axes[1,1],[('P8',36),('base',55),('P12',78)],blue)
 for ax,title,xlabel in [(axes[0,0],r'(a) Energy nodes',r'$N_\mu$'),(axes[0,1],r'(b) Spin cutoff',r'$J_{\max}$'),
                         (axes[1,0],r'(c) Gravity nodes',r'$N_t$'),(axes[1,1],r'(d) Null constraints',r'$N_{\rm null}$')]:
  ax.axhline(0,color=REGION['string'],ls='--',lw=.8);ax.set(title=title,xlabel=xlabel,ylabel=ylab)
  if ax is not axes[0,0]:prettify(ax)
 prettify(axes[0,0]);axes[0,0].minorticks_off()
 export(fig,'fig-gravity-nl3-parameters-sdpx256',used)

VARIANTS=[('base','base'),('N256',r'$N_\mu\!=\!256$'),('N1024',r'$N_\mu\!=\!1024$'),('J6000',r'$J\!=\!6000$'),('J16000',r'$J\!=\!16000$'),
          ('T4',r'$N_t\!=\!4$'),('T8',r'$N_t\!=\!8$'),('P8',r'36 nulls'),('P12',r'78 nulls')]

def nonlinear_gravity_cutoffs():
 nl3={}
 for d in [d for d in GOOD if cfg(d)['kind']=='gnl3_convergence']:
  nl3.setdefault(cfg(d)['variant'],{})['min' if float(cfg(d)['objective']['c10'])>0 else 'max']=d
 if 'base' not in nl3:return
 fig,ax=plt.subplots(figsize=(7,3.1));fig.subplots_adjust(left=.1,right=.98,bottom=.27,top=.83)
 used=[]
 for i,(v,label) in enumerate(VARIANTS):
  s=nl3.get(v,{})
  if 'min' in s and 'max' in s:
   lo,hi=val(s['min'],'c10')-C1,val(s['max'],'c10')-C1
   ax.plot([i,i],[lo,hi],color=COL[3],lw=5,alpha=.45,solid_capstyle='butt');used+=list(s.values())
  ok,bad=nl6_tests(v,'gnl6_convergence')
  if ok:
   cs=[c-C1 for _,c in ok];ax.plot([i,i],[min(cs),max(cs)],color=REGION[6],lw=1.6);ax.plot([i,i],[min(cs),max(cs)],'_',color=REGION[6],ms=7)
  used+=[d for d in DATA if cfg(d)['kind']=='gnl6_convergence' and cfg(d)['variant']==v]
 ax.axhline(0,color=REGION['string'],ls='--',lw=.8)
 ax.set_xticks(range(len(VARIANTS)));ax.set_xticklabels([l for _,l in VARIANTS],rotation=35,ha='right')
 ax.set(ylabel=r'$[c_{1,0}+2\zeta(5)]M^{10}/(8\pi G)$',title=r'Fixed $c_{0,0}M^6/(8\pi G)=2\zeta(3)$: one truncation changed at a time');prettify(ax);ax.minorticks_off()
 fig.legend([Line2D([],[],color=COL[3],lw=5,alpha=.45),Line2D([],[],color=REGION[6],lw=1.6),Line2D([],[],color=REGION['string'],ls='--')],
            ['NL3 interval','NL6 feasible tests (span)','string value'],loc='upper center',ncol=3,frameon=False)
 export(fig,'fig-gravity-nl3-cutoff-checks-sdpx256',used)

def nl6_convergence():
 """NL6 alone, with the uncertainty between each feasible endpoint and its nearest excluded test."""
 fig,axes=plt.subplots(2,2,figsize=(7,5.7))
 fig.subplots_adjust(left=.12,right=.98,bottom=.1,top=.88,wspace=.4,hspace=.48)
 used=[];limits=[]
 panels=[(axes[0,0],[('N256',256),('base',512),('N1024',1024)],r'$N_\mu$','(a) Energy nodes'),
         (axes[0,1],[('J6000',6000),('base',10000),('J16000',16000)],r'$J_{\max}$','(b) Spin cutoff'),
         (axes[1,0],NT_CHEB,r'$N_t$','(c) Gravity nodes'),
         (axes[1,1],[('P8',36),('base',55),('P12',78)],r'$N_{\rm null}$','(d) Null constraints')]
 for ax,series,xlabel,title in panels:
  for variant,x in series:
   records=nl6_string_records(variant)
   feasible=[float(cfg(d)['c10']) for d in records if d['classification']=='feasible']
   if not feasible:continue
   lo,hi=min(feasible),max(feasible)
   excluded=[float(cfg(d)['c10']) for d in records if d['classification']=='infeasible']
   below=max((q for q in excluded if q<lo),default=lo)
   above=min((q for q in excluded if q>hi),default=hi)
   a,b,c,d=[(q-C1)*1e4 for q in (below,lo,hi,above)]
   ax.plot([x,x],[b,c],color=REGION[6],lw=4,alpha=.72,solid_capstyle='butt')
   ax.plot([x,x],[b,c],'_',color=REGION[6],ms=8,zorder=5)
   ax.plot([x,x],[a,b],color='#666666',ls='--',lw=1.0)
   ax.plot([x,x],[c,d],color='#666666',ls='--',lw=1.0)
   ax.plot([x,x],[a,d],'_',color='#666666',ms=5)
   limits.extend([a,d]);used.extend(records)
  ax.axhline(0,color=REGION['string'],ls='--',lw=.8)
  ax.set(title=title,xlabel=xlabel,xticks=[x for _,x in series],
         ylabel=r'$10^4[c_{1,0}+2\zeta(5)]M^{10}/(8\pi G)$')
  prettify(ax)
 if limits:
  span=max(limits)-min(limits)
  for ax in axes.flat:ax.set_ylim(min(limits)-.06*span,max(limits)+.06*span)
 fig.legend([Line2D([],[],color=REGION[6],lw=4,alpha=.72),
             Line2D([],[],color='#666666',ls='--'),Line2D([],[],color=REGION['string'],ls='--')],
            ['Feasible tests (span)','Adjacent excluded-test bracket','String value'],
            loc='upper center',ncol=3,frameon=False,fontsize=8)
 export(fig,'fig-gravity-nl6-cutoff-checks-sdpx256',used)

def cutoff_plot():
 grouped={}
 for d in [d for d in GOOD if cfg(d)['kind']=='pconv']:
  grouped.setdefault(cfg(d)['variant'],{})['min' if float(cfg(d)['objective']['c20'])>0 else 'max']=d
 if 'base' not in grouped:return
 fig,axes=plt.subplots(1,3,figsize=(7,2.8));fig.subplots_adjust(left=.1,right=.98,bottom=.25,top=.84,wspace=.4)
 panels=[(axes[0],[('N256',256),('base',512),('N1024',1024)],r'$N_\mu$','Energy nodes'),
         (axes[1],[('J1000',1000),('base',2000),('J4000',4000)],r'$J_{\max}$','Spin cutoff'),
         (axes[2],[('P8',36),('P10',55),('base',78),('P14',105)],r'$N_{\rm null}$','Null tower')]
 used=[]
 for ax,series,xlabel,title in panels:
  for v,x in series:
   s=grouped.get(v,{})
   if 'min' not in s or 'max' not in s:continue
   lo=(val(s['min'],'c20')-C2)*1e6;hi=(val(s['max'],'c20')-C2)*1e6
   ax.plot([x,x],[lo,hi],color=COL[6]);ax.plot([x,x],[lo,hi],'_',color=COL[6],ms=7);used+=list(s.values())
  ax.axhline(0,color='#303030',ls='--',lw=.7);prettify(ax);ax.set(title=title,xlabel=xlabel,xticks=[x for _,x in series])
 axes[0].set_ylabel(r'$10^6\,[c_{2,0}-2\zeta(7)]M^8$')
 export(fig,'fig-pole-cutoff-checks-sdpx256',used)

def small_alpha_plot():
 rows=reference_checks(DATA)
 selected=[r for r in rows if float(r['alpha'])<=.9 or r['alpha']=='1']
 if not any(r['gravity_nl3'] for r in selected):return
 fig,axes=plt.subplots(1,2,figsize=(7,3.15))
 fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.83,wspace=.40)
 ax=axes[0];n3=[r for r in selected if r['gravity_nl3']]
 x=[float(r['alpha']) for r in n3]
 lo=[float((r['gravity_nl3'][0]-r['string_c10'])/abs(r['string_c10'])) for r in n3]
 hi=[float((r['gravity_nl3'][1]-r['string_c10'])/abs(r['string_c10'])) for r in n3]
 ax.fill_between(x,lo,hi,color=REGION[3],alpha=.19)
 ax.plot(x,lo,color=REGION[3],lw=.9);ax.plot(x,hi,color=REGION[3],lw=.9)
 for r in selected:
  for a,b in r['gravity_nl6']:
   y=[float((q-r['string_c10'])/abs(r['string_c10'])) for q in (a,b)]
   ax.plot([float(r['alpha'])]*2,y,color=REGION[6],lw=3,solid_capstyle='butt')
 ax.axhline(0,color=REGION['string'],ls='--',lw=.9)
 ax.set(title='(a) Gravitational NL3 and NL6',xlabel=r'$\alpha$',
        ylabel=r'$[c_{1,0}-c_{1,0}^{\rm VS}]/|c_{1,0}^{\rm VS}|$')
 ax=axes[1];pol=[r for r in selected if r['pole_nl6']]
 for r in pol:
  y=[float((q-r['string_c20'])/r['string_c20']) for q in r['pole_nl6']]
  ax.plot([float(r['alpha'])]*2,y,color=POLE_REGION[6],lw=3,solid_capstyle='butt')
 ax.axhline(0,color=REGION['string'],ls='--',lw=.9)
 ax.set(title='(b) Pole-subtracted NL6 string slice',xlabel=r'$\alpha$',
        ylabel=r'$[c_{2,0}-c_{2,0}^{\rm VS}]/c_{2,0}^{\rm VS}$')
 for ax in axes:
  ax.set_xlim(.06,1.04);ax.set_xticks([.1,.2,.4,.6,.8,1]);prettify(ax)
 fig.legend([Patch(facecolor=REGION[3],alpha=.25),Line2D([],[],color=REGION[6],lw=3),
             Line2D([],[],color=POLE_REGION[6],lw=3),Line2D([],[],color=REGION['string'],ls='--')],
            ['NL3','NL6 tests','NL6 interval','String value'],loc='upper center',ncol=4,frameon=False,fontsize=8)
 sources={case for r in selected for case in r['source_cases']}
 export(fig,'fig-small-alpha-checks-sdpx256',[d for d in DATA if d['case'] in sources])

def small_alpha_tail_plot():
 rows=tail_checks(DATA)
 if not any(r['statuses']['tail_E12'] in ('feasible','infeasible') for r in rows):return
 from matplotlib.colors import ListedColormap
 colors=ListedColormap(['#ececec','#c0ddd8','#efd4b5'])
 codes={'pending':0,'unresolved':0,'feasible':1,'infeasible':2}
 labels={'pending':'?','unresolved':'?','feasible':'F','infeasible':'I'}
 variants=['base','tail_E6','tail_E8','tail_E12']
 fig,axes=plt.subplots(1,2,figsize=(7,3.3))
 fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.80,wspace=.36)
 for ax,nl in zip(axes,(3,6)):
  selected=[r for r in rows if r['nl']==nl]
  matrix=np.array([[codes[r['statuses'][v]] for v in variants] for r in selected])
  ax.imshow(matrix,cmap=colors,vmin=0,vmax=2,aspect='auto',origin='upper')
  for i,r in enumerate(selected):
   for j,v in enumerate(variants):ax.text(j,i,labels[r['statuses'][v]],ha='center',va='center',fontsize=10)
  ax.set_xticks(range(4));ax.set_xticklabels(['baseline',r'$10^6$',r'$10^8$',r'$10^{12}$'])
  ax.set_yticks(range(5));ax.set_yticklabels([r['alpha'] for r in selected])
  ax.set(xlabel=r'Added energy reach $\mu_E/M^2$',ylabel=r'$\alpha$',title='(%s) NL%d'%('a' if nl==3 else 'b',nl))
  ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,5,1),minor=True)
  ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='both',length=0)
 fig.legend([Patch(facecolor='#c0ddd8'),Patch(facecolor='#efd4b5')],
            ['F: feasible reference projection','I: excluded on this finite ansatz'],
            loc='upper center',ncol=2,frameon=False,fontsize=8)
 sources={case for r in rows for case in r['source_cases']}
 export(fig,'fig-small-alpha-tail-checks-sdpx256',[d for d in DATA if d['case'] in sources])

def main():
 if args.only:
  previous=OUT/'figure_sources.json'
  if previous.exists():MANIFEST.update(json.loads(previous.read_text()))
  selected={'gravity-convergence':gravity_convergence,'nl3-convergence':nl3_convergence,
            'nl6-convergence':nl6_convergence,'gravity-linear':gravity_linear,'gravity-nl3-region':gravity_nl3,
            'gravity-nl6-region':gravity_nl6,'pole-region':pole_region,
            'pole-band-truncations':pole_band_truncations,'pole-cutoff':cutoff_plot}
  for name in args.only:
   result=selected[name]()
   if name=='pole-region' and result:MANIFEST['_pole_zoom_ranges']=[list(map(float,r)) for r in result]
  if any(name in args.only for name in ('gravity-convergence','nl3-convergence','nl6-convergence')):
   MANIFEST['_convergence_display']=dict(energy_counts=[256,512,1024],gravity_sample_counts=[4,6,8],
       input_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
       higher_resolution_records_preserved=True,unrestricted_convergence_claimed=False)
  if any(name in args.only for name in ('gravity-nl3-region','gravity-nl6-region','pole-region','pole-band-truncations')):
   MANIFEST['_region_outline_display']=dict(end_segments_at_existing_sampled_coordinates=True,
       excluded_and_unresolved_window_splits_preserved=True,extrapolated_coordinates=False,
       continuous_NL6_boundary_certified=False,
       input_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
  (OUT/'figure_sources.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')
  print('Rendered',len(args.only),'selected figure groups; other figure bytes preserved.')
  return
 gravity_linear();gravity_nl3();gravity_nl6();ranges=pole_region();cutoff_plot();gravity_convergence();nonlinear_gravity_cutoffs()
 pole_band_truncations();nl3_convergence();nl6_convergence()
 small_alpha_plot();small_alpha_tail_plot()
 for names,prefix,jlabel,window in [(['pconv_N256_min','pconv_base_min','pconv_N1024_min'],'fig-subtracted-spectrum-sdpx256','2000',(-1,25,0,7)),
                                    (['glin_c00max_N256','glin_c00max_base','glin_c00max_N1024'],'fig-gpole-spectrum-sdpx256','10^4',(-1,20,0,8))]:
  found=[next((d for d in GOOD if d['case']==name and 'spectrum' in d['audit']),None) for name in names]
  if all(found):
   if prefix.startswith('fig-gpole'):
    joint=[next((d for d in GOOD if d['case']==name and 'spectrum' in d['audit']),None)
           for name in ['glin_c00max_S1','glin_c00max_base','glin_c00max_S3']]
    if all(joint):spectrum_comparison(joint,prefix,jlabel,window,joint=True)
   else:spectrum_comparison([found[1]],prefix,jlabel,window)
   spectrum_comparison(found,prefix.replace('-sdpx256','-grid-checks-sdpx256'),jlabel,window)
 if ranges:MANIFEST['_pole_zoom_ranges']=[list(map(float,r)) for r in ranges]
 pairs,end=nl3_slices()
 if pairs:MANIFEST['_gravity_zoom_ranges']=[list(map(float,x)) for x in zoom_window(pairs,end)]
 (OUT/'figure_sources.json').write_text(json.dumps(MANIFEST,indent=2)+'\n')
 print('Rendered',len([k for k in MANIFEST if not k.startswith('_')]),'verified scientific figures.')
if __name__=='__main__':main()
