"""One bounded planner for the new namespace; it never changes the preceding campaign."""
import argparse,json,time
from pathlib import Path
from plan import plan,signature,M

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--once',action='store_true')
    a=p.parse_args();root=Path(a.root);q=root/'queue'
    for x in ['tasks','claims']:(q/x).mkdir(parents=True,exist_ok=True)
    (root/'logs').mkdir(exist_ok=True);(root/'results').mkdir(exist_ok=True)
    def lookup(name):
        if not (q/'claims'/name/'done').exists():return None
        f=root/'results'/name/'verified_summary_v5.json'
        if not f.exists():return {'cls':'unresolved'}
        d=json.loads(f.read_text());r={'cls':d['classification'] if d.get('accepted') else 'unresolved'}
        for k,v in d.get('audit',{}).get('wilson',{}).items():r[k]=M(v)
        return r
    while True:
        P=plan(lookup);new=0
        for name,c in P.tasks.items():
            f=q/'tasks'/('%d__%s__%s.json'%(c['priority'],signature(c),name))
            if f.exists():continue
            tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(c,sort_keys=True,indent=1)+'\n');tmp.rename(f);new+=1
        (root/'data/aliases.json').write_text(json.dumps(P.aliases,sort_keys=True,indent=1)+'\n')
        (root/'data/current_manifest.json').write_text(json.dumps(list(P.tasks.values()),sort_keys=True,indent=1)+'\n')
        classes={};done=0;running=0
        for name in P.tasks:
            r=lookup(name)
            if r:done+=1;classes[r['cls']]=classes.get(r['cls'],0)+1
            elif (q/'claims'/name).exists():running+=1
        status=dict(tasks=len(P.tasks),aliases=len(P.aliases),done=done,running=running,new=new,
                    classifications=classes,time=time.time())
        (root/'logs/planner_status.json').write_text(json.dumps(status,indent=1)+'\n')
        print(json.dumps(status),flush=True)
        if new==0 and done==len(P.tasks) and not running:
            (q/'planner_done').write_text(json.dumps(status)+'\n');return
        if a.once:return
        time.sleep(20)

if __name__=='__main__':main()
