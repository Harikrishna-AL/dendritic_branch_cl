"""Long-budget, multiple-initialization test on the extra-feature toy."""
import json
from pathlib import Path
import numpy as np
from bdh_cl.scripts.frozen_reuse_pilot import GRID, update, predict, target

def main():
    rows=[]
    for seed in range(5):
        rng=np.random.default_rng(seed)
        old=np.array([[.2+rng.normal(0,.03),rng.normal(0,.03)]]);head=np.ones(1)
        for _ in range(800):
            x=rng.uniform(-1,1,32)
            update(x,np.tanh(2*x)+rng.normal(0,.03,32),old,head,np.array([True]))
        for init in range(5):
            for arm in ['frozen_residual','scratch']:
                irng=np.random.default_rng(100+init)
                b=irng.normal(0,1,(2,2));c=irng.normal(0,.3,2)
                if arm=='frozen_residual': b[0]=old[0];c[0]=0
                rng=np.random.default_rng(seed+10000)
                curve=[]
                for step in range(16001):
                    if step in [0,100,1600,4000,8000,16000]:
                        curve.append({'step':step,'mse':float(np.mean((predict(GRID,b,c)-target(GRID,'extra_feature'))**2))})
                    if step<16000:
                        x=rng.uniform(-1,1,32)
                        update(x,target(x,'extra_feature')+rng.normal(0,.03,32),b,c,np.array([arm=='scratch',True]))
                if arm=='frozen_residual': assert np.array_equal(b[0],old[0])
                rows.append({'seed':seed,'init':init,'arm':arm,'curve':curve})
    out=Path('results/reuse_extended');out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(rows,indent=2))
    for arm in ['frozen_residual','scratch']:
        group=[r for r in rows if r['arm']==arm]
        print(arm,{step:float(np.mean([next(c['mse'] for c in r['curve'] if c['step']==step) for r in group])) for step in [1600,16000]},flush=True)
        print('final range',min(r['curve'][-1]['mse'] for r in group),max(r['curve'][-1]['mse'] for r in group),flush=True)

if __name__=='__main__':main()
