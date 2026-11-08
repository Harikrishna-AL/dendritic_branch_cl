"""Locked-model stress tests: dwell time, gradual transitions, ambiguous predictability."""
import argparse,json,statistics,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.scripts.shared_context_memory_pilot import Model
from bdh_cl.scripts.deep_predictive_memory_pilot import Model as Deep,train

CONDITIONS={'short':(64,.8,0),'reference':(256,.8,0),'long':(1024,.8,0),
            'gradual32':(256,.8,32),'gradual128':(256,.8,128),
            'gap04':(256,.4,0),'gap01':(256,.1,0)}

def stress_stream(seed,block,gap,ramp,order=(0,1)*4,streams=16):
    rng=torch.Generator().manual_seed(seed);state=torch.randn(streams,2,generator=rng);history=[]
    def params(task):
        w=torch.tensor([1.,0.] if task==0 else [0.,1.]);return .5+gap*(w-.5),w
    rho,_=params(order[0])
    for _ in range(8):
        history.append(state+.05*torch.randn(streams,2,generator=rng))
        state=rho*state+torch.sqrt(1-rho.square())*torch.randn(streams,2,generator=rng)
    for t in range(block*len(order)):
        phase=t//block;offset=t%block;task=order[phase];rho,w=params(task)
        if phase and ramp and offset<ramp:
            a=(offset+1)/ramp;oldrho,oldw=params(order[phase-1]);rho=(1-a)*oldrho+a*rho;w=(1-a)*oldw+a*w
        observation=state+.05*torch.randn(streams,2,generator=rng);history=history[1:]+[observation]
        nxt=rho*state+torch.sqrt(1-rho.square())*torch.randn(streams,2,generator=rng)
        target=(nxt*w).sum(-1)+.05*torch.randn(streams,generator=rng)
        # Diagnostic oracle conditional mean uses latent state and environment parameters.
        oracle=(rho*state*w).sum(-1)
        yield torch.stack(history,1).flatten(1),target,oracle,phase
        state=nxt

@torch.no_grad()
def evaluate(model,seed,condition,first=0):
    block,gap,ramp=CONDITIONS[condition];model.reset();rows=[{'errors':[],'oracle':[],'routes1':[],'routes2':[]} for _ in range(8)];previous=None
    churn=[]
    for x,y,oracle,phase in stress_stream(seed,block,gap,ramp,order=(first,1-first)*4):
        pred,q,w=model.predict(x);model.commit(q);r=rows[phase]
        r['errors'].append((pred-y).square().mean().item());r['oracle'].append((oracle-y).square().mean().item())
        r['routes1'].append(F.one_hot(w[0],4).float());r['routes2'].append(F.one_hot(w[1],32).float())
        if previous is not None:churn.append((w[1]!=previous).float().mean().item())
        previous=w[1].clone()
    phases=[]
    for i,r in enumerate(rows):
        phases.append({'task':first if i%2==0 else 1-first,'all_mse':statistics.mean(r['errors']),'first32_mse':statistics.mean(r['errors'][:32]),
                       'late_mse':statistics.mean(r['errors'][32:]),'oracle_mse':statistics.mean(r['oracle']),
                       'usage1':torch.stack(r['routes1'][block//2:]).mean((0,1)).tolist(),
                       'usage2':torch.stack(r['routes2'][block//2:]).mean((0,1)).tolist(),
                       'binned_mse':[statistics.mean(r['errors'][j:j+16]) for j in range(0,block,16)]})
    a=torch.tensor([p['usage2'] for p in phases[::2]]).mean(0);b=torch.tensor([p['usage2'] for p in phases[1::2]]).mean(0)
    return {'phases':phases,'all_mse':statistics.mean(p['all_mse'] for p in phases),
            'switch32_mse':statistics.mean(p['first32_mse'] for p in phases[1:]),
            'late_mse':statistics.mean(p['late_mse'] for p in phases),
            'old_revisit_mse':statistics.mean(p['all_mse'] for p in phases[2::2]),
            'oracle_mse':statistics.mean(p['oracle_mse'] for p in phases),
            'route_overlap':torch.minimum(a,b).sum().item(),'output_route_churn':statistics.mean(churn)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='3,4,5,6,7');args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({'conditions':CONDITIONS,'seeds':args.seeds,
      'selection':'all settings locked; no tuning on this stress test',
      'training':'reuse existing final checkpoints: original 4096 A then4096 B or reverse; history_only trained identically, single Adam .003 updates every8; no boundary resets',
      'evaluation':'eight continuous visits, starts with first trained task; reversed for reverse-trained models through environment schedule only; frozen weights, no target in fast writes, no task IDs or boundary signals; all observations counted',
      'gradual':'linearly interpolate both AR coefficients and teacher channel mixture over32 or128 steps; target itself smoothly changes, not just input distribution',
      'ambiguity':'meanrho.5, gap.8/.4/.1 gives .9/.1,.7/.3,.55/.45; next slowchannel target unchanged; noise floor changes',
      'oracle':'conditional mean using latent state and true environment parameters, diagnostic lower bound, never given to model; report excess MSE',
      'limits':'evaluation-only robustness of weights learned on original task; does not test learning under new schedules; predictable-signal objective remains engineered; timescale visits differ total evaluation observations; shared state budgets differ'},indent=2))
    records=[]
    for seed in map(int,args.seeds.split(',')):
        for first in (0,1):
            for variant in ('baseline','linear','neuron_shared','history_only'):
                torch.manual_seed(seed)
                if variant=='history_only':
                    model=Deep('history_only');states=train(model,seed+1000,4096,(first,1-first));torch.save(states,out/f'{seed}_{first}_{variant}.pt');model.load_state_dict(states[-1])
                else:
                    folder='shared_context_memory_confirmation_20261011' if variant=='neuron_shared' else 'branch_memory_confirmation_20261011'
                    model=Model(variant);states=torch.load(Path('results')/folder/f'{seed}_{first}_{variant}.pt',weights_only=True);model.load_state_dict(states[-1])
                result={c:evaluate(model,seed+5000,c,first) for c in CONDITIONS}
                records.append({'seed':seed,'first':first,'variant':variant,'conditions':result})
                (out/'results.json').write_text(json.dumps(records,indent=2));print(seed,first,variant,round(result['reference']['all_mse'],3),flush=True)
    lines=['# Fixed-model robustness tests','','All observations counted, including transitions. Five fresh seeds; both training orders. No settings selected here.','','| Condition | Baseline | Improved linear | Shared | History-only | Oracle |','|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        vals=[statistics.mean(r['conditions'][c]['all_mse'] for r in records if r['variant']==v) for v in ('baseline','linear','neuron_shared','history_only')]
        oracle=statistics.mean(r['conditions'][c]['oracle_mse'] for r in records)
        lines.append('| '+c+' | '+' | '.join(f'{x:.3f}' for x in vals+[oracle])+' |')
    lines+=['','## Errors during first32 observations after switches','','| Condition | Baseline | Linear | Shared | History-only |','|---|---:|---:|---:|---:|']
    for c in CONDITIONS:lines.append('| '+c+' | '+' | '.join(f'{statistics.mean(r["conditions"][c]["switch32_mse"] for r in records if r["variant"]==v):.3f}' for v in ('baseline','linear','neuron_shared','history_only'))+' |')
    lines+=['','## Recruitment and old-task revisits','','| Condition | Model | Output overlap | Route changes/step | Old revisit MSE | Excess above oracle |','|---|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        for v in ('baseline','linear','neuron_shared','history_only'):
            rs=[r['conditions'][c] for r in records if r['variant']==v]
            vals=[statistics.mean(r[k] for r in rs) for k in ('route_overlap','output_route_churn','old_revisit_mse')]+[statistics.mean(r['all_mse']-r['oracle_mse'] for r in rs)]
            lines.append('| '+c+' | '+v+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
