"""Two-task message intervention, with gain selection on training-only validation."""
import argparse
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.scripts.local_orchestrator_pilot import Network, fit, ROOT, data_for_seed, build_pools

@torch.no_grad()
def diagnose(model, x, y, classes):
    z, _ = model.encode(x)
    on = model.head(z)
    saved = model.gain
    model.gain = 0
    off = model(x)
    model.gain = saved
    layers = []
    incoming = x
    for layer in model.layers:
        outgoing, gate, winner = layer(incoming, model.arm, model.steps, model.gain)
        _, gate_off, _ = layer(incoming, model.arm, model.steps, 0)
        delta = (gate-gate_off).abs()
        hist = torch.bincount(winner, minlength=gate.shape[1]).float()/len(x)
        layers.append({'gate_change_mean': delta.mean().item(),
                       'gate_change_max': delta.max().item(),
                       'branches_changed_over_001': (delta>.01).float().mean().item(),
                       'winner_max_share': hist.max().item(),
                       'winner_histogram': hist.tolist()})
        incoming = outgoing
    return {'accuracy_on': (on.argmax(-1)==y).float().mean().item(),
            'accuracy_off': (off.argmax(-1)==y).float().mean().item(),
            'within_task_accuracy': (on[:,classes].argmax(-1)==y-classes[0]).float().mean().item(),
            'prediction_flip': (on.argmax(-1)!=off.argmax(-1)).float().mean().item(),
            'probability_change': (on.softmax(-1)-off.softmax(-1)).abs().mean().item(),
            'layers': layers}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--epochs', type=int, default=20)
    args=p.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=False)
    variants=[('context',0),('local',1),('local',4),('winner',1),('winner',4)]
    (out/'protocol.json').write_text(json.dumps({**vars(args),
        'variants':variants, 'tasks':[[0,1],[2,3]], 'initialization':'scratch; matched across arms',
        'data':'512 train + 128 validation per task, disjoint training-pool rows; 200 official test per task',
        'selection':'one gain per mechanism, maximizing mean final global validation accuracy across seeds; lower gain breaks ties; test read only after selection',
        'intervention':'all messages removed at inference on the same learned weights; per-layer gate deltas hold incoming activations fixed',
        'head_diagnostic':'within-task accuracy restricts logits to true digit pair; diagnostic only, uses task identity',
        'limitations':'two tasks, three seeds, no replay/protection/pretraining; off intervention may be out of training distribution'},indent=2))
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'model_source.py').write_text((ROOT/'bdh_cl/scripts/local_orchestrator_pilot.py').read_text())
    pools=build_pools(ROOT/'data_cache/mnist',side=14,task_tags=False)
    records=[]; snapshots={}; tests={}
    for seed in map(int,args.seeds.split(',')):
        data, indices=data_for_seed(pools,seed,640,200)
        data=data[:2]
        tests[seed]=[(d[2],d[3]) for d in data]
        (out/f'indices_seed{seed}.json').write_text(json.dumps(indices[:2]))
        torch.manual_seed(seed); initial=Network()
        for arm,gain in variants:
            model=copy.deepcopy(initial); model.arm=arm; model.gain=gain
            stages=[]; saved=[]
            for task,(x,y,_,_) in enumerate(data):
                fit(model,x[:512],y[:512],args.epochs,seed+200+task)
                stages.append([diagnose(model,d[0][512:],d[1][512:],[2*t,2*t+1]) for t,d in enumerate(data)])
                saved.append(copy.deepcopy(model.state_dict()))
            key=f'{seed}_{arm}_{gain}'
            torch.save(saved,out/f'{key}.pt'); snapshots[key]=saved
            row={'seed':seed,'arm':arm,'gain':gain,'validation':stages}
            records.append(row)
            print(key,'validation final',round(sum(d['accuracy_on'] for d in stages[-1])/2,3),flush=True)
    chosen={}
    for arm in ('context','local','winner'):
        gains=[g for a,g in variants if a==arm]
        def score(g):
            rows=[r for r in records if r['arm']==arm and r['gain']==g]
            return sum(sum(d['accuracy_on'] for d in r['validation'][-1])/2 for r in rows)/len(rows)
        chosen[arm]=max(gains,key=lambda g:(score(g),-g))
    (out/'selection.json').write_text(json.dumps(chosen,indent=2))
    # Only now evaluate official test images for the selected settings.
    for r in records:
        if r['gain']!=chosen[r['arm']]: continue
        model=Network(r['arm'],gain=r['gain'])
        r['test']=[]
        for state in snapshots[f'{r["seed"]}_{r["arm"]}_{r["gain"]}']:
            model.load_state_dict(state)
            r['test'].append([diagnose(model,x,y,[2*t,2*t+1]) for t,(x,y) in enumerate(tests[r['seed']])])
    (out/'results.json').write_text(json.dumps(records,indent=2))
    lines=['# Two-task message diagnostic','','Gain selected on validation; official test results below. Means across three paired seeds. Gate change is in absolute sigmoid units.','','| Mechanism | Gain | Final avg | Old task | New task | Old within-pair | Message-off final | Prediction flips | Gate delta L1 / L2 | Winner max share L1 / L2 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in chosen:
        rows=[r for r in records if r['arm']==arm and 'test' in r]
        def avg(f): return sum(f(r['test'][-1]) for r in rows)/len(rows)
        def pair(key): return ' / '.join(f'{avg(lambda ds: sum(d["layers"][i][key] for d in ds)/2):.3f}' for i in range(2))
        vals=[avg(lambda ds:sum(d['accuracy_on'] for d in ds)/2), avg(lambda ds:ds[0]['accuracy_on']),avg(lambda ds:ds[1]['accuracy_on']),avg(lambda ds:ds[0]['within_task_accuracy']),avg(lambda ds:sum(d['accuracy_off'] for d in ds)/2),avg(lambda ds:sum(d['prediction_flip'] for d in ds)/2)]
        lines.append(f'| {arm} | {chosen[arm]} | '+' | '.join(f'{v*100:.1f}%' for v in vals)+f' | {pair("gate_change_mean")} | {pair("winner_max_share")} |')
    lines+=['','Global prediction remains ten-class; uniform chance is 10%. Perfect final-task-only learning gives 50% average over these two equally weighted tasks. Within-pair evaluation uses task identity and is diagnostic only. Removing messages changes inference distribution; a small effect does not rule out training-time effects.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
