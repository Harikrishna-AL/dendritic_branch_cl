"""Held-out branch/regime association and history-only baseline diagnostics."""
import argparse
import json
from pathlib import Path
import torch
from bdh_cl.scripts.taskfree_local_pilot import make_data


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);rows=json.loads((out/'results.json').read_text());diagnostics=[]
    for r in rows:
        accuracy=[]
        for layer in range(2):
            counts=[torch.bincount(torch.tensor(d['winner'][layer]),minlength=128) for d in r['validation'][-1]['tasks']]
            mapping=(counts[1]>counts[0]).long()
            hits=total=0
            for task,d in enumerate(r['test'][-1]['tasks']):
                winners=torch.tensor(d['winner'][layer]);hits+=(mapping[winners]==task).sum().item();total+=len(winners)
            accuracy.append(hits/total)
        diagnostics.append({'seed':r['seed'],'arm':r['arm'],'held_out_winner_regime_accuracy':accuracy})
    baselines=[]
    for seed in sorted(set(r['seed'] for r in rows)):
        data=make_data(seed+3000,1024)
        for task,(x,y) in enumerate(data):
            history=x.reshape(-1,8,2)
            correlation=(history[:,:-1]*history[:,1:]).sum(1)/history[:,:-1].square().sum(1).clamp_min(1e-8)
            predicted_regime=correlation.argmax(-1)
            pred=.9*history[:,-1].gather(1,predicted_regime[:,None]).squeeze(1)
            baselines.append({'seed':seed,'regime':task,'history_inference_accuracy':(predicted_regime==task).float().mean().item(),'history_prediction_mse':(pred-y).square().mean().item()})
    report={'winner_diagnostics':diagnostics,'history_baselines':baselines,
            'meaning':'branch-to-regime mappings fitted only on held-out validation windows, then evaluated on independent test windows; regime labels used for analysis only; heuristic selects largest sample lag correlation, no learned task-ID routing'}
    (out/'held_out_analysis.json').write_text(json.dumps(report,indent=2))
    lines=['# Held-out context association','','Branch→regime mappings built on validation, evaluated on test; analysis only. Chance is 50%.','','| Arm | Layer 1 accuracy | Layer 2 accuracy |','|---|---:|---:|']
    for arm in sorted(set(r['arm'] for r in rows)):
        ds=[d for d in diagnostics if d['arm']==arm]
        acc=[sum(d['held_out_winner_regime_accuracy'][i] for d in ds)/len(ds) for i in range(2)]
        lines.append(f'| {arm} | {acc[0]*100:.1f}% | {acc[1]*100:.1f}% |')
    lines+=['',f'History-only lag-correlation heuristic: regime accuracy {sum(b["history_inference_accuracy"] for b in baselines)/len(baselines)*100:.1f}%; prediction MSE {sum(b["history_prediction_mse"] for b in baselines)/len(baselines):.3f}.']
    (out/'held_out_analysis.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
