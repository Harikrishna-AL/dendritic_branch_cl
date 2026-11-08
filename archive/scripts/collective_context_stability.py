"""Freeze observer cue decoder at checkpoint one and test at checkpoint two."""
import argparse,json
from pathlib import Path
import torch
from bdh_cl.scripts.collective_context_pilot import Model,make_data

@torch.no_grad()
def features(model,data):
    layers=[[],[]]
    for x,_ in data:
        _,states=model.encode(x)
        for i,s in enumerate(states):layers[i].append(s['context'].flatten(1))
    return [torch.cat(parts) for parts in layers]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(4);out=Path(args.out);records=json.loads((out/'results.json').read_text());result=[]
    for r in records:
        states=torch.load(out/f'{r["seed"]}_{r["arm"]}.pt',weights_only=True)['stages']
        model=Model(r['arm']);val=make_data(r['seed']+2000,512);test=make_data(r['seed']+3000,1024)
        model.load_state_dict(states[0]);a=features(model,val);before=features(model,test)
        model.load_state_dict(states[1]);after=features(model,test)
        y=torch.cat([-torch.ones(512),torch.ones(512)]);target=torch.cat([torch.zeros(1024),torch.ones(1024)]).bool()
        metrics=[]
        for train,old,new in zip(a,before,after):
            mean=train.mean(0);std=train.std(0).clamp_min(.01)
            def normalize(x):return torch.cat([(x-mean)/std,torch.ones(len(x),1)],1)
            train=normalize(train);coef=torch.linalg.solve(train.T@train+10*torch.eye(513),train.T@y)
            metrics.append({'before':((normalize(old)@coef>0)==target).float().mean().item(),
                            'after':((normalize(new)@coef>0)==target).float().mean().item()})
        result.append({'seed':r['seed'],'arm':r['arm'],'layers':metrics})
    (out/'stability.json').write_text(json.dumps(result,indent=2))
    lines=['# Frozen observer decoder stability','','Train decoder using checkpoint-1 validation contexts; keep weights and normalization fixed for checkpoint-2 test. This decoder never enters the network.','','| Exchange | L1 before → after | L2 before → after |','|---|---:|---:|']
    for arm in ('independent','average','learned'):
        rows=[r for r in result if r['arm']==arm];cells=[]
        for i in range(2):
            before=sum(r['layers'][i]['before'] for r in rows)/len(rows)*100
            after=sum(r['layers'][i]['after'] for r in rows)/len(rows)*100
            cells.append(f'{before:.1f}% → {after:.1f}%')
        lines.append(f'| {arm} | '+' | '.join(cells)+' |')
    (out/'stability.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
