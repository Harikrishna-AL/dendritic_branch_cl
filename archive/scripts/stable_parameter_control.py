"""Reproduce the parameter-matched frozen-feature MLP control."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.continual_stability_pilot import stable
from bdh_cl.scripts.eligibility_router_pilot import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rows=[]
    for order in ['forward','reverse']:
        pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
        groups=[torch.nonzero((y==u)|(y==v),as_tuple=True)[0] for u,v in pairs]
        for seed in [84,85,86]:
            m,states,stats=stable('mlp',seed,x,y,groups,1.,1522)
            stages=[]
            for state in states:m.load_state_dict(state);stages.append(evaluate(m,tx,ty))
            old=[i for i,pair in enumerate(SPLITS) if pair!=pairs[-1]]
            row={'order':order,'seed':seed,'kind':'parameter_matched_mlp','width':1522,
                 'accuracy':stages[-1]['accuracy'],'old_accuracy':statistics.mean(stages[-1]['pairs'][i] for i in old),
                 'stages':stages,**stats,'numeric_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values()),
                 'effective_weight_count':sum(p.numel() for p in m.first.parameters())+m.head.numel()}
            rows.append(row);torch.save(m.state_dict(),out/f'{order}_{seed}_parameter_matched_mlp.pt')
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            print(order,seed,row['accuracy'],flush=True)


if __name__=='__main__':main()
