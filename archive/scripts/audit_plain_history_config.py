"""Audit every plain-MLP mask size and distinguish weight restoration controls."""
import json
import torch
from bdh_cl.data.mnist import load_raw, SPLITS
from bdh_cl.scripts.plain_history_config import OUT, Flat, evaluate, logits, accuracy, save


def main():
    torch.set_num_threads(2)
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rows=[]
    for path in sorted(OUT.glob('*_plain_*.pt')):
        seed,optimizer,_,direction=path.stem.split('_')
        order=SPLITS if direction=='forward' else SPLITS[::-1]
        state=torch.load(path,weights_only=True)
        m=Flat(int(seed),False);m.load_state_dict(state['final'])
        r={'seed':int(seed),'optimizer':optimizer,'order':direction,'masks':{}}
        for k in [25,50,75,100]:
            bank=[]
            for stage in state['stage_checkpoints_analysis_only']:
                mask=torch.zeros(100);mask[stage['utility'].topk(k).indices]=1;bank.append(mask)
            r['masks'][str(k)]=evaluate(m,bank,x,y,order)
        # These interventions restore WEIGHTS and are explicitly not config-only recovery.
        controls={}
        for label,keys in [('historical_hidden_weights',['hidden.weight','hidden.bias']),
                           ('historical_head_weights',['head.weight','head.bias']),
                           ('historical_all_weights',['hidden.weight','hidden.bias','head.weight','head.bias'])]:
            pred=torch.empty_like(y)
            for i,pair in enumerate(order):
                changed={key:v.clone() for key,v in state['final'].items()}
                for key in keys:changed[key]=state['stage_checkpoints_analysis_only'][i][key]
                m.load_state_dict(changed)
                ids=torch.isin(y,torch.tensor(pair));pred[ids]=logits(m,x[ids]).argmax(-1)
            controls[label]=accuracy(pred,y)
            m.load_state_dict(state['final'])
        r['weight_restoration_controls_not_config_recovery']=controls
        assert all(torch.equal(v,m.state_dict()[key]) for key,v in state['final'].items())
        rows.append(r)
    assert len(rows)==12
    save('plain_mask_and_weight_audit.json',rows)
    print(json.dumps({'audited_plain_models':len(rows),'all_mask_sizes':[25,50,75,100],'final_states_preserved':True}))


if __name__=='__main__':main()
