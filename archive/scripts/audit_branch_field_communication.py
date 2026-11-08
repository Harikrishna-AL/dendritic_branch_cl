"""Frozen-model learned-versus-scrambled regional compatibility diagnostic."""
import json
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.branch_fields_retrieval import OUT,Model,Memory,evaluate,save


def main():
    torch.set_num_threads(2)
    selection=json.loads((OUT/'selection.json').read_text())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rows=[]
    for record in json.loads((OUT/'results.json').read_text()):
        arm=record['arm']
        if 'flat' in arm:continue
        seed,name=record['seed'],record['order'];order=SPLITS if name=='forward' else SPLITS[::-1]
        state=torch.load(OUT/f'{seed}_{name}_{arm}.pt',weights_only=True)
        m=Model(seed,arm);m.load_state_dict(state['model']);mem=Memory(m.field)
        for k,v in state['memory'].items():setattr(mem,k,v)
        s=next(s for s in selection[arm]['readouts'] if s['method']=='local' and s['scope']=='field' and s['gamma']>0)
        rr,diag=evaluate(m,mem,state['bank_analysis_only'],x,y,order,[s,{**s,'gamma':0.}])
        original=mem.links.clone();scrambled=[]
        # Five predetermined branch-label permutations, never selected on their results.
        for shuffle in range(5):
            gen=torch.Generator().manual_seed(991+shuffle)
            perms=torch.stack([torch.randperm(4,generator=gen) for _ in range(25)])
            mem.links=torch.stack([original[i][perms[i]][:,perms[(i+1)%25]] for i in range(25)])
            sr,_=evaluate(m,mem,state['bank_analysis_only'],x,y,order,[s])
            scrambled.append(sr[0]['accuracy'])
        mem.links=original
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in state['model'].items())
        rows.append({'seed':seed,'order':name,'arm':arm,'gamma':s['gamma'],
                     'learned':rr[0]['accuracy'],'off':rr[1]['accuracy'],
                     'scrambled':scrambled,'scrambled_mean':sum(scrambled)/5,
                     'final_weights_unchanged':True})
        save('communication_scramble_audit.json',rows)
    assert len(rows)==42
    save('communication_audit_completion.json',{'complete':True,'models':42,'fixed_permutations_per_model':5,
        'scope':'Post-primary diagnostic on same checkpoints; no retuning or independent replication. Permutes source/destination branch labels at each group, preserving link values but breaking unary correspondence.'})


if __name__=='__main__':main()
