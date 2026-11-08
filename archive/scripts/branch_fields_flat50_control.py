"""More-active ordinary-neuron controls; validation selected before own test cohort."""
import json
import statistics as st
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.branch_fields_retrieval import OUT,train,evaluate,settings,family,save


def main():
    torch.set_num_threads(2)
    save('flat50_protocol.json',{'purpose':'Separate two participating branches from simply using50 active ordinary neurons.',
        'status':'Added while primary confirmation was running; follows previously stated active-count control. Same fixed selection procedure and paired cohort; not independent replication.',
        'selection':'Seed560 50k/10k; LR.001/.003 on historical validation; automatic readouts separately selected; test does not select settings.',
        'arms':['full_flat50','quadrant_flat50']})
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(560)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
    selection={}
    for arm in ['full_flat50','quadrant_flat50']:
        candidates=[]
        for lr in [.001,.003]:
            m,mem,bank,acq=train(560,arm,lr,x[mask],y[mask],x[valid],y[valid],SPLITS)
            rr,diag=evaluate(m,mem,bank,x[valid],y[valid],SPLITS,settings(arm))
            candidates.append({'lr':lr,'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq)})
            print('DEV',arm,lr,round(diag['historical']['accuracy']*100,2),flush=True)
        chosen=max(candidates,key=lambda r:(r['diagnostics']['historical']['accuracy'],r['acquisition']))
        selected=[]
        for name in sorted(set(family(s) for s in chosen['methods'])):
            s=max([s for s in chosen['methods'] if family(s)==name],key=lambda s:s['accuracy'])
            selected.append({k:s[k] for k in ['method','scope','gamma','ridge'] if k in s})
        selection[arm]={'lr':chosen['lr'],'readouts':selected,'candidates':candidates}
        save('flat50_selection.json',selection)
    rows=[]
    for seed in [561,562,563]:
        for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for arm in ['full_flat50','quadrant_flat50']:
                c=selection[arm];m,mem,bank,acq=train(seed,arm,c['lr'],x,y,tx,ty,order)
                rr,diag=evaluate(m,mem,bank,tx,ty,order,c['readouts'])
                rows.append({'seed':seed,'order':name,'arm':arm,'lr':c['lr'],'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq),
                             'parameters_allocated':sum(p.numel() for p in m.parameters()),
                             'effective_parameters':sum(p.numel() for p in m.parameters())-(784-int(m.field[0].sum()))*100})
                save('flat50_results.json',rows)
                torch.save({'model':m.state_dict(),'bank_analysis_only':bank,'memory':{'cov':mem.cov,'cross':mem.cross,'direct':mem.direct,'sums':mem.sums,'links':mem.links}},OUT/f'{seed}_{name}_{arm}.pt')
                print('TEST',seed,name,arm,round(100*diag['historical']['accuracy'],2),[(family(s),round(100*s['accuracy'],2)) for s in rr],flush=True)
    save('flat50_completion.json',{'complete':len(rows)==12,'confirmation_backbones':len(rows),'development_backbones':4})


if __name__=='__main__':main()
