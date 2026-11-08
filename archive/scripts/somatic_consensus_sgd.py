"""Compact strict-persistent-budget control: SGD without momentum/history."""
import hashlib
import json
import statistics as st
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.somatic_consensus import OUT,train,evaluate,family,save

ARMS=[f'patch5_independent_keep{k}' for k in [24,17,12]]


def main():
    torch.set_num_threads(2)
    protocol={'purpose':'Count optimizer history as persistent numerical state and test a compact model with no such history.',
        'optimizer':'SGD without momentum, hence no optimizer state arrays. Gradients/workspace temporary and excluded for all methods.',
        'arms':ARMS,'validation':'Seed70050k/10k,LR.03/.1/.3. ChooseONEcommonLR bymeanvalidation local/no-feedback at matchedtrainingkeep. Per-family inferencekeep17/12 chosenvalidation. Directridge fromprimaryfrozenmanifest.',
        'confirmation':'Seeds701-710,bothorders,60runs. Samepairedcohort asAdam,notindependentreplication. No retuning on confirmation. Supplementalcontrol motivated by counting optimizerstate, not an independentprimary study.',
        'other_settings':'Same patches/initialization/batches/EMA/localmemory/top2feedback asprimary. No replay/pretraining/taskIDs/bankautomaticreadout.',
        'budget':'18520bytes model+localmemory+coordinates+usefulnesstraces, plus0optimizerhistory. CountretaineddirectRLS18944B; full model+Adam estimate separately.'}
    save('sgd_protocol.json',protocol)
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(700)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;fx,fy,vx,vy=x[mask],y[mask],x[valid],y[valid]
    dev=[]
    for lr in [.03,.1,.3]:
        for arm in ARMS:
            m,mem,bank,acq,seconds=train(700,arm,lr,fx,fy,vx,vy,SPLITS,optimizer_kind='sgd')
            rr,diag,direct=evaluate(m,mem,bank,vx,vy,SPLITS)
            dev.append({'arm':arm,'lr':lr,'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq)})
            save('sgd_development.json',dev)
            print('DEV',lr,arm,round(100*next(r['accuracy'] for r in rr if r['family']=='baseline'),2),flush=True)
    candidates=[]
    for lr in [.03,.1,.3]:
        score=st.mean(next(r['accuracy'] for r in d['methods'] if r['cue']=='local' and not r['feedback'] and r['rank']=='trace' and r['keep']==int(d['arm'].split('keep')[-1])) for d in dev if d['lr']==lr)
        candidates.append({'lr':lr,'score':score})
    lr=max(candidates,key=lambda r:(r['score'],-r['lr']))['lr'];readouts={}
    for arm in ARMS:
        d=next(d for d in dev if d['arm']==arm and d['lr']==lr);ss=[]
        for name in sorted(set(r['family'] for r in d['methods'])):
            r=max([r for r in d['methods'] if r['family']==name],key=lambda r:(r['accuracy'],r['keep']))
            ss.append({k:r[k] for k in ['cue','keep','rank','feedback']})
        readouts[arm]=ss
    primary=json.loads((OUT/'selection.json').read_text())
    selection={'lr':lr,'candidates':candidates,'readouts':readouts,'direct_ridge':primary['direct_ridge'],
               'frozen_before_sgd_confirmation':True,'protocol_sha256':hashlib.sha256((OUT/'sgd_protocol.json').read_bytes()).hexdigest()}
    save('sgd_selection.json',selection);print('SELECTED',lr,flush=True)
    rows=[]
    for seed in range(701,711):
        for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for arm in ARMS:
                m,mem,bank,acq,seconds=train(seed,arm,lr,x,y,tx,ty,order,optimizer_kind='sgd')
                rr,diag,direct=evaluate(m,mem,bank,tx,ty,order,direct_ridges=[selection['direct_ridge']])
                rows.append({'seed':seed,'order':name,'arm':arm,'optimizer':'sgd_without_momentum','lr':lr,'methods':rr,
                             'diagnostics':diag,'direct':direct,'acquisition':st.mean(acq),'optimizer_numeric_bytes':0})
                save('sgd_results.json',rows)
                torch.save({'model':m.state_dict(),'memory':vars(mem),'historical_masks_diagnostic_only':bank},OUT/f'sgd_{seed}_{name}_{arm}.pt')
                print('TEST',seed,name,arm,'base',round(100*next(r['accuracy'] for r in rr if r['family']=='baseline'),2),'acq',round(100*st.mean(acq),2),flush=True)
    save('sgd_completion.json',{'complete':len(rows)==60,'confirmation_runs':len(rows),'development_runs':len(dev)})


if __name__=='__main__':main()
