"""Alternative literal top-two weighted consensus, with meaningful row scaling."""
import json
import hashlib
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts import complementary_exchange as base
from bdh_cl.scripts.protected_feedback import OUT,SOURCE,ARMS,THRESHOLDS,evaluate,save


def main():
    torch.set_num_threads(2)
    protocol={'purpose':'Resolve top-two feedback ambiguity: a probability-weighted common signal versus separate class-conditioned routes.',
        'equation':'feedback_direction=sum_c_in_top2 softmax(top2_logits)_c * W_hat[c]; branch_score=response*feedback_direction; choosewithinbottomup top2. W_hatrowL2normalized. Controlusesrawrows.',
        'gate':'Samefixedtaugrid0/.25/.5/1/2/4/8/always,selectvalidationpermode/arm.',
        'scope':'Supplemental implementation check on same frozen cohort, after primary separate-class diagnostic. Not independent confirmation. No neural retraining of officialtestbackbones. Normonlyappliesfeedback, classifierunchanged.',
        'ordinary':'Same singleton-only swaps and active-count preservation as primary.'}
    save('consensus_protocol.json',protocol)
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(600)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
    settings=[{'mode':mode,'tau':tau} for mode in ['mixed_normalized','mixed_raw'] for tau in THRESHOLDS]
    selected={};dev=[]
    for arm in ARMS:
        m,mem,bank,acq,seconds=base.train(600,arm,.001,x[mask],y[mask],x[valid],y[valid],SPLITS)
        rr=evaluate(m,mem,x[valid],y[valid],SPLITS,settings)
        dev.append({'arm':arm,'methods':rr});selected[arm]={}
        for mode in ['mixed_normalized','mixed_raw']:
            chosen=max([r for r in rr if r['mode']==mode],key=lambda r:(r['accuracy'],-(float('inf') if r['tau'] is None else r['tau'])))
            selected[arm][mode]={'mode':mode,'tau':chosen['tau'],'validation_accuracy':chosen['accuracy']}
        print('DEV',arm,selected[arm],flush=True)
    save('consensus_development.json',dev)
    save('consensus_selection.json',{'per_arm':selected,'frozen_before_own_confirmation':True,'protocol_sha256':hashlib.sha256((OUT/'consensus_protocol.json').read_bytes()).hexdigest()})
    prior=json.loads((OUT/'results.json').read_text());rows=[]
    for record in prior:
        arm,seed,name=record['arm'],record['seed'],record['order']
        state=torch.load(SOURCE/f'{seed}_{name}_{arm}.pt',weights_only=True)
        m=base.Model(seed,arm);m.load_state_dict(state['model']);mem=base.Memory()
        for key,value in state['memory'].items():setattr(mem,key,value)
        order=SPLITS if name=='forward' else SPLITS[::-1]
        rr=evaluate(m,mem,tx,ty,order,settings)
        choices=[]
        for mode in ['mixed_normalized','mixed_raw']:
            choice=selected[arm][mode]
            choices.append(next(r for r in rr if r['mode']==mode and r['tau']==choice['tau']))
        assert abs(rr[0]['before_accuracy']-record['baseline']['accuracy'])<1e-7
        rows.append({'seed':seed,'order':name,'arm':arm,'baseline':record['baseline'],'selected':choices,'methods':rr,'model_and_memory_unchanged':True})
        save('consensus_results.json',rows)
    save('consensus_completion.json',{'complete':len(rows)==60,'frozen_models':len(rows),'development_reproductions':3})


if __name__=='__main__':main()
