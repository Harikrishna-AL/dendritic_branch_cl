"""Paired training-cost ablations of the original trace CE-only branch model."""
import argparse
import json
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.matched_numenta_comparison import construct, predictions, ROOT
from bdh_cl.scripts.numenta_split_mnist_comparison import state_hash

VARIANTS=['original','cached_exact','chunked_exact','feedback16','feedback16_every4','feedback16_every8']

def cached_step(model,bx,by,step,variant):
    branches=torch.tanh(model.first(bx).reshape(-1,168,4))
    hidden=(branches*model.gates().detach().clone()).sum(-1)
    logits=model.head(hidden)
    period=4 if variant.endswith('every4') else 8 if variant.endswith('every8') else 1
    feedback=None
    if step%period==0:
        with torch.no_grad():
            count=min(16,len(bx)) if variant.startswith('feedback16') else len(bx)
            b=branches[:count].detach();h=hidden[:count].detach();l=logits[:count].detach()
            chunk=21 if variant=='chunked_exact' else 168
            parts=[]
            for start in range(0,168,chunk):
                delta=b[:,start:start+chunk]-h[:,start:start+chunk,None]
                candidates=l[:,None,None,:]+delta[:,:,:,None]*model.head.weight.T[None,start:start+chunk,None,:]
                target=by[:count,None,None].expand(candidates.shape[:-1]).reshape(-1)
                errors=F.cross_entropy(candidates.reshape(-1,10),target,reduction='none').reshape(candidates.shape[:-1])
                parts.append(errors.mean(0))
            feedback=torch.cat(parts,dim=0)
    return F.cross_entropy(logits,by),feedback


def original_step(model,bx,by):
    with torch.no_grad():
        _,candidates=model(bx,counterfactual=True)
        targets=by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
        errors=F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1])
        feedback=errors.mean(0)
    return F.cross_entropy(model(bx),by),feedback


def train(variant,seed,order,x,y,indices):
    m=construct('branch_ce',seed);opt=torch.optim.Adam(m.parameters(),lr=.001)
    routes=torch.zeros(5,168,dtype=torch.uint8)
    steps=presentations=0;seconds=0.;feedback_updates=0
    for stage,g in enumerate(order):
        m.train()
        for epoch in range(2):
            gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ids in indices[g][torch.randperm(len(indices[g]),generator=gen)].split(128):
                start=time.perf_counter()
                bx,by=x[ids],y[ids];opt.zero_grad(set_to_none=True)
                if variant=='original':loss,feedback=original_step(m,bx,by)
                else:loss,feedback=cached_step(m,bx,by,steps,variant)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                if feedback is not None:m.observe(feedback);feedback_updates+=1
                steps+=1;presentations+=len(ids);seconds+=time.perf_counter()-start
        routes[g]=m.gates().argmax(-1).to(torch.uint8)
    assert presentations==110000
    return m,routes,dict(training_seconds=seconds,presentations=presentations,steps=steps,feedback_updates=feedback_updates)

@torch.no_grad()
def score(m,routes,x,y,contexts):
    before=state_hash(m);rr=routes.clone()
    scores={mode:float((predictions(m,'branch_ce',x,contexts,routes,mode,y//2)==y).float().mean()) for mode in ['inferred','oracle','current']}
    assert state_hash(m)==before and torch.equal(routes,rr)
    return scores

@torch.no_grad()
def sparse_predict(m,x,contexts,routes):
    out=[]
    for start in range(0,len(x),256):
        bx=x[start:start+256];tasks=torch.cdist(bx,contexts).argmin(-1)
        pred=torch.empty(len(bx),dtype=torch.long)
        w=m.first.weight.reshape(168,4,784);b=m.first.bias.reshape(168,4)
        for g in tasks.unique():
            selected=routes[g].long();neurons=torch.arange(168)
            h=torch.tanh(F.linear(bx[tasks==g],w[neurons,selected],b[neurons,selected]))
            pred[tasks==g]=m.head(h).argmax(-1)
        out.append(pred)
    return torch.cat(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(variants=VARIANTS,seed_calibration=113,seeds=[101,102,103],orders=['forward','reverse'],lr=.001,epochs_per_task=2,batch_size=128,parameters=529210,auxiliary_loss=False,trace_decay=.95,feedback_schedule='Every4/8 uses global batch index, same.95decay per OBSERVED update. No EMA compensation. Prespecified approximations.',timing='Serial CPU2threads, accumulated minibatch training wall time including indexing, feedback, backward,Adam andtraceupdates. Excludes acquisition,evaluation,checkpointwrites. Not GPU/M4 or FLOP benchmark.',selection='All6variants evaluated on validation seed113forward before final tests. Approximate acceptance: mean autonomous and oracle drops each <=1pp relative original across3seedmeans, plus pairedseed results reported. Descriptive tolerance, not statistical equivalence proof.',inference='Common input-prototype router with saved choices. Exact selected-weight inference also benchmarked.',memory='Only max counterfactual-logit tensor size reported, not peak RAM. Numerical model/state unchanged.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    raw=load_raw(str(ROOT/'data_cache/mnist'))
    x=(torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=(torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=np.random.default_rng(20261101);vi=np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==c))[:500] for c in range(10)]);ti=np.setdiff1d(np.arange(len(y)),vi)
    np.savez(out/'split_indices.npz',train=ti,validation=vi);assert not np.intersect1d(ti,vi).size
    vx,vy=x[vi],y[vi];x,y=x[ti],y[ti];indices=[torch.where(y//2==g)[0] for g in range(5)];contexts=torch.stack([x[i].mean(0) for i in indices]);torch.save(contexts,out/'contexts.pt')
    # Warm up CPU kernels before measured runs. Reset model/optimizer for actual runs.
    warm=construct('branch_ce',999)
    for _ in range(5):
        loss,e=cached_step(warm,x[:128],y[:128],0,'cached_exact');loss.backward();warm.zero_grad(set_to_none=True)
    trials=[]
    for variant in VARIANTS:
        m,r,t=train(variant,113,list(range(5)),x,y,indices);scores=score(m,r,vx,vy,contexts)
        trials.append(dict(variant=variant,scores=scores,**t));print('VALIDATION',variant,{k:round(v*100,2) for k,v in scores.items()},round(t['training_seconds'],3),flush=True)
    (out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
    (out/'frozen_settings.json').write_text(json.dumps(protocol,indent=2))
    rows=[];infer=[]
    for seed in [101,102,103]:
        for ordername,order in [('forward',list(range(5))),('reverse',list(range(4,-1,-1)))]:
            offset=(seed-101+(ordername=='reverse'))%len(VARIANTS)
            variants=VARIANTS[offset:]+VARIANTS[:offset]
            states={}
            for variant in variants:
                m,r,t=train(variant,seed,order,x,y,indices);scores=score(m,r,tx,ty,contexts)
                row=dict(variant=variant,seed=seed,order=ordername,scores=scores,**t);rows.append(row)
                states[variant]=({k:v.clone() for k,v in m.state_dict().items()},r.clone())
                torch.save(dict(model=m.state_dict(),routes=r),out/f'{seed}_{ordername}_{variant}.pt')
                (out/'results.json').write_text(json.dumps(rows,indent=2))
                print('FINAL',seed,ordername,variant,{k:round(v*100,2) for k,v in scores.items()},round(t['training_seconds'],3),flush=True)
            base,br=states['original']
            for variant in ['cached_exact','chunked_exact']:
                state,rr=states[variant]
                row=next(z for z in rows if z['seed']==seed and z['order']==ordername and z['variant']==variant)
                row['bitwise_model_equal']=all(torch.equal(state[k],base[k]) for k in base)
                row['routes_equal']=torch.equal(rr,br)
                row['max_tensor_difference']=max(float((state[k]-base[k]).abs().max()) for k in base)
            m=construct('branch_ce',seed);m.load_state_dict(base);m.eval()
            before=state_hash(m)
            dense=predictions(m,'branch_ce',tx,contexts,br)
            sparse=sparse_predict(m,tx,contexts,br)
            assert torch.equal(dense,sparse)
            times={'dense':[],'selected_weights':[]}
            for _ in range(9):
                for mode in (['dense','selected_weights'] if _%2==0 else ['selected_weights','dense']):
                    start=time.perf_counter()
                    if mode=='dense':predictions(m,'branch_ce',tx,contexts,br)
                    else:sparse_predict(m,tx,contexts,br)
                    times[mode].append(time.perf_counter()-start)
            assert state_hash(m)==before
            infer.append(dict(seed=seed,order=ordername,identical_predictions=True,seconds={k:st.median(v) for k,v in times.items()}))
            (out/'inference.json').write_text(json.dumps(infer,indent=2));(out/'results.json').write_text(json.dumps(rows,indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
