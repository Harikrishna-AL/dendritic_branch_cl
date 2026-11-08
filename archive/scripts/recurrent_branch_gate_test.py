"""Prospective supervised training of label-free recurrent branch retrieval.

Backbone, gate updaters and controls learn only current sequential examples.
Inference updates only per-image gate activations, never long-term weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts import controlled_branch_depth as b
from bdh_cl.scripts import online_proxy_branch_retrieval as old
from bdh_cl.scripts.overnight_multicue_pilot import compress, Direct
from bdh_cl.data.mnist import load_raw, SPLITS

ROOT=Path('results/recurrent_branch_gate_20261026')
FAMILIES=['one_pass','one_pass_extra','recurrent_full','recurrent_no_neighbor','recurrent_local']
LRS=[.001,.003]
TRAIN_GATE_BATCH=32

def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))

class Updater(nn.Module):
    def __init__(self,seed):
        super().__init__()
        torch.manual_seed(seed)
        self.identity=nn.Parameter(torch.randn(25,4)*.05)
        self.rule=nn.Sequential(nn.Linear(42,16),nn.Tanh(),nn.Linear(16,4))
        nn.init.normal_(self.rule[-1].weight,std=.01)
        nn.init.zeros_(self.rule[-1].bias)

    def forward(self,response,q,initial,weight,bias,steps,communication):
        a=initial
        for _ in range(steps):
            g=a.softmax(-1)
            h=(g*response).sum(-1)
            logits=F.linear(h,weight,bias)
            neighbor=(g.roll(1,1)+g.roll(-1,1))*.5 if communication=='full' else torch.zeros_like(g)
            probability=logits.softmax(-1) if communication!='local' else torch.zeros_like(logits)
            features=torch.cat([response,g,neighbor,q[:,None,:].expand(-1,25,-1),
                                probability[:,None,:].expand(-1,25,-1),
                                self.identity[None].expand(len(q),-1,-1)],-1)
            delta=self.rule(features).tanh()
            a=a+delta-delta.mean(-1,keepdim=True)
        gates=a.softmax(-1)
        return F.linear((gates*response).sum(-1),weight,bias),gates

def config(family,lr):
    return {'family':family,'lr':lr,'steps':1 if family.startswith('one_pass') else 3,
            'updates':3 if family=='one_pass_extra' else 1,
            'communication':'full' if family in ['one_pass','one_pass_extra','recurrent_full'] else ('global' if family=='recurrent_no_neighbor' else 'local')}

def initial_gate(route,batch,mode):
    a=torch.zeros(batch,25,4)
    if mode=='final':a.scatter_(-1,route[None,:,None].expand(batch,-1,1),1.)
    return a

@torch.no_grad()
def predictions(m,router,x,setting,steps,initial,hard=False):
    # No y/task/bank argument, and no gradients/weight adaptation at inference.
    answer=[]
    route=m.routes()[0]
    for bx in x.split(256):
        response=torch.tanh(m.layers[0](bx)).reshape(-1,25,4)
        q=torch.tensor(compress(old.keys(bx),16))
        logits,g=router(response,q,initial_gate(route,len(bx),initial),
                        m.head.weight,m.head.bias,steps,setting['communication'])
        if hard:
            logits=m.head(response.gather(-1,g.argmax(-1)[:,:,None]).squeeze(-1))
        answer.append(logits.argmax(-1))
    return torch.cat(answer)

def metric(pred,y,order):
    p=[float((pred[torch.isin(y,torch.tensor(pair))]==y[torch.isin(y,torch.tensor(pair))]).float().mean()) for pair in order]
    return {'accuracy':float((pred==y).float().mean()),'pair_accuracy':p,
            'old_accuracy':st.mean(p[:-1]),'new_accuracy':p[-1]}

def train(seed,x,y,order,settings,out,label,probes=None):
    m=b.make(seed,'branch1_proxy')
    opt=torch.optim.Adam(m.parameters(),lr=.003)
    routers=[Updater(seed+10000) for _ in settings]
    optimizers=[torch.optim.Adam(r.parameters(),lr=s['lr']) for r,s in zip(routers,settings)]
    local=old.Local(.995)
    direct={d:Direct(d) for d in [16,64]}
    bank=[];curves=[];backbone_seconds=0.;router_seconds=[0.]*len(settings);steps=0;presentations=0;router_presentations=0
    init_rng=torch.Generator().manual_seed(seed+20000)
    started=time.perf_counter()
    for stage,pair in enumerate(order):
        ids=torch.nonzero(torch.isin(y,torch.tensor(pair))).flatten()
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):
                bx,by=x[ix],y[ix]
                q64=old.keys(bx);route=m.routes()[0]
                local.update(q64,route.numpy().astype(np.uint8))
                for obj in direct.values():obj.update(q64,by)
                # All sidecars see identical PRE-update responses/head and labels.
                with torch.no_grad():
                    response=torch.tanh(m.layers[0](bx[:TRAIN_GATE_BATCH])).reshape(-1,25,4)
                    q=torch.tensor(compress(q64[:TRAIN_GATE_BATCH],16))
                    initial=initial_gate(route,len(q),'final')
                    initial*=torch.randint(0,2,(len(q),1,1),generator=init_rng)
                    initial+=torch.randn(initial.shape,generator=init_rng)*.25
                    weight=m.head.weight.detach().clone();bias=m.head.bias.detach().clone()
                for j,(r,op,s) in enumerate(zip(routers,optimizers,settings)):
                    start=time.perf_counter()
                    for _ in range(s['updates']):
                        op.zero_grad(set_to_none=True)
                        logits,_=r(response,q,initial,weight,bias,s['steps'],s['communication'])
                        loss=F.cross_entropy(logits,by[:len(q)])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(r.parameters(),5.)
                        op.step()
                    router_seconds[j]+=time.perf_counter()-start
                start=time.perf_counter();b.step(m,opt,bx,by,'branch1_proxy');backbone_seconds+=time.perf_counter()-start
                steps+=1;presentations+=len(by);router_presentations+=len(q)
        bank.append(m.routes()[0].clone())
        if probes is not None:
            px,py=probes
            probe_rows=[]
            for r,s in zip(routers,settings):
                for initial in ['final','uniform']:
                    pred=predictions(m,r,px,s,s['steps'],initial)
                    probe_rows.append({'family':s['family'],'initial':initial,**metric(pred,py,order)})
            curves.append({'stage_for_analysis_only':stage,'methods':probe_rows})
        print('STAGE',label,seed,stage+1,round(time.perf_counter()-started,1),flush=True)
    timing={'all_candidate_seconds':time.perf_counter()-started,'backbone_seconds':backbone_seconds,
            'router_seconds':router_seconds,'backbone_steps':steps,'backbone_presentations':presentations,
            'router_unique_current_presentations':router_presentations,
            'router_optimizer_steps':[steps*s['updates'] for s in settings],
            'router_unrolled_calls':[steps*s['updates']*s['steps'] for s in settings],
            'diagnostic_probe_curves':curves}
    torch.save({'backbone':m.state_dict(),'routers':[r.state_dict() for r in routers],
                'settings':settings,'bank_for_oracle_analysis_only':torch.stack(bank)},out/f'{label}_{seed}.pt')
    return m,routers,local,direct,torch.stack(bank),timing

def digest(m,routers):
    h=hashlib.sha256()
    for module in [m,*routers]:
        for v in module.state_dict().values():h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def evaluate(m,routers,local,direct,bank,x,y,order,settings,full):
    before=digest(m,routers);rows=[]
    for r,s in zip(routers,settings):
        counts=[1] if s['steps']==1 else ([1,3,5] if full else [3])
        for initial in ['final','uniform']:
            for steps in counts:
                pred=predictions(m,r,x,s,steps,initial)
                rows.append({**s,'initial':initial,'inference_steps':steps,'hard':False,**metric(pred,y,order)})
        if full:
            pred=predictions(m,r,x,s,s['steps'],'final',hard=True)
            rows.append({**s,'initial':'final','inference_steps':s['steps'],'hard':True,**metric(pred,y,order)})
    q=np.concatenate([old.keys(z) for z in x.split(512)])
    response=torch.cat([torch.tanh(m.layers[0](z)).reshape(-1,25,4) for z in x.split(512)])
    def classify(route):return m.head(response.gather(-1,torch.tensor(route.astype(np.int64))[:,:,None]).squeeze(-1)).argmax(-1)
    current=np.broadcast_to(m.routes()[0].numpy(),(len(x),25)).copy()
    baseline={'final_hard':metric(classify(current),y,order),
              'final_soft':metric(m.head((initial_gate(m.routes()[0],len(x),'final').softmax(-1)*response).sum(-1)).argmax(-1),y,order),
              'uniform_soft':metric(m.head(response.mean(-1)).argmax(-1),y,order),
              'local_average':metric(classify(local.routes(q,0)),y,order)}
    for d,obj in direct.items():baseline[f'direct_ridge_{d}']=metric(torch.tensor(obj.predict(q)),y,order)
    # Label-assisted historical configuration is computed AFTER all predictions.
    task=torch.empty(len(y),dtype=torch.long)
    for i,p in enumerate(order):task[torch.isin(y,torch.tensor(p))]=i
    baseline['historical_oracle']=metric(classify(bank.numpy()[task.numpy()]),y,order)
    assert digest(m,routers)==before
    return rows,baseline,{'state_immutable':True,'oracle_bank_used_only_after_automatic_predictions':True}

def tensors(raw,which):
    return (torch.tensor(raw[which+'_images'].copy(),dtype=torch.float32).flatten(1)/255,
            torch.tensor(raw[which+'_labels'].copy(),dtype=torch.long))

def verify():
    torch.manual_seed(42);r=Updater(42);response=torch.randn(3,25,4);q=torch.randn(3,16)
    a=torch.zeros(3,25,4);w=torch.randn(10,25);bias=torch.randn(10)
    initial=a.clone();log,g=r(response,q,a,w,bias,3,'full')
    assert torch.equal(a,initial) and torch.allclose(g.sum(-1),torch.ones(3,25),atol=1e-6)
    assert torch.allclose(log,F.linear((g*response).sum(-1),w,bias),atol=1e-6)
    F.cross_entropy(log,torch.tensor([0,1,2])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters())
    assert sum(p.numel() for p in r.parameters())==856
    return {'gate_probabilities_normalized':True,'initialization_not_mutated':True,
            'mixture_readout_matches':True,'all_updater_gradients_finite':True,'parameters':856}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['development','confirmation'])
    args=parser.parse_args();torch.set_num_threads(2)
    ROOT.mkdir(parents=True,exist_ok=True)
    raw=load_raw('data_cache/mnist');x,y=tensors(raw,'train')
    if args.mode=='development':
        assert not (ROOT/'development.json').exists(),'Refusing overwrite'
        (ROOT/'source.py').write_text(Path(__file__).read_text())
        save(ROOT/'verification.json',verify())
        save(ROOT/'protocol.json',{
            'scope':'MNIST five digit pairs, all10outputs. Prospective sidecar training during sequential backbone learning; no old-task replay, task IDs, boundary input, saved route bank or inference labels. Current labels train updater/backbone/direct controls.',
            'backbone':'Existing25x4 tanhbranch, gradient-usefulness proxy, CE,Adam.003,clip5,2epochs/currentpair,batch128. Gate sidecars never affect backbone updates; PRE-update responses/head detached.',
            'updater':'856parameters: shared42->16tanh->4boundedresidual gate update plus25x4neuron embeddings. Inputs localresponses4,gates4,neighbor gate mean4,pixelkey16,globalprediction10,embedding4. No backprop/CE/weight changes during inference. Long-term updater trained by global backprop through recurrent unroll; not fully biological local learning.',
            'training':'Each sidecar sees same first32current examples/minibatch; random50/50current-route/uniform initialization plus Gaussian.25noise, no per-phase resets. Sidecar labels only those currentexamples. Onepass_extra gets3optimizerupdates/currentbatch; recurrent3 gets1update through3unrolledcalls, approximating matched recurrent-cell call budget, not identical FLOPs.',
            'families':FAMILIES,'learning_rates':LRS,
            'development':'Seeds810/811, stratified50kfit/10ktrainingvalidation, forwardorder. No official test metrics. Per-family LR selected by mean all-ten-class accuracy at trained horizon, final-route initialization, softmixture. All candidates/initial controls retained.',
            'confirmation':'Fresh812-816,forward/reverse,full60k/10k. Primary recurrent_full3 vsone_pass1 andone_pass_extra1; secondary no-neighbor/local variants,1/3/5iteration curves,uniforminit,hardgateprojection. All predeclared conditions retained; no test tuning even if negative.',
            'primary_readout':'Soft mixture over branches; hard projection is secondary. Finalhard/globalroute, finalsoft, uniformsoft, localaverage, directridge16/64 controls. Historicaloracle label-assisted diagnostic only.',
            'cost':'856params3424Bextraweights, Adam moments during training,100floatper-image fast gates400Btemporary plus recurrentworkspace. All100branchresponses computed once/input and reused, not fullbackbone pass eachround. No guaranteeconvergence; finite unroll called recurrentrefinement, not proven attractor.',
            'note':'load_raw returns both splits, but development uses only training arrays/labels; no official-test feedback selects settings.'})
        settings=[config(f,lr) for f in FAMILIES for lr in LRS];records=[]
        for seed in [810,811]:
            rng=torch.Generator().manual_seed(seed)
            valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
            mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False
            m,rs,local,direct,bank,timing=train(seed,x[mask],y[mask],SPLITS,settings,ROOT,'dev')
            rr,base,audit=evaluate(m,rs,local,direct,bank,x[valid],y[valid],SPLITS,settings,False)
            records.append({'seed':seed,'methods':rr,'baselines':base,'timing':timing,'audit':audit})
            save(ROOT/'development.json',records)
            print('DEV',seed,[(r['family'],r['lr'],r['initial'],round(100*r['accuracy'],2)) for r in rr],flush=True)
        selected=[]
        for family in FAMILIES:
            candidates=[]
            for lr in LRS:
                values=[r['accuracy'] for record in records for r in record['methods'] if r['family']==family and r['lr']==lr and r['initial']=='final']
                candidates.append((st.mean(values),lr))
            selected.append(config(family,max(candidates,key=lambda t:(t[0],-t[1]))[1]))
        save(ROOT/'selection.json',selected);print('LOCKED',selected,flush=True)
    else:
        assert not (ROOT/'confirmation.json').exists(),'Refusing overwrite'
        settings=json.loads((ROOT/'selection.json').read_text())
        save(ROOT/'selection_lock.json',{'sha256':hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest(),'settings':settings,'seeds':list(range(812,817))})
        (ROOT/'confirmation_source.py').write_text(Path(__file__).read_text())
        save(ROOT/'confirmation_addendum.json',{'locked_before_confirmation':True,'change':'Record intermediate automatic accuracy curves on fixed100testexamples/class, solely as diagnostics. No training/selection changes; no intermediate feedback to models or settings.','rationale':'Development around20% all-class accuracy motivates distinguishing current acquisition from retention. No claim this is updater-only forgetting; backbone/head also evolve.'})
        tx,ty=tensors(raw,'test');records=[]
        probe_ids=torch.cat([torch.nonzero(ty==c).flatten()[:100] for c in range(10)])
        for seed in range(812,817):
            for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
                m,rs,local,direct,bank,timing=train(seed,x,y,order,settings,ROOT,name,probes=(tx[probe_ids],ty[probe_ids]))
                rr,base,audit=evaluate(m,rs,local,direct,bank,tx,ty,order,settings,True)
                records.append({'seed':seed,'order':name,'methods':rr,'baselines':base,'timing':timing,'audit':audit})
                save(ROOT/'confirmation.json',records)
                print('RESULT',seed,name,[(r['family'],round(100*r['accuracy'],2)) for r in rr if r['initial']=='final' and r['inference_steps']==r['steps'] and not r['hard']],flush=True)
        save(ROOT/'completion.json',{'complete':len(records)==10,'backbones':len(records),'router_training_runs':len(records)*len(settings),'automatic_router_readouts':sum(len(r['methods']) for r in records)})

if __name__=='__main__':main()
