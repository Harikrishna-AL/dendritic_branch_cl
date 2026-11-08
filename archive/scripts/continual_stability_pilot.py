"""Prototype-free stability pilots with held-out training validation.

SI-like online weight protection for the hybrid; frozen branch features plus
streaming least-squares readout as a transparent engineering control.
"""
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.eligibility_router_pilot import build,evaluate


def protected(seed,x,y,groups,penalty):
    m=build('reward_router',seed)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
    pars=list(m.parameters())
    anchor=[p.detach().clone() for p in pars]
    importance=[torch.zeros_like(p) for p in pars]
    baseline=0.;start=time.perf_counter();states=[]
    for stage,ix in enumerate(groups):
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            choice=torch.Generator().manual_seed(seed*100000+stage*100+epoch+10000000)
            order=ix[torch.randperm(len(ix),generator=rng)]
            for ids in order.split(128):
                bx,by=x[ids],y[ids]
                before=[p.detach().clone() for p in pars]
                opt.zero_grad(set_to_none=True)
                z,tag,p,j=m(bx,choice)
                reward=(z.detach().argmax(-1)==by).float()
                advantage=reward-baseline;baseline=.95*baseline+.05*float(reward.mean())
                loss=F.cross_entropy(z,by)
                loss.backward()
                with torch.no_grad():
                    for k,par in enumerate(pars):
                        if par.grad is not None:
                            # Online, per-synapse damping; no phase-end consolidation.
                            par.grad.add_(2*penalty*importance[k]*(par-anchor[k]))
                torch.nn.utils.clip_grad_norm_([p for p in pars if p.requires_grad],5);opt.step()
                m.reinforce(bx,tag,advantage,1.)
                with torch.no_grad():
                    for k,par in enumerate(pars):
                        delta=par-before[k]
                        # Accumulated change magnitude: heuristic metaplasticity,
                        # not a claim to implement canonical SI or EWC.
                        importance[k].add_(delta.abs()).clamp_(max=1.)
                        if par is m.router.weight or par is m.router.bias:
                            par.sub_(penalty*.01*importance[k]*(par-anchor[k]))
                        # Slow reference persists; never reset at boundaries.
                        anchor[k].lerp_(par,.001)
        states.append({k:v.clone() for k,v in m.state_dict().items()})
    return m,states,{'seconds':time.perf_counter()-start,'extra_state_bytes':sum(t.numel()*t.element_size() for t in anchor+importance)}


class StableFeatures(nn.Module):
    def __init__(self,kind,seed,width=None):
        super().__init__();torch.manual_seed(seed);self.kind=kind
        # MLP width380 and branches95x4 have nearly identical first-layer cost.
        self.dim=(95 if kind=='branch' else 380) if width is None else width
        self.first=nn.Linear(784,4*self.dim if kind=='branch' else self.dim)
        self.first.requires_grad_(False)
        self.register_buffer('gram',torch.zeros(self.dim+1,self.dim+1,dtype=torch.float64))
        self.register_buffer('cross',torch.zeros(self.dim+1,10,dtype=torch.float64))
        self.register_buffer('head',torch.zeros(self.dim+1,10,dtype=torch.float64))

    def features(self,x):
        b=torch.tanh(self.first(x))
        if self.kind=='branch':
            # One response per neuron, selected by current response. No route table.
            b=b.reshape(-1,self.dim,4).max(-1).values
        return torch.cat([b.double(),torch.ones(len(x),1,dtype=torch.float64)],1)

    @torch.no_grad()
    def observe(self,x,y):
        z=self.features(x);target=F.one_hot(y,10).double()
        self.gram.add_(z.T@z);self.cross.add_(z.T@target)

    @torch.no_grad()
    def solve(self,ridge):
        penalty=torch.eye(self.dim+1,dtype=torch.float64)*ridge;penalty[-1,-1]=0
        self.head.copy_(torch.linalg.solve(self.gram+penalty,self.cross))

    def forward(self,x):return self.features(x)@self.head


def stable(kind,seed,x,y,groups,ridge,width=None):
    m=StableFeatures(kind,seed,width);states=[];start=time.perf_counter()
    for stage,ix in enumerate(groups):
        # Explicit one pass; no task boundary inputs. Solve after EACH minibatch.
        rng=torch.Generator().manual_seed(seed*100000+stage*100)
        order=ix[torch.randperm(len(ix),generator=rng)]
        for ids in order.split(512):m.observe(x[ids],y[ids]);m.solve(ridge)
        states.append({k:v.clone() for k,v in m.state_dict().items()})
    return m,states,{'seconds':time.perf_counter()-start,'extra_state_bytes':m.gram.numel()*8+m.cross.numel()*8}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'development_seed':80,'final_seeds':[81,82,83],'orders':['forward','reverse'],
              'development':'50k historical training images for sequential fit;10k stratified training validation for configuration choice. No official test tuning.',
              'protected':'Reward-routed hybrid, two epochs per pair, online anchor EMA .001 and cumulative absolute update importance. Penalties0/1/10/100. Extra2x parameter state; no boundary events. Heuristic protection, not canonical EWC/SI or biologically validated.',
              'stable_features':'Random tanh380 inputs; branch95neuronsx4 with response max selection, or MLP380 outputs. Frozen from initialization; no pretraining. Single pass through stream, batch512. Accumulated Gram and feature/target cross-products give a ridge classifier, solved after EACH batch, not after task boundaries. Additional full covariance explicitly counted.',
              'feedback':'Supervised labels during training; all10 classes at inference. No task IDs, prototype/examples/configuration banks or test feedback. Distinct engineering learning rule, not a proof of three-factor biological learning.',
              'limits':'Different capacities and compute across stable-feature arms, and different epoch count vs hybrid; controls not called parameter/storage matched. Predefined ranges only; developmental comparisons exploratory.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=torch.Generator().manual_seed(80)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False
    groups=[torch.nonzero(mask&((y==a)|(y==b)),as_tuple=True)[0] for a,b in SPLITS]
    dev=[];selected={}
    for arm,values in [('protected',[0.,1.,10.,100.]),('branch',[.1,1.,10.]),('mlp',[.1,1.,10.])]:
        for value in values:
            m,_,stats=protected(80,x,y,groups,value) if arm=='protected' else stable(arm,80,x,y,groups,value)
            score=evaluate(m,x[val],y[val]);row={'arm':arm,'value':value,**score,**stats};dev.append(row)
            print('dev',arm,value,round(score['accuracy']*100,2),flush=True)
        selected[arm]=max([r for r in dev if r['arm']==arm],key=lambda r:r['accuracy'])
    (out/'selection.json').write_text(json.dumps({'candidates':dev,'selected':selected},indent=2))
    rows=[]
    for order in ['forward','reverse']:
        pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
        groups=[torch.nonzero((y==a)|(y==b),as_tuple=True)[0] for a,b in pairs]
        for seed in [81,82,83]:
            for arm in ['protected','branch','mlp']:
                value=selected[arm]['value']
                m,states,stats=protected(seed,x,y,groups,value) if arm=='protected' else stable(arm,seed,x,y,groups,value)
                stages=[]
                for state in states:m.load_state_dict(state);stages.append(evaluate(m,tx,ty))
                old=[i for i,pair in enumerate(SPLITS) if pair!=pairs[-1]]
                row={'order':order,'seed':seed,'arm':arm,'value':value,**stats,'accuracy':stages[-1]['accuracy'],
                     'old_accuracy':st.mean(stages[-1]['pairs'][i] for i in old),'stages':stages,
                     'model_numeric_bytes':sum(v.numel()*v.element_size() for v in m.state_dict().values())}
                rows.append(row);torch.save(m.state_dict(),out/f'{order}_{seed}_{arm}.pt')
                (out/'results.json').write_text(json.dumps(rows,indent=2))
                print(order,seed,arm,round(row['accuracy']*100,2),'old',round(row['old_accuracy']*100,2),flush=True)
    lines=['# Prototype-free continual stability pilot','',
           'All models start before observing any task. No task identity or boundaries passed to learning '
           'rules; no old-image replay or prototype/configuration archive. Supervised training labels used. '
           'Engineering controls have extra synaptic/statistical state, fully disclosed.', '',
           '| Method | Accuracy mean ± SD | Older pairs | Numeric state bytes |', '|---|---:|---:|---:|']
    for arm in ['protected','branch','mlp']:
        rs=[r for r in rows if r['arm']==arm];a=[r['accuracy']*100 for r in rs]
        lines.append(f'| {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | {100*st.mean(r["old_accuracy"] for r in rs):.2f}% | {rs[0]["model_numeric_bytes"]:,} |')
    lines+=['','The protected model stores two extra parameter-sized arrays (anchors and importance) '
            'beyond model_numeric_bytes. Stable-feature models include Gram/cross/head tensors in '
            'model_numeric_bytes. Full covariance is additional persistent statistical memory, not a '
            'short-term activity trace. Frozen branches use max response per neuron; the MLP uses all380 '
            'random features. A standard streaming ridge solution is an engineering baseline, not a '
            'novel dendritic learning algorithm or biologically plausible local update.', '',
            'The frozen feature transform and cumulative sufficient statistics make final readout '
            'approximately order-independent (up to numerical summation), so task-order robustness '
            'alone does not establish a learned protection mechanism. Every solve happens after a '
            'minibatch, never on a supplied boundary. Development choices used only training validation. '
            'All test state remains frozen. Detailed protocol, selections, states and scores saved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:12]),flush=True)


if __name__=='__main__':main()
