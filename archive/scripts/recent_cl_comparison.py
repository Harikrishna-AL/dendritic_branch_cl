"""Explicitly adapted OCAR (2025) and RFE (2026), shared Split MNIST data.

Not original-paper reproductions or a verified SOTA leaderboard.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts import strong_replay_comparison as replay

base = replay.baseline
common = replay.common
sys.path.insert(0,str(common.ROOT/'external/ocar'))
from nngeometry.metrics import FIM
from nngeometry.object import PMatKFAC, PVector


def rfe_node(name):
    """Execute the unmodified official module class, without framework imports."""
    from functools import partial
    from einops.layers.torch import Reduce, Rearrange
    path = common.ROOT/'external/rfe/src/methods/rfe.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    scope = dict(nn=nn,fn=F,torch=torch,partial=partial,Reduce=Reduce,Rearrange=Rearrange)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


class RFE(nn.Module):
    def __init__(self,seed,width=444):
        super().__init__()
        torch.manual_seed(seed)
        # ReLU matches the nonnegative native feature space of the retrospector.
        self.extractor = nn.Sequential(nn.Linear(784,width),nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(width,2) for _ in range(5)])
        self.aux = nn.ModuleList()
        self.retro = nn.ModuleList()
        self.width = width

    def image(self,x):
        # Official auxiliary module consumes RGB 32px images and pools to 16px.
        return F.interpolate(x.reshape(-1,1,28,28),size=(32,32),mode='bilinear',align_corners=False).expand(-1,3,-1,-1)

    def scores(self,x):
        features = self.extractor(x)
        values = [None]*5
        # Equivalent to official mask/softmax/mean: each class has exactly one
        # task-associated prediction and all tasks compete without a task hint.
        values[4] = self.heads[4](features).softmax(1)
        images = self.image(x)
        for stage in range(3,-1,-1):
            features = self.retro[stage](features,self.aux[stage](images))
            values[stage] = self.heads[stage](features).softmax(1)
        return torch.cat(values,1)/5


def batches(indices,seed,stage,epoch):
    gen = torch.Generator().manual_seed(seed*100000+stage*100+epoch)
    return indices[torch.randperm(len(indices),generator=gen)].split(128)


def fit_rfe(config,seed,order,x,y,indices):
    model = RFE(seed)
    mapping = torch.tensor([c for k in order for c in common.SPLITS[k]])
    inverse = mapping.argsort()
    # Keep optimizer continuous across backbone task updates, like prior arms.
    opt = torch.optim.Adam(list(model.extractor.parameters())+list(model.heads.parameters()),lr=config['lr'])
    previous = None
    seconds = time.perf_counter()
    module_presentations = 0
    max_opt_bytes = 0
    adapter = None
    for stage,group in enumerate(order):
        ix = indices[group]
        for epoch in range(2):
            for ids in batches(ix,seed,stage,epoch):
                bx = x[ids]
                features = model.extractor(bx)
                loss = F.cross_entropy(model.heads[stage](features),inverse[y[ids]]-2*stage)
                if previous is not None:
                    with torch.no_grad():
                        old = previous(bx)
                    loss = loss+config['regularize']*F.mse_loss(features,old)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.extractor.parameters())+list(model.heads.parameters()),5)
                opt.step()
        model.heads[stage].requires_grad_(False)
        # Official two-phase RFE: train current auxiliary extractor to mimic
        # current features and previous retrospector to mimic previous features.
        model.aux.append(rfe_node('AuxFeatureExtractor')(32))
        adapter = nn.Linear(32,model.width)
        if stage:
            model.retro.append(rfe_node('Retrospector')(model.width,32,16))
        parameters = list(model.aux[-1].parameters())+list(adapter.parameters())
        if stage:
            parameters += list(model.retro[-1].parameters())
        module_opt = torch.optim.Adam(parameters,lr=.005)
        for epoch in range(2):
            for ids in batches(ix,seed+1000,stage,epoch):
                bx = x[ids]
                image = model.image(bx)
                with torch.no_grad():
                    features = model.extractor(bx)
                loss = F.mse_loss(adapter(model.aux[-1](image)),features)
                if stage:
                    with torch.no_grad():
                        old = previous(bx)
                        aux = model.aux[stage-1](image)
                    loss = loss+F.mse_loss(model.retro[-1](features,aux),old)
                module_opt.zero_grad(set_to_none=True)
                loss.backward()
                module_opt.step()
                module_presentations += len(ids)
        max_opt_bytes = max(max_opt_bytes,optimizer_bytes(module_opt)+optimizer_bytes(opt))
        model.aux[-1].requires_grad_(False)
        if stage:
            model.retro[-1].requires_grad_(False)
        previous = deepcopy(model.extractor).requires_grad_(False)
    model_bytes = base.tensor_bytes(model.state_dict().values())
    teacher_bytes = base.tensor_bytes(previous.state_dict().values())
    # Upstream also retains a copy of all classifiers for continuing learning.
    teacher_head_bytes = base.tensor_bytes(model.heads.state_dict().values())
    maps = base.tensor_bytes([mapping,inverse])
    stats = dict(parameters=sum(p.numel() for p in model.parameters()),model_bytes=model_bytes,
                 teacher_bytes=teacher_bytes+teacher_head_bytes,mapping_bytes=maps,buffer_bytes=0,
                 persistent_numeric_bytes=model_bytes+teacher_bytes+teacher_head_bytes+maps,
                 optimizer_state_bytes=optimizer_bytes(opt),max_optimizer_state_bytes=max_opt_bytes,
                 training_seconds=time.perf_counter()-seconds,current_presentations=2*len(y),
                 extra_module_presentations=module_presentations,replayed_presentations=0)
    state = dict(model=model.state_dict(),teacher=previous.state_dict(),teacher_heads=deepcopy(model.heads.state_dict()),mapping=mapping)
    return model,mapping,stats,state


def optimizer_bytes(opt):
    return sum(base.tensor_bytes(v for v in state.values() if torch.is_tensor(v)) for state in opt.state.values())


class Curvature:
    """Paper's Class-IL weighting, using official FIM/KFAC/inverse kernels.

    Released NGPlugin has CLEAR-specific n_new drift and does not implement its
    documented Class-IL branch. Here replay weights use known/current class ratio.
    """
    def __init__(self,config):
        self.config = config
        self.ema = self.inv = None
        self.tau = 0.
        self.known = set()

    def update(self,model,x,y,n_current):
        self.known.update(y[:n_current].tolist())
        weights = torch.ones(len(y))
        if len(y)>n_current:
            weights[n_current:] = len(self.known)/len(y[:n_current].unique())
        self.tau = self.config['lr'] if self.tau==0 else self.tau+self.config['tau_increment']
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=len(y))
        fresh = FIM(model,loader,PMatKFAC,n_output=10,device='cpu',lambda_=0.,weights=weights)
        if self.ema is not None:
            for key,(a,g) in fresh.data.items():
                olda,oldg = self.ema.data[key]
                alpha = self.config['ema']
                fresh.data[key] = ((1-alpha)*olda+alpha*a,(1-alpha)*oldg+alpha*g)
        self.ema = fresh
        self.inv = self.ema.inverse(regul=self.tau)
        # FIM computes separate gradients; restore the task loss gradient
        # explicitly rather than relying on backward-hook side effects.

    def tensors(self):
        return [t for mat in [self.ema,self.inv] if mat is not None for pair in mat.data.values() for t in pair]


def fit_ocar(config,seed,order,x,y,indices,capacity=200):
    model,mapping,inverse = replay.make('derpp',seed,order)
    # Natural-gradient SGD is intrinsic to OCAR; applying Adam after curvature
    # preconditioning would test a different optimizer than the paper.
    opt = torch.optim.SGD(model.parameters(),lr=config['lr'])
    buffer = replay.Replay(capacity,seed,False)
    curvature = Curvature(config)
    start = time.perf_counter()
    replayed = 0
    for stage,group in enumerate(order):
        for epoch in range(2):
            for ids in batches(indices[group],seed,stage,epoch):
                bx,by = x[ids],inverse[y[ids]]
                if buffer.used:
                    sample = buffer.draw(len(ids))
                    xx,yy = torch.cat([bx,sample['x']]),torch.cat([by,sample['y']])
                    replayed += len(sample['y'])
                else:
                    xx,yy = bx,by
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(xx),yy)
                loss.backward()
                gradients = [p.grad.detach().clone() for p in model.parameters()]
                curvature.update(model,xx,yy,len(ids))
                for p,g in zip(model.parameters(),gradients):
                    p.grad = g
                curvature.inv.mv(PVector.from_model_grad(model)).to_model_grad(model)
                # Official OCAR disables gradient clipping. A fixed raw-gradient
                # norm bound would suppress its natural-gradient step scale.
                assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step()
                buffer.observe(bx,by,None)
    model_bytes = base.tensor_bytes(model.state_dict().values())
    curvature_bytes = base.tensor_bytes(curvature.tensors())
    mapping_bytes = base.tensor_bytes([mapping,inverse])
    # 10 boolean flags + FP64 tau/logical step state, explicitly charged.
    scalar_bytes = 26
    stats = dict(parameters=sum(p.numel() for p in model.parameters()),model_bytes=model_bytes,
                 curvature_bytes=curvature_bytes,buffer_bytes=buffer.nbytes(),mapping_bytes=mapping_bytes,
                 persistent_numeric_bytes=model_bytes+curvature_bytes+buffer.nbytes()+mapping_bytes+scalar_bytes,
                 optimizer_state_bytes=optimizer_bytes(opt),training_seconds=time.perf_counter()-start,
                 current_presentations=2*len(y),replayed_presentations=replayed,extra_module_presentations=0)
    state = dict(model=model.state_dict(),buffer=buffer.state(),mapping=mapping,
                 factors=curvature.ema.data,inverse_factors=curvature.inv.data,tau=curvature.tau)
    return model,mapping,stats,state


@torch.no_grad()
def predict(method,model,x,mapping):
    model.eval()
    logits = torch.cat([(model.scores(b) if method=='rfe' else model(b)) for b in x.split(256)])
    return mapping[logits.argmax(1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--method',choices=['rfe','ocar','both'],default='both')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    configs = {
        'rfe':[dict(lr=lr,regularize=reg) for lr in [.0005,.001,.003] for reg in [0.,1.]],
        'ocar':[dict(lr=lr,ema=.13851270272015948,tau_increment=tau) for lr in [.0005,.001,.003] for tau in [3.671644595771195e-8,1e-5]],
    }
    methods = ['rfe','ocar'] if args.method=='both' else [args.method]
    if args.smoke:
        configs = {m:configs[m][:1] for m in methods}
    protocol = dict(methods=methods,configs=configs,seeds=[101,102,103],orders=common.ORDERS,
                    tuning_seed=113,smoke=args.smoke,scope='Adaptations; not original-paper reproduction or a verified latest-SOTA ranking',
                    shared='Same 55k/5k split, 10k test, 2 epochs/task, batch128, no test task IDs. Validation-only tuning.',
                    rfe='Official auxiliary CNN and retrospector classes, width444 ReLU main features, d_aux32/d_proj16; two extra module epochs/task; bilinear gray-to-RGB input for auxiliary only; regularization0/1; module Adam .005. Native two-way heads and Class-IL aggregation.',
                    ocar='Official nngeometry FIM and KFAC inverse; paper Class-IL known/current class replay weighting, replacing CLEAR-specific released branch; SGD; one step/batch; no gradient clipping as upstream; 200-image reservoir. Not the unmodified NGPlugin.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    revisions = {}
    for name in ['rfe','ocar']:
        root = common.ROOT/'external'/name
        revisions[name] = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    (out/'upstream.json').write_text(json.dumps(revisions,indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    raw = common.load_raw(str(common.ROOT/'data_cache/mnist'))
    x = (torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    y = torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx = (torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    ty = torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    split = np.load(common.ROOT/'results/matched_numenta_20261031/split_indices.npz')
    vx,vy = x[split['validation']],y[split['validation']]
    x,y = x[split['train']],y[split['train']]
    indices = [torch.nonzero(y//2==k,as_tuple=True)[0] for k in range(5)]
    if args.smoke:
        indices = [ix[:256] for ix in indices]
    def read(name,default):
        return json.loads((out/name).read_text()) if (out/name).exists() else default
    trials = read('validation_trials.json',[])
    selected = read('selected_hyperparameters.json',{})
    for method in methods:
        fit = fit_rfe if method=='rfe' else fit_ocar
        if method in selected:
            continue
        for config in configs[method]:
            if any(r['method']==method and r['config']==config for r in trials):
                continue
            model,mapping,stats,_ = fit(config,113,common.ORDERS['forward'],x,y,indices)
            accuracy = float((predict(method,model,vx,mapping)==vy).float().mean())
            trials.append(dict(method=method,config=config,validation_accuracy=accuracy,**stats))
            (out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
            print('TUNE',method,config,round(100*accuracy,2),round(stats['training_seconds'],2),flush=True)
        selected[method] = max((r for r in trials if r['method']==method),key=lambda r:r['validation_accuracy'])['config']
    # Freeze every selection before any official-test inference.
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    rows = read('results.json',[])
    for method in methods:
        fit = fit_rfe if method=='rfe' else fit_ocar
        for seed in ([101] if args.smoke else [101,102,103]):
            for name,order in common.ORDERS.items():
                if any(r['method']==method and r['seed']==seed and r['order']==name for r in rows):
                    continue
                model,mapping,stats,state = fit(selected[method],seed,order,x,y,indices)
                before = common.state_hash(model)
                predictions = predict(method,model,tx,mapping)
                assert before==common.state_hash(model)
                scores = common.metrics(predictions,ty,order)
                rows.append(dict(method=method,seed=seed,order=name,config=selected[method],scores=scores,**stats))
                torch.save(state,out/f'{method}_{seed}_{name}.pt')
                (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                print('FINAL',method,seed,name,round(100*scores['accuracy'],2),stats['persistent_numeric_bytes'],flush=True)
    for method in methods:
        values = [statistics.mean(100*r['scores']['accuracy'] for r in rows if r['method']==method and r['seed']==seed) for seed in sorted({r['seed'] for r in rows if r['method']==method})]
        print('SUMMARY',method,statistics.mean(values),statistics.stdev(values) if len(values)>1 else 0,flush=True)


if __name__=='__main__':
    main()
