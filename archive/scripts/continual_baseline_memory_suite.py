"""Common-protocol Split MNIST baseline suite with explicit numeric state.

Reuses the previous split and cohort. Local SI and online EWC implementations;
no claim of reproducing a particular publication's tuned scores.
"""
import argparse
import hashlib
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, grad, vmap

from bdh_cl.scripts import matched_numenta_comparison as common


def make(seed, frozen=False):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(784,666),nn.Tanh(),nn.Linear(666,10))
    if frozen:
        model = nn.Sequential(model[0],model[1])
        model.requires_grad_(False)
    return model


def tensor_bytes(values):
    return sum(v.numel()*v.element_size() for v in values)


class SI:
    def __init__(self, model, strength):
        self.strength = strength
        self.reference = [p.detach().clone() for p in model.parameters()]
        self.omega = [torch.zeros_like(p) for p in model.parameters()]
        self.path = [torch.zeros_like(p) for p in model.parameters()]

    def penalty(self, model):
        return self.strength*sum((o*(p-r).square()).sum() for p,r,o in zip(model.parameters(),self.reference,self.omega))

    @torch.no_grad()
    def observe(self,model,before,task_gradients):
        for w,p,old,g in zip(self.path,model.parameters(),before,task_gradients):
            w.add_(-g*(p-old))

    @torch.no_grad()
    def consolidate(self,model):
        for p,r,o,w in zip(model.parameters(),self.reference,self.omega,self.path):
            o.add_((w/((p-r).square()+.1)).clamp_min(0))
            r.copy_(p)
            w.zero_()

    def state(self):
        return self.reference+self.omega+self.path


class OnlineEWC:
    def __init__(self,model,strength):
        self.strength = strength
        self.reference = {n:p.detach().clone() for n,p in model.named_parameters()}
        self.fisher = {n:torch.zeros_like(p) for n,p in model.named_parameters()}

    def penalty(self,model):
        return .5*self.strength*sum((self.fisher[n]*(p-self.reference[n]).square()).sum() for n,p in model.named_parameters())

    def consolidate(self,model,x,y):
        # Empirical Fisher = mean squared per-example log-likelihood gradients.
        # Squaring a minibatch-mean gradient would not estimate this quantity.
        params = {n:p.detach() for n,p in model.named_parameters()}
        def single_loss(parameters,bx,by):
            return F.cross_entropy(functional_call(model,parameters,(bx[None],)),by[None])
        per_example = vmap(grad(single_loss),in_dims=(None,0,0))
        estimates = {n:torch.zeros_like(p) for n,p in params.items()}
        for bx,by in zip(x.split(8),y.split(8)):
            grads = per_example(params,bx,by)
            for n,g in grads.items():
                estimates[n].add_(g.detach().square().sum(0))
        with torch.no_grad():
            for n,p in model.named_parameters():
                self.fisher[n].add_(estimates[n]/len(x))
                self.reference[n].copy_(p)
                assert torch.isfinite(self.fisher[n]).all() and (self.fisher[n]>=0).all()

    def state(self):
        return list(self.reference.values())+list(self.fisher.values())


class Replay:
    def __init__(self,capacity,seed):
        self.images = torch.zeros((capacity,784),dtype=torch.uint8)
        self.labels = torch.zeros(capacity,dtype=torch.uint8)
        self.priority = np.full(capacity,-1.,dtype=np.float64)
        self.rng = np.random.default_rng(seed+7000)
        self.used = 0
        self.seen = 0

    def observe(self,bx,by):
        priorities = np.concatenate([self.priority,self.rng.random(len(by))])
        keep = np.argsort(priorities)[-len(self.priority):]
        raw = ((bx*.30810780+.13062755)*255).round().clamp(0,255).to(torch.uint8)
        self.images = torch.cat([self.images,raw])[keep]
        self.labels = torch.cat([self.labels,by.to(torch.uint8)])[keep]
        self.priority = priorities[keep]
        self.used = int((self.priority>=0).sum())
        self.seen += len(by)

    def sample(self,n):
        valid = np.flatnonzero(self.priority>=0)
        if not len(valid):
            return None
        ids = self.rng.choice(valid,n,replace=True)
        bx = (self.images[ids].float()/255-.13062755)/.30810780
        return bx,self.labels[ids].long()

    def nbytes(self):
        return tensor_bytes([self.images,self.labels])+self.priority.nbytes+16


@torch.no_grad()
def predict(model,x):
    model.eval()
    return torch.cat([model(b).argmax(1) for b in x.split(256)])


def trained_run(arm,config,seed,order,x,y,vx,vy,indices):
    model = make(seed)
    opt = torch.optim.Adam(model.parameters(),lr=config['lr'])
    regularizer = SI(model,config['strength']) if arm=='si' else OnlineEWC(model,config['strength']) if arm=='ewc' else None
    replay = Replay(config['capacity'],seed) if arm=='replay' else None
    seconds = 0.
    replayed = clipped = steps = 0
    groups = [None] if arm=='joint' else order
    for stage,group in enumerate(groups):
        ix = torch.arange(len(y)) if group is None else indices[group]
        model.train()
        for epoch in range(2):
            gen = torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ids in ix[torch.randperm(len(ix),generator=gen)].split(128):
                start = time.perf_counter()
                bx,by = x[ids],y[ids]
                train_x,train_y = bx,by
                if replay is not None:
                    sample = replay.sample(len(by))
                    if sample is not None:
                        train_x,train_y = torch.cat([bx,sample[0]]),torch.cat([by,sample[1]])
                        replayed += len(sample[1])
                opt.zero_grad(set_to_none=True)
                task_loss = F.cross_entropy(model(train_x),train_y)
                if isinstance(regularizer,SI):
                    task_grads = torch.autograd.grad(task_loss,tuple(model.parameters()),retain_graph=True)
                    before = [p.detach().clone() for p in model.parameters()]
                loss = task_loss+(regularizer.penalty(model) if regularizer is not None else 0)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                clipped += int(norm>5.)
                opt.step()
                if isinstance(regularizer,SI):
                    regularizer.observe(model,before,task_grads)
                if replay is not None:
                    replay.observe(bx,by)
                steps += 1
                seconds += time.perf_counter()-start
        if isinstance(regularizer,SI):
            start = time.perf_counter()
            regularizer.consolidate(model)
            seconds += time.perf_counter()-start
        elif isinstance(regularizer,OnlineEWC):
            # Fixed 64 examples per current class, from training data only.
            g = torch.Generator().manual_seed(seed*100000+stage*100+98)
            take = torch.cat([ids[torch.randperm(len(ids),generator=g)[:64]] for ids in
                              [torch.nonzero(y==c,as_tuple=True)[0] for c in common.SPLITS[group]]])
            start = time.perf_counter()
            regularizer.consolidate(model,x[take],y[take])
            seconds += time.perf_counter()-start
    params = sum(p.numel() for p in model.parameters())
    model_bytes = tensor_bytes(model.state_dict().values())
    extra_bytes = tensor_bytes(regularizer.state()) if regularizer is not None else replay.nbytes() if replay is not None else 0
    opt_bytes = sum(tensor_bytes(v for v in state.values() if torch.is_tensor(v)) for state in opt.state.values())
    statistics = {'parameters':params,'model_bytes':model_bytes,'auxiliary_state_bytes':extra_bytes,
                  'persistent_numeric_bytes':model_bytes+extra_bytes,'optimizer_state_bytes':opt_bytes,
                  'replay_bytes':replay.nbytes() if replay is not None else 0,
                  'training_seconds':seconds,'replayed_presentations':replayed,'current_presentations':2*len(y),
                  'fisher_presentations':640 if arm=='ewc' else 0,'steps':steps,'clipped_steps':clipped}
    state = {'model':model.state_dict(), 'regularizer':regularizer.state() if regularizer else None,
             'replay':{'images':replay.images,'labels':replay.labels,'priority':replay.priority,'seen':replay.seen} if replay else None}
    return model,statistics,state


def ncm_run(arm,config,seed,order,x,y,indices):
    feature_model = make(seed,True) if arm=='frozen_ncm' else nn.Identity()
    dimension = 666 if arm=='frozen_ncm' else 784
    sums = torch.zeros(10,dimension)
    counts = torch.zeros(10,dtype=torch.int64)
    start = time.perf_counter()
    with torch.no_grad():
        for group in order:
            for _ in range(2):
                for ids in indices[group].split(128):
                    z = feature_model(x[ids])
                    if config['distance']=='cos_unit':
                        z = F.normalize(z,dim=1)
                    sums.index_add_(0,y[ids],z)
                    counts.add_(torch.bincount(y[ids],minlength=10))
    seconds = time.perf_counter()-start
    statistics = {'parameters':0,'fixed_parameters':sum(p.numel() for p in feature_model.parameters()),
                  'model_bytes':tensor_bytes(feature_model.state_dict().values()),
                  'auxiliary_state_bytes':tensor_bytes([sums,counts]),'optimizer_state_bytes':0,
                  'replay_bytes':0,'training_seconds':seconds,'replayed_presentations':0,'current_presentations':2*len(y)}
    statistics['persistent_numeric_bytes'] = statistics['model_bytes']+statistics['auxiliary_state_bytes']
    return feature_model,sums,counts,statistics


@torch.no_grad()
def ncm_predict(model,sums,counts,config,x):
    means = sums/counts.clamp_min(1)[:,None]
    outputs = []
    for bx in x.split(256):
        z = model(bx)
        if config['distance']=='euclidean':
            logits = -torch.cdist(z,means)
        else:
            logits = F.normalize(z,dim=1)@F.normalize(means,dim=1).T
        outputs.append(logits.masked_fill(counts[None]==0,-torch.inf).argmax(1))
    return torch.cat(outputs)


def write_report(out,rows,base):
    lines = ['# Common-protocol continual-learning baselines and retained memory','',
             'Split MNIST; same 55,000 training / 5,000 validation / official 10,000 test split as the matched branch comparison. '
             'Same standard normalization, three evaluation seeds (101–103), forward/reverse pair orders, two current-task epochs, '
             'batch 128, Adam without boundary resets and gradient clip 5. All ten classes compete at final inference; no task IDs or output masks.', '',
             '| Method | Final accuracy ± seed SD | Retained CL state (MB) | Adam state (MB) | Replay state (MB) |',
             '|---|---:|---:|---:|---:|']
    groups = [(arm,[r for r in rows if r['arm']==arm]) for arm in
              ['finetune','si','ewc','raw_ncm','frozen_ncm','replay200','replay1000','joint']]
    old = json.loads((base/'results.json').read_text())
    for arm in ['branch_aux','branch_ce','numenta']:
        selected = [r for r in old if r['arm']==arm]
        seeds = sorted({r['seed'] for r in selected})
        vals = [st.mean(100*r['scores']['inferred']['accuracy'] for r in selected if r['seed']==s) for s in seeds]
        lines.append(f'| {arm} (previous matched run) | {st.mean(vals):.2f} ± {st.stdev(vals):.2f}% | '
                     f'{selected[0]["persistent_numeric_bytes"]/1e6:.4f} | {st.mean(r["optimizer_state_bytes"] for r in selected)/1e6:.4f} | 0 |')
    for arm,selected in groups:
        if not selected:
            continue
        vals = [st.mean(100*r['scores']['accuracy'] for r in selected if r['seed']==s) for s in sorted({r['seed'] for r in selected})]
        sd = st.stdev(vals) if len(vals)>1 else 0.
        lines.append(f'| {arm} | {st.mean(vals):.2f} ± {sd:.2f}% | '
                     f'{st.mean(r["persistent_numeric_bytes"] for r in selected)/1e6:.4f} | '
                     f'{st.mean(r["optimizer_state_bytes"] for r in selected)/1e6:.4f} | '
                     f'{st.mean(r["replay_bytes"] for r in selected)/1e6:.4f} |')
    lines += ['', 'MB = 1,000,000 bytes. Replay is INCLUDED in retained CL state; do not add that column again. '
              'Adam state is additional. Retained CL state counts model tensors and all auxiliary numeric state kept by this implementation to continue learning, '
              'not peak RAM or the minimum possible inference-only deployment. It excludes input dataset caches, Python/RNG objects, transient gradients/activations and OS allocator overhead. '
              'SI retains reference weights, importance and the allocated path-integral accumulator, even when zero at a task boundary; EWC retains Fisher and reference weights. '
              'The final classifier alone can be smaller if continued learning is no longer needed.', '',
              'Dense baselines use the same 784→666 tanh→10 MLP (529,480 trainable parameters), within 0.06% of the branch parameter count. '
              'Frozen NCM retains the same frozen hidden layer plus ten class sums/counts, with zero trained weights. Raw NCM is an intentionally much smaller statistical baseline. '
              'NCM does not use a task-mean router or configuration archive; labels only update class means during supervised training.', '',
              'Fine-tuning, joint and each replay size receive three LR trials (.0005,.001,.003). '
              'SI and online EWC each receive nine trials (three LRs × three regularization strengths), so they receive MORE validation search than the branch method. '
              'NCM receives three distance/normalization candidates. All choose final validation accuracy on seed 113, forward order, then freeze before official test evaluation.', '',
              'Replay uses a uniform random-priority reservoir of raw uint8 images and uint8 labels, retained float64 priorities, '
              'and two numeric counters. Current examples enter only after their training update. One replay example per current example is added to each nonempty-buffer minibatch, '
              'so replay has extra gradient exposures and compute. Capacities 200 and 1,000 are separate fixed comparisons, not test-selected.', '',
              '**Joint is an offline reference, not a continual-learning result:** all 55,000 training inputs are available simultaneously. '
              'Its dataset-access requirement is not included in the model-state column (55,000 raw images/labels require 43.175 MB before framework overhead). '
              'Three joint seeds are run once each, rather than pretending forward/reverse order supplies extra independent runs.', '',
              'Online EWC uses cumulative diagonal empirical Fisher (mean squared per-example gradients) from 128 class-balanced current-task training examples after each phase, '
              'with latest-weight anchors. SI uses unregularized task gradients in its path integral, damping .1, cumulative nonnegative importance, and boundary-updated references. '
              'These are local PyTorch implementations, not reproductions of published tuned leaderboards. No HAT or PackNet results are claimed.', '',
              'The branch and Numenta rows reuse completed matched runs; their training-context protocol allows known boundaries and prototype preprocessing. '
              'Regularization also uses training boundaries; fine-tuning, replay and NCM need no task identity at inference. '
              'Standard deviation is across three seed means, after averaging the two orders per seed where applicable. Raw NCM is deterministic.', '',
              'Sources: [SI](https://proceedings.mlr.press/v70/zenke17a.html); '
              '[EWC](https://doi.org/10.1073/pnas.1611835114). '
              'Per-trial configurations, exact bytes, sample exposures, timings, final states and frozen-evaluation checks are saved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(2)
    rates = [.0005,.001,.003]
    configurations = {
        'finetune':[{'lr':r} for r in rates],
        'si':[{'lr':r,'strength':c} for r in rates for c in [.01,.1,1.]],
        'ewc':[{'lr':r,'strength':c} for r in rates for c in [1.,100.,10000.]],
        'raw_ncm':[{'distance':d} for d in ['euclidean','cosine','cos_unit']],
        'frozen_ncm':[{'distance':d} for d in ['euclidean','cosine','cos_unit']],
        'replay200':[{'lr':r,'capacity':200} for r in rates],
        'replay1000':[{'lr':r,'capacity':1000} for r in rates],
        'joint':[{'lr':r} for r in rates]}
    (out/'protocol.json').write_text(json.dumps({'base':str(args.base),'configurations':configurations,
                'seeds':[101,102,103],'tuning_seed':113,'orders':common.ORDERS,
                'validation_rule':'Maximum final ten-class validation accuracy; deterministic listed-order tie breaking. SI/EWC have more search trials than other arms.',
                'memory_definition':'Allocated persistent numeric state for continuing learning, separate Adam state; not peak RAM. Replay bytes already included in persistent state.',
                'si':'Task-gradient path integral, damping .1, cumulative positive importance; strength grid .01/.1/1.',
                'ewc':'Online summed empirical diagonal Fisher, gamma1, latest-weight reference; 64 current training examples per class; strength grid1/100/10000.'},indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    raw = common.load_raw(str(common.ROOT/'data_cache/mnist'))
    x = (torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    y = torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx = (torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    ty = torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    split = np.load(args.base/'split_indices.npz')
    vx,vy = x[split['validation']],y[split['validation']]
    x,y = x[split['train']],y[split['train']]
    indices = [torch.nonzero(y//2==k,as_tuple=True)[0] for k in range(5)]
    trials = json.loads((out/'validation_trials.json').read_text()) if args.resume and (out/'validation_trials.json').exists() else []
    selected = json.loads((out/'selected_hyperparameters.json').read_text()) if args.resume and (out/'selected_hyperparameters.json').exists() else {}
    for arm,candidates in configurations.items():
        if arm in selected:
            continue
        kind = 'replay' if arm.startswith('replay') else arm
        for config in candidates:
            if arm.endswith('ncm'):
                model,sums,counts,stats = ncm_run(arm,config,113,common.ORDERS['forward'],x,y,indices)
                pred = ncm_predict(model,sums,counts,config,vx)
            else:
                model,stats,_ = trained_run(kind,config,113,common.ORDERS['forward'],x,y,vx,vy,indices)
                pred = predict(model,vx)
            value = float((pred==vy).float().mean())
            trials.append({'arm':arm,'config':config,'validation_accuracy':value})
            (out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
            print('TUNE',arm,config,round(100*value,2),flush=True)
        best = max((r for r in trials if r['arm']==arm),key=lambda r:r['validation_accuracy'])
        selected[arm] = best['config']
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    rows = json.loads((out/'results.json').read_text()) if args.resume and (out/'results.json').exists() else []
    for arm in configurations:
        kind = 'replay' if arm.startswith('replay') else arm
        for seed in [101,102,103]:
            for name,order in ({'joint':list(range(5))} if arm=='joint' else common.ORDERS).items():
                if any(r['arm']==arm and r['seed']==seed and r['order']==name for r in rows):
                    continue
                config = selected[arm]
                if arm.endswith('ncm'):
                    model,sums,counts,stats = ncm_run(arm,config,seed,order,x,y,indices)
                    before = common.state_hash(model)
                    copies = [sums.clone(),counts.clone()]
                    pred = ncm_predict(model,sums,counts,config,tx)
                    assert common.state_hash(model)==before and torch.equal(copies[0],sums) and torch.equal(copies[1],counts)
                    state = {'model':model.state_dict(),'class_sums':sums,'counts':counts}
                else:
                    model,stats,state = trained_run(kind,config,seed,order,x,y,vx,vy,indices)
                    before = common.state_hash(model)
                    pred = predict(model,tx)
                    assert common.state_hash(model)==before
                row = {'arm':arm,'seed':seed,'order':name,'config':config,
                       'scores':common.metrics(pred,ty,order),**stats}
                rows.append(row)
                torch.save(state,out/f'{arm}_{seed}_{name}.pt')
                (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                write_report(out,rows,args.base)
                print('FINAL',arm,seed,name,round(100*row['scores']['accuracy'],2),stats['persistent_numeric_bytes'],flush=True)
    print('Completed:',out,flush=True)


if __name__=='__main__':
    main()
