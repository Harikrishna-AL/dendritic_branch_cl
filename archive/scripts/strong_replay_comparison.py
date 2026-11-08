"""DER++ and ER-ACE under the existing matched Split MNIST protocol.

Audited objective ports, upstream cosine classifier, local common experiment
runner. This is not the full Mammoth CLI or a published-score reproduction.
"""
import argparse
import ast
import hashlib
import json
import statistics as st
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bdh_cl.scripts import continual_baseline_memory_suite as baseline

common = baseline.common
UPSTREAM = common.ROOT/'external/mammoth'


def official_node(filename,name,method=None):
    """Load a verified source node without importing the unrelated framework."""
    path = UPSTREAM/'models'/filename
    nodes = ast.parse(path.read_text()).body
    node = next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==name)
    if method:
        node = next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==method)
    namespace = {'torch':torch,'F':F,'Variable':torch.autograd.Variable}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[method or name]


def make(method,seed,order):
    model = baseline.make(seed)
    if method=='er_ace':
        # Use the official implementation verbatim, including its epsilon/scale.
        model[2] = official_node('er_ace.py','CustomLinear')(666,10)
        classifier = model[2].L
    else:
        classifier = model[2]
    stream_to_class = torch.tensor([c for k in order for c in common.SPLITS[k]],dtype=torch.long)
    class_to_stream = torch.argsort(stream_to_class)
    # Preserve initial canonical predictions while renaming output coordinates.
    with torch.no_grad():
        classifier.weight.copy_(classifier.weight.detach().clone()[stream_to_class])
        if classifier.bias is not None:
            classifier.bias.copy_(classifier.bias.detach().clone()[stream_to_class])
    return model,stream_to_class,class_to_stream


class Replay:
    def __init__(self,capacity,seed,with_logits):
        self.images = torch.zeros(capacity,784,dtype=torch.uint8)
        self.labels = torch.zeros(capacity,dtype=torch.uint8)
        self.logits = torch.zeros(capacity,10,dtype=torch.float32) if with_logits else None
        self.priority = np.full(capacity,-1.,dtype=np.float64)
        self.write_rng = np.random.default_rng(seed+7000)
        self.sample_rng = np.random.default_rng(seed+8000)
        self.used = self.seen = 0

    def draw(self,n):
        ids = self.sample_rng.choice(np.flatnonzero(self.priority>=0),min(n,self.used),replace=False)
        return {'x':(self.images[ids].float()/255-.13062755)/.30810780,
                'y':self.labels[ids].long(), 'logits':self.logits[ids] if self.logits is not None else None}

    def observe(self,x,y,logits):
        raw = ((x*.30810780+.13062755)*255).round().clamp(0,255).to(torch.uint8)
        priority = np.concatenate([self.priority,self.write_rng.random(len(y))])
        keep = np.argsort(priority)[-len(self.priority):]
        self.images = torch.cat([self.images,raw])[keep]
        self.labels = torch.cat([self.labels,y.to(torch.uint8)])[keep]
        if self.logits is not None:
            self.logits = torch.cat([self.logits,logits.detach().float()])[keep]
        self.priority = priority[keep]
        self.used = int((self.priority>=0).sum())
        self.seen += len(y)

    def nbytes(self):
        return baseline.tensor_bytes([self.images,self.labels]+([self.logits] if self.logits is not None else []))+self.priority.nbytes+16

    def state(self):
        return {'images':self.images,'labels':self.labels,'logits':self.logits,'priorities':self.priority,
                'used':self.used,'seen':self.seen}


def ace_mask(logits,labels,seen):
    present = labels.unique()
    seen = torch.cat([seen,present]).unique()
    mask = torch.zeros_like(logits,dtype=torch.bool)
    mask[:,present] = True
    mask[:,seen.max():] = True
    return logits.masked_fill(~mask,-1e9),seen


def objective(method,model,x,y,buffer,config,seen):
    outputs = model(x)
    replayed = 0
    if method=='derpp':
        loss = F.cross_entropy(outputs,y)
        if buffer.used:
            sample = buffer.draw(len(y))
            loss = loss+config['alpha']*F.mse_loss(model(sample['x']),sample['logits'])
            replayed += len(sample['y'])
            sample = buffer.draw(len(y))
            loss = loss+config['beta']*F.cross_entropy(model(sample['x']),sample['y'])
            replayed += len(sample['y'])
    else:
        masked,seen = ace_mask(outputs,y,seen)
        loss = F.cross_entropy(masked,y)
        if buffer.used:
            sample = buffer.draw(len(y))
            loss = loss+F.cross_entropy(model(sample['x']),sample['y'])
            replayed += len(sample['y'])
    return loss,outputs,seen,replayed


@torch.no_grad()
def predict(model,x,stream_to_class):
    model.eval()
    return stream_to_class[torch.cat([model(b).argmax(1) for b in x.split(256)])]


def fit(method,capacity,config,seed,order,x,y,vx,vy,indices):
    model,stream_to_class,class_to_stream = make(method,seed,order)
    opt = torch.optim.Adam(model.parameters(),lr=config['lr'])
    buffer = Replay(capacity,seed,method=='derpp')
    seen = torch.tensor([],dtype=torch.long)
    seconds = 0.
    replayed = steps = clipped = 0
    for stage,group in enumerate(order):
        model.train()
        ix = indices[group]
        for epoch in range(2):
            gen = torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ids in ix[torch.randperm(len(ix),generator=gen)].split(128):
                start = time.perf_counter()
                bx,by = x[ids],class_to_stream[y[ids]]
                opt.zero_grad(set_to_none=True)
                loss,outputs,seen,extra = objective(method,model,bx,by,buffer,config,seen)
                # Must archive pre-update logits, not predictions from new weights.
                stored_logits = outputs.detach().clone()
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                clipped += int(norm>5.)
                opt.step()
                buffer.observe(bx,by,stored_logits)
                replayed += extra
                steps += 1
                seconds += time.perf_counter()-start
    model_bytes = baseline.tensor_bytes(model.state_dict().values())
    mapping_bytes = baseline.tensor_bytes([stream_to_class,class_to_stream])
    seen_bytes = baseline.tensor_bytes([seen])
    opt_bytes = sum(baseline.tensor_bytes(v for v in state.values() if torch.is_tensor(v)) for state in opt.state.values())
    stats = {'parameters':sum(p.numel() for p in model.parameters()),'model_bytes':model_bytes,
             'buffer_bytes':buffer.nbytes(),'logit_bytes':baseline.tensor_bytes([buffer.logits]) if buffer.logits is not None else 0,
             'mapping_bytes':mapping_bytes,'seen_class_state_bytes':seen_bytes,
             'persistent_numeric_bytes':model_bytes+buffer.nbytes()+mapping_bytes+seen_bytes,
             'optimizer_state_bytes':opt_bytes,'training_seconds':seconds,'steps':steps,'clipped_steps':clipped,
             'current_presentations':2*len(y),'replayed_presentations':replayed,
             'classifier':'official cosine head, scale10' if method=='er_ace' else 'same ordinary linear head as baseline MLP'}
    state = {'model':model.state_dict(),'buffer':buffer.state(),'seen':seen,
             'stream_to_class':stream_to_class,'class_to_stream':class_to_stream}
    return model,stream_to_class,stats,state


def write_report(out,rows,prior):
    lines = ['# Strong replay comparators under the matched Split MNIST protocol','',
             '**DER++ and ER-ACE are strong published comparators, not a verified complete latest-SOTA leaderboard.** '
             'These are objective-audited adaptations to the shared backbone and schedule, not reproductions of original paper scores.', '',
             '| Method | Stored images | Final accuracy ± seed SD | Retained state (MB) | Buffer included (MB) | Additional Adam (MB) |',
             '|---|---:|---:|---:|---:|---:|']
    for method in ['derpp','er_ace']:
        for capacity in [200,1000]:
            selected = [r for r in rows if r['method']==method and r['capacity']==capacity]
            if not selected:
                continue
            vals = [st.mean(100*r['scores']['accuracy'] for r in selected if r['seed']==s) for s in sorted({r['seed'] for r in selected})]
            sd = st.stdev(vals) if len(vals)>1 else 0.
            row = selected[0]
            lines.append(f'| {method} | {capacity} | {st.mean(vals):.2f} ± {sd:.2f}% | '
                         f'{row["persistent_numeric_bytes"]/1e6:.6f} | {row["buffer_bytes"]/1e6:.6f} | {row["optimizer_state_bytes"]/1e6:.6f} |')
    lines += ['', '## Shared experimental settings','',
              'Same cached MNIST, 55,000 training / 5,000 validation / official 10,000 test split; standardized pixels; '
              '784→666 tanh backbone; two current-task epochs per pair; batch 128; Adam; one optimizer across tasks; gradient clip 5; '
              'three evaluation seeds (101–103), each with forward/reverse task order. SD is across seed means after averaging the two orders, not across six independent seeds. '
              'All ten classes compete at test time with no task ID or prediction mask.', '',
              '## Algorithms and audit','',
              'DER++: current-example CE + alpha × all-logit replay MSE + beta × replay CE, using two independent replay draws. '
              'Archive the current forward pass\'s pre-update logits after the optimizer step, exactly as the reference algorithm specifies. '
              'The stored logits stay FP32 and their memory is charged.', '',
              'ER-ACE: asymmetric current-example CE plus unmasked replay CE. Uses the upstream cosine classifier verbatim, including scale 10 and epsilon 1e-5; '
              'this removes the ordinary linear head\'s ten bias values (529,470 vs 529,480 parameters). '
              'The official task_free=True option is used so replay and asymmetric loss are enabled throughout the stream. '
              'Only TRAINING logits are masked; evaluation is unrestricted.', '',
              'The upstream ACE mask assumes increasing stream class indices. Each task order therefore uses a fixed label renaming, '
              'paired with the same classifier-row permutation and a label-only inverse map at inference. '
              'This preserves initial canonical predictions and does not supply an image-specific task hint. '
              'Both new methods use the same coordinate renaming; no statistical selection uses test labels.', '',
              'Objectives are ported into the existing local runner; the Mammoth CLI and its scheduler/dataset wrappers are not used. '
              'Pinned official source snapshots, commit and SHA256 hashes are saved. Focused tests compare optimizer updates and stored logits '
              'against the actual official observe functions on fixed replay draws.', '',
              '## Validation and replay budgets','',
              'For each capacity independently, DER++ receives nine validation configurations: LR .0005/.001/.003 × '
              '(alpha,beta)=(.1,.5),(.5,.5),(.5,1). ER-ACE receives the same three LRs with its official cosine head. '
              'Tune on seed113, forward order, final validation accuracy; freeze all selections before any official test evaluation. '
              'DER++ gets more tuning than the branch model. This is a small search, not exhaustive or convergence-matched tuning.', '',
              'Each replay draw matches the current minibatch size (up to 128 images), sampled without replacement. '
              'The incoming short final minibatch also shrinks its replay draws; this differs from the reference CLI\'s fixed requested replay size. '
              'The earlier simple-replay implementation sampled with replacement, so exact buffer draws differ. '
              'A uniform random-priority reservoir is used rather than the original integer-reservoir implementation; '
              'both are uniform over observed presentations. Separate write/sample RNG streams keep buffer membership matched between the two new algorithms. '
              'Images and labels are stored losslessly as uint8, consistently with the preceding simple-replay memory comparison. '
              'These storage choices differ from default Mammoth FP32-image allocation.', '',
              'Current labeled exposure is the same as before (110,000 presentations). DER++ has two additional replay passes per update; '
              'ER-ACE has one. Consequently training compute and total gradient exposures are not equal to the branch model. Counts and times are recorded per run.', '',
              '## Memory definition and limitations','',
              'Decimal MB. Retained state includes model tensors, raw image/label buffer, retained reservoir priorities/counters, '
              'logits (DER++), class maps, and seen-class state (ER-ACE). Buffer is ALREADY included in retained state. '
              'Adam is additional. These are allocated numeric costs to continue learning, not peak RAM; '
              'dataset caches, gradients, activations and Python/RNG object overhead are excluded. '
              'No historical weight checkpoints or hidden replay buffer are used.', '',
              f'The prior complete comparison remains in `{prior}`. These results extend that comparison without deleting older results. '
              'One dataset, one model budget, a short schedule and three seeds cannot establish a general SOTA claim. '
              'X-DER, generative replay and newer methods are not claimed as tested here.', '',
              'References: [DER++](https://papers.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html), '
              '[ER-ACE](https://openreview.net/pdf?id=N8MaByOzUfb), '
              '[official Mammoth repository](https://github.com/aimagelab/mammoth).', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(2)
    configs = {'derpp':[{'lr':lr,'alpha':a,'beta':b} for lr in [.0005,.001,.003] for a,b in [(.1,.5),(.5,.5),(.5,1.)]],
               'er_ace':[{'lr':lr,'task_free':True,'classifier':'official_cosine'} for lr in [.0005,.001,.003]]}
    revision = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    protocol = {'configurations':configs,'capacities':[200,1000],'tuning_seed':113,'seeds':[101,102,103],
                'orders':common.ORDERS,'base':str(args.base),'prior':str(args.prior),
                'upstream_revision':revision,'algorithm_reference':'Mammoth objective audit + verbatim official cosine classifier; local matched runner.',
                'scope':'Strong established comparators under a common resource protocol, not complete current-SOTA coverage or original-paper reproduction.',
                'hyperparameter_selection':'Maximum final validation accuracy; listed-order tie breaking; selections saved before official test evaluation.',
                'buffer':'Uniform random-priority reservoir, raw uint8 images/labels, FP32 logits for DER++, float64 priorities. Independent write/sample RNG streams. Without-replacement replay draws.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    hashes = {}
    for name in ['derpp.py','er_ace.py']:
        content = (UPSTREAM/'models'/name).read_bytes()
        (out/f'official_{name}').write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    (out/'MAMMOTH_LICENSE').write_bytes((UPSTREAM/'LICENSE').read_bytes())
    (out/'upstream.json').write_text(json.dumps({'revision':revision,'sha256':hashes},indent=2))
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
    for method in configs:
        for capacity in [200,1000]:
            key = f'{method}_{capacity}'
            if key in selected:
                continue
            for config in configs[method]:
                if any(r['method']==method and r['capacity']==capacity and r['config']==config for r in trials):
                    continue
                model,mapping,stats,_ = fit(method,capacity,config,113,common.ORDERS['forward'],x,y,vx,vy,indices)
                before = common.state_hash(model)
                pred = predict(model,vx,mapping)
                assert common.state_hash(model)==before
                accuracy = float((pred==vy).float().mean())
                trials.append({'method':method,'capacity':capacity,'config':config,'validation_accuracy':accuracy,
                               'training_seconds':stats['training_seconds']})
                (out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
                print('TUNE',method,capacity,config,round(100*accuracy,2),flush=True)
            selected[key] = max((r for r in trials if r['method']==method and r['capacity']==capacity),key=lambda r:r['validation_accuracy'])['config']
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    rows = json.loads((out/'results.json').read_text()) if args.resume and (out/'results.json').exists() else []
    for method in configs:
        for capacity in [200,1000]:
            config = selected[f'{method}_{capacity}']
            for seed in [101,102,103]:
                for name,order in common.ORDERS.items():
                    if any(r['method']==method and r['capacity']==capacity and r['seed']==seed and r['order']==name for r in rows):
                        continue
                    model,mapping,stats,state = fit(method,capacity,config,seed,order,x,y,vx,vy,indices)
                    before = common.state_hash(model)
                    pred = predict(model,tx,mapping)
                    assert common.state_hash(model)==before
                    row = {'method':method,'capacity':capacity,'seed':seed,'order':name,'config':config,
                           'scores':common.metrics(pred,ty,order),**stats}
                    rows.append(row)
                    torch.save(state,out/f'{method}_{capacity}_{seed}_{name}.pt')
                    (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                    write_report(out,rows,args.prior)
                    print('FINAL',method,capacity,seed,name,round(100*row['scores']['accuracy'],2),stats['persistent_numeric_bytes'],flush=True)
    print('Completed:',out,flush=True)


if __name__=='__main__':
    main()
