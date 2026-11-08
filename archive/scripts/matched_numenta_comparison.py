"""Validation-selected, matched-free-parameter Numenta/branch comparison.

Known training boundaries and common task-mean context inference for all arms.
No benchmark-wide superiority or exact paper-reproduction claim is intended.
"""
import argparse
import hashlib
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bdh_cl.scripts.numenta_split_mnist_comparison import (
    DendriticMLP, ROOT, prototype_module, rezero_weights, state_hash,
)
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier
from bdh_cl.data.mnist import SPLITS, load_raw

ARMS = ['numenta', 'branch_aux', 'branch_ce']
ORDERS = {'forward':list(range(5)), 'reverse':list(range(4,-1,-1))}


def construct(arm, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if arm=='numenta':
        return DendriticMLP(784,10,[64,64],5,784,True,kw_percent_on=.05,
                            weight_sparsity=.5,dendrite_weight_sparsity=0.,context_percent_on=.1)
    return BranchClassifier('local',neurons=168,branches=4,gate='hard')


def accounting(model):
    total = sum(p.numel() for p in model.parameters())
    fixed_zeros = sum(int(torch.count_nonzero(b)) for name,b in model.named_buffers() if name.endswith('zero_mask'))
    return {'allocated_parameters':total, 'free_coefficients':total-fixed_zeros,
            'fixed_zero_coefficients':fixed_zeros,
            'model_state_bytes':sum(v.numel()*v.element_size() for v in model.state_dict().values())}


@torch.no_grad()
def predictions(model, arm, x, contexts, routes, mode='inferred', pair_ids=None):
    model.eval()
    result = []
    for start in range(0,len(x),256):
        bx = x[start:start+256]
        if mode=='current':
            result.append(model(bx).argmax(1))
            continue
        ids = pair_ids[start:start+256] if mode=='oracle' else torch.cdist(bx,contexts).argmin(1)
        if arm=='numenta':
            cx = contexts[ids] if mode=='oracle' else prototype_module.infer_prototype(contexts)(bx)
            logits = model(bx,cx)
        else:
            z = torch.tanh(model.first(bx).reshape(-1,168,4))
            selected = routes[ids].long()
            hidden = z.gather(2,selected[:,:,None]).squeeze(2)
            logits = model.head(hidden)
        result.append(logits.argmax(1))
    return torch.cat(result)


def metrics(pred, y, order):
    old = torch.zeros_like(y,dtype=torch.bool)
    for group in order[:-1]:
        old |= y//2==group
    return {'accuracy':float((pred==y).float().mean()),
            'old_accuracy':float((pred[old]==y[old]).float().mean()),
            'pair_accuracy':[float((pred[y//2==k]==y[y//2==k]).float().mean()) for k in range(5)]}


def train(arm, seed, rate, order, x, y, vx, vy, contexts, indices, verbose=False, on_stage=None):
    model = construct(arm,seed)
    opt = torch.optim.Adam(model.parameters(),lr=rate)
    routes = torch.zeros((5,168),dtype=torch.uint8)
    stages = []
    seconds = 0.
    clipped = steps = 0
    for stage,group in enumerate(order):
        model.train()
        ix = indices[group]
        for epoch in range(2):
            gen = torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ids in ix[torch.randperm(len(ix),generator=gen)].split(128):
                begin = time.perf_counter()
                bx,by = x[ids],y[ids]
                opt.zero_grad(set_to_none=True)
                if arm=='numenta':
                    loss = F.cross_entropy(model(bx,contexts[group].expand(len(bx),-1)),by)
                    feedback = None
                elif arm=='branch_aux':
                    logits,candidates = model(bx,counterfactual=True)
                    targets = by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                    errors = F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1])
                    feedback = errors.detach().mean(0)
                    loss = F.cross_entropy(logits,by)+.5*errors.mean()
                else:
                    # Same feedback as the auxiliary arm, without a redundant
                    # counterfactual backward graph in the CE-only control.
                    with torch.no_grad():
                        _,candidates = model(bx,counterfactual=True)
                        targets = by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                        errors = F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1])
                        feedback = errors.mean(0)
                    loss = F.cross_entropy(model(bx),by)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                clipped += int(norm>5)
                opt.step()
                if feedback is not None:
                    model.observe(feedback)
                else:
                    model.apply(rezero_weights)
                steps += 1
                seconds += time.perf_counter()-begin
        if arm!='numenta':
            routes[group] = model.gates().argmax(-1).to(torch.uint8)
        # Validation only during acquisition; test data is never used here.
        mask = vy//2==group
        before = state_hash(model)
        pred = predictions(model,arm,vx[mask],contexts,routes,'oracle',vy[mask]//2)
        assert state_hash(model)==before
        stages.append({'group':group,'validation_oracle_acquisition':float((pred==vy[mask]).float().mean())})
        if on_stage is not None:
            on_stage(model, stage)
        if verbose:
            print('stage',seed,arm,stage+1,'acquisition',round(100*stages[-1]['validation_oracle_acquisition'],2),flush=True)
    opt_bytes = sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v))
    return model,routes,{'training_seconds_excluding_validation':seconds,'steps':steps,
                        'clipped_steps':clipped,'optimizer_state_bytes':opt_bytes,'stages':stages}


def evaluate(model,arm,x,y,contexts,routes,order):
    before = state_hash(model)
    before_routes = routes.clone()
    scores = {}
    for mode in ['inferred','oracle'] + (['current'] if arm!='numenta' else []):
        pred = predictions(model,arm,x,contexts,routes,mode,y//2)
        scores[mode] = metrics(pred,y,order)
    assert state_hash(model)==before and torch.equal(routes,before_routes)
    return scores


def latency(model,arm,x,contexts,routes):
    # Includes the common nearest-mean router; CPU batch 256, paired same images.
    bx = x[:256]
    for _ in range(3):
        predictions(model,arm,bx,contexts,routes)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        predictions(model,arm,bx,contexts,routes)
        times.append((time.perf_counter()-start)*1000)
    return float(np.median(times))


def write_report(out,records,selection,sizes,protocol):
    lines = ['# Matched-size Numenta comparison','',
             '**Primary budget: free learnable coefficients after fixed masks. All methods are retrained under a common protocol.**','',
             '| Model | Free coefficients | Allocated parameters | Selected learning rate |', '|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f'| {arm} | {sizes[arm]["free_coefficients"]:,} | {sizes[arm]["allocated_parameters"]:,} | {selection[arm]} |')
    lines += ['', '| Model | Inferred-context accuracy ± seed SD | Oracle-context accuracy | Old-class accuracy |', '|---|---:|---:|---:|']
    for arm in ARMS:
        rows = [r for r in records if r['arm']==arm]
        if not rows:
            continue
        # Orders are paired within each seed, never counted as independent seeds.
        seeds = sorted({r['seed'] for r in rows})
        vals = [st.mean(100*r['scores']['inferred']['accuracy'] for r in rows if r['seed']==s) for s in seeds]
        sd = st.stdev(vals) if len(vals)>1 else 0
        oracle = st.mean(100*r['scores']['oracle']['accuracy'] for r in rows)
        old = st.mean(100*r['scores']['inferred']['old_accuracy'] for r in rows)
        lines.append(f'| {arm} | {st.mean(vals):.2f} ± {sd:.2f}% | {oracle:.2f}% | {old:.2f}% |')
    lines += ['', '## Fairness controls','', protocol['matching'],'',protocol['data'],'',protocol['selection'],'',
              protocol['context'],'',protocol['training'],'',
              '## Interpretation limits','',protocol['limits'],'',
              'Primary results average forward/reverse orders within each seed before computing SD. '
              'Task-mean context acquisition uses known training boundaries in every arm. '
              'Oracle context is only a diagnostic; no output label mask is applied. '
              'The CE-only branch arm retains the same error-trace feedback and archives, so it isolates the auxiliary-loss effect.', '',
              'Exact per-order results, allocation and persistent-state bytes, optimizer state, clipped-gradient counts, acquisition diagnostics, '
              'training times and batch-256 inference latencies are in results.json. Training seconds exclude validation, prototype preparation and output saving. '
              'These are wall-clock measurements on this CPU environment, not FLOP estimates or M4 benchmarks.', '',
              'Evaluation asserts that model tensors and route archives remain unchanged. Official test data is evaluated only after validation selection is frozen. '
              'Validation indices, common prototype contexts, protocol, source, chosen hyperparameters and final checkpoints are saved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    protocol = {'seeds':[101,102,103],'tuning_seed':113,'orders':ORDERS,'learning_rates':[.0005,.001,.003],
                'matching':'Numenta: original upstream two-layer architecture scaled to [64,64], five 784D segments per neuron, 50% feedforward weight masks, original 5% kWinners (3 winners/layer). Branch: 168 neurons × 4 tanh branches. Approximately 529k free coefficients each, within 0.03%; allocated dense parameter counts differ by about 5.2%. Masks, buffers and route archives are separately counted. No claim of equal active units, depth, executed FLOPs or memory.',
                'data':'Split MNIST, 500 held-out training images per digit (5,000 validation) selected with seed 20261101; remaining 55,000 for all training runs. Official 10,000 test images never used for tuning. Identical standard normalization (x/255-.13062755)/.30810780 for all methods. Forward and reverse pair orders; matched seed/stage/epoch minibatch permutations across methods.',
                'selection':'Exactly three learning-rate candidates (.0005,.001,.003) per method, one separate calibration seed (113), forward order, two epochs per task. Select highest final unrestricted ten-class validation accuracy with inferred context; ties select smaller rate. Freeze before evaluating test data. Same small validation budget, not an exhaustive hyperparameter optimization or equal-FLOP tuning budget.',
                'context':'Five task-mean prototypes from the same 55,000 training inputs. Each method is allowed the same training boundaries and current-task prototype pass. Numenta uses current-task mean during learning; branch arms archive their current route at each boundary. Inference independently selects nearest prototype by Euclidean distance, then supplies its vector to Numenta or its associated route to branch arms. No test IDs, label masks, replay or historical weight checkpoints. Extra archived branch routes and traces count toward persistent state.',
                'training':'Two epochs per pair, batch 128, Adam, one optimizer across tasks, gradient clip 5 for every method; learning rate independently validation-selected. Numenta trains ordinary CE, retains native initialization and kWinners, and rezeros masked weights after updates. Branch auxiliary objective is CE+.5 mean single-neuron counterfactual CE; CE-only arm drops auxiliary gradients but retains identical counterfactual error-trace updates. No SI in any arm.',
                'limits':'One matched coefficient budget, three final seeds, two related task orders, and three LR trials per arm. Different depth, activation sparsity, initialization and model structure remain inherent method differences. Fixed task-mean routing is boundary-aware during training, not the paper\'s autonomous statistical clustering. This is a stronger controlled Split MNIST comparison, not an exact paper reproduction, proof of general superiority, or evidence that the auxiliary loss uniquely causes an advantage.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    sizes = {arm:accounting(construct(arm,0)) for arm in ARMS}
    assert abs(sizes['numenta']['free_coefficients']/sizes['branch_aux']['free_coefficients']-1)<.001
    (out/'sizes.json').write_text(json.dumps(sizes,indent=2))
    raw = load_raw(str(ROOT/'data_cache/mnist'))
    x = (torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    y = torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx = (torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    ty = torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng = np.random.default_rng(20261101)
    val_ids = np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==c))[:500] for c in range(10)])
    train_ids = np.setdiff1d(np.arange(len(y)),val_ids)
    np.savez(out/'split_indices.npz',train=train_ids,validation=val_ids)
    vx,vy = x[val_ids],y[val_ids]
    x,y = x[train_ids],y[train_ids]
    indices = [torch.nonzero(y//2==k,as_tuple=True)[0] for k in range(5)]
    contexts = torch.stack([x[ix].mean(0) for ix in indices])
    torch.save(contexts,out/'contexts.pt')
    trials,selected = [],{}
    for arm in ARMS:
        for rate in protocol['learning_rates']:
            model,routes,trace = train(arm,113,rate,ORDERS['forward'],x,y,vx,vy,contexts,indices)
            before = state_hash(model)
            pred = predictions(model,arm,vx,contexts,routes)
            assert state_hash(model)==before
            value = float((pred==vy).float().mean())
            trials.append({'arm':arm,'rate':rate,'validation_accuracy':value,**trace})
            (out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
            print('TUNE',arm,rate,round(100*value,2),flush=True)
            del model
        best = max((r for r in trials if r['arm']==arm),key=lambda r:(r['validation_accuracy'],-r['rate']))
        selected[arm] = best['rate']
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    # The selection is persisted before the first official test prediction.
    records = []
    for seed in protocol['seeds']:
        for name,order in ORDERS.items():
            for arm in ARMS:
                model,routes,trace = train(arm,seed,selected[arm],order,x,y,vx,vy,contexts,indices,True)
                scores = evaluate(model,arm,tx,ty,contexts,routes,order)
                state_bytes = sizes[arm]['model_state_bytes']+contexts.numel()*contexts.element_size()+(routes.numel()*routes.element_size() if arm!='numenta' else 0)
                record = {'seed':seed,'order':name,'arm':arm,'learning_rate':selected[arm],
                          'scores':scores,**trace,**sizes[arm], 'persistent_numeric_bytes':state_bytes,
                          'inference_batch256_median_ms':latency(model,arm,tx,contexts,routes)}
                records.append(record)
                torch.save({'model':model.state_dict(),'routes':routes if arm!='numenta' else None},out/f'{seed}_{name}_{arm}.pt')
                (out/'results.json').write_text(json.dumps(records,indent=2,allow_nan=False))
                write_report(out,records,selected,sizes,protocol)
                print('FINAL',seed,name,arm,{k:round(100*v['accuracy'],2) for k,v in scores.items()},flush=True)
                del model
    print('Completed:',out,flush=True)


if __name__=='__main__':
    main()
