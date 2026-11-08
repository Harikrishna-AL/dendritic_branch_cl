"""Run upstream Numenta DendriticMLP on Split MNIST, not Permuted MNIST.

Uses pinned, unmodified upstream model/activations; a local experimental runner.
Protocol changes and task-information privileges are recorded explicitly.
"""
import argparse
import hashlib
import importlib.util
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT/'external/nupic.research'
TORCH = ROOT/'external/nupic.torch'
for path in [TORCH, RESEARCH/'src', RESEARCH/'packages/dendrites/src']:
    sys.path.insert(0, str(path))

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.torch.modules import rezero_weights
from bdh_cl.data.mnist import SPLITS, load_raw
from bdh_cl.scripts.branch_associative_retrieval import responses, predict
from bdh_cl.scripts.split_mnist_usefulness import make

spec = importlib.util.spec_from_file_location('upstream_prototype_context', RESEARCH/'packages/dendrites/src/nupic/research/frameworks/dendrites/mixins/prototype_context.py')
prototype_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prototype_module)


def state_hash(model):
    h = hashlib.sha256()
    for v in model.state_dict().values():
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def score(pred, y):
    return {'accuracy':float((pred==y).mean()), 'old_accuracy':float((pred[y<8]==y[y<8]).mean()),
            'pair_accuracy':[float((pred[y//2==k]==y[y//2==k]).mean()) if (y//2==k).any() else None for k in range(5)]}


@torch.no_grad()
def evaluate(model, tx, ty, contexts, mode, seed):
    model.eval()
    before = state_hash(model)
    rows = {}
    rng = np.random.default_rng(seed)
    inferred = torch.cdist(tx, contexts).argmin(1)
    # The oracle and random context variants are evaluation diagnostics only.
    names = ['autonomous','oracle_context','random_context'] if mode=='prototype' else ['autonomous']
    for name in names:
        preds = []
        context_ids = inferred if name=='autonomous' else (torch.tensor(ty//2) if name=='oracle_context' else torch.tensor(rng.integers(len(contexts),size=len(ty))))
        for start in range(0,len(tx),256):
            bx = tx[start:start+256]
            if mode=='input':
                cx = bx
            elif name=='autonomous':
                cx = prototype_module.infer_prototype(contexts)(bx)
            else:
                cx = contexts[context_ids[start:start+256]]
            preds.append(model(bx,cx).argmax(1))
        rows[name] = score(torch.cat(preds).numpy(),ty)
    assert state_hash(model)==before, 'Evaluation mutated model state'
    return rows, float((inferred.numpy()==ty//2).mean())


@torch.no_grad()
def existing_branch_scores(seed, tx_raw, ty, centers_raw, route_records):
    model = make('local','hard')
    model.load_state_dict(torch.load(ROOT/f'results/split_mnist_compute_comparison_20261013/{seed}_local.pt',weights_only=True))
    routes = np.array(next(r for r in route_records if r['seed']==seed and r['arm']=='local')['routes'])
    assignment = torch.cdist(tx_raw,centers_raw).argmin(1).numpy()
    b = responses(model,tx_raw)
    pred = predict(b,routes[assignment],model.head.weight.numpy(),model.head.bias.numpy())
    return score(pred,ty)


def write_report(out, rows, branch, context_acc, protocol):
    lines = ['# Actual Numenta model: Split MNIST comparison', '',
             '**Upstream DendriticMLP and KWinners are imported unmodified from pinned Numenta repositories.** '
             'This adapts their architecture to Split MNIST; it is not a reproduction of their Permuted MNIST results.', '',
             '| Model / context | Final ten-class accuracy ± seed SD | Old digits 0–7 |', '|---|---:|---:|']
    for mode in ['prototype','input']:
        selected = [r for r in rows if r['context']==mode]
        if not selected:
            continue
        for name in selected[0]['scores']:
            values = [100*r['scores'][name]['accuracy'] for r in selected]
            old = st.mean(100*r['scores'][name]['old_accuracy'] for r in selected)
            sd = st.stdev(values) if len(values)>1 else 0
            lines.append(f'| Numenta / {mode} / {name} | {st.mean(values):.2f} ± {sd:.2f}% | {old:.2f}% |')
    values = [100*r['accuracy'] for r in branch]
    lines.append(f'| Existing branch model / same five task-mean prototype assignments | {st.mean(values):.2f} ± {st.stdev(values):.2f}% | {st.mean(100*r["old_accuracy"] for r in branch):.2f}% |')
    lines += ['', f'The shared nearest-task-mean router identifies the true digit pair on **{100*context_acc:.2f}%** of test images. '
              'All predictions use all ten logits, including the oracle-context diagnostics.', '',
              '## What is being compared', '', protocol['architecture'], '', protocol['training'], '',
              protocol['prototype_context'], '', protocol['input_context'], '', protocol['branch_control'], '',
              '## Limits', '', protocol['limitations'], '', protocol['autonomous_clustering'], '',
              'The earlier 95-prototype online routing experiment is a different acquisition protocol and should not be treated as a matched router here. '
              'This comparison fixes the dataset, class order, two-epoch example budget, batch size and optimizer rate; it does not isolate a single architectural mechanism. '
              'No claim of superiority follows without accounting for model size, activation sparsity, normalization and context information.', '',
              'All evaluation calls hash full model state before and after. Checkpoints, contexts, copied runner, upstream revision hashes, '
              'per-task learning curves, parameter/state sizes and wall times are saved alongside this report.', '',
              'Sources: [Numenta paper](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846219/full), '
              '[paper code](https://github.com/numenta/htmpapers/tree/c59a339a05478f5ae76fc6243c553ef125c0c51a/biorxiv/going_beyond_the_point_neuron), '
              '[upstream model](https://github.com/numenta/nupic.research/blob/85d9961b3a2ee0590d336baf4ead89f29fe03340/packages/dendrites/src/nupic/research/frameworks/dendrites/modules/dendritic_mlp.py).', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='34,35,36')
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    protocol = {**vars(args), 'device':'cpu', 'width':2048, 'epochs':2, 'batch_size':128,
                'architecture':'Original DendriticMLP: two 2,048-unit hidden layers; five segments per neuron; 784D context; absolute-max sigmoid dendritic gating; 50% fixed feedforward weight sparsity; 5% kWinners activity; dense dendritic weights; original modified initialization with context_percent_on=.1; ten shared output logits. No SI regularization.',
                'training':'Full 60,000 training images in five natural digit pairs; two epochs per pair; same seed-derived minibatch order as existing branch runs, Adam .001, cross-entropy only, rezero fixed-mask weights after every update. One optimizer across tasks, no gradient clipping. Matched exposure budget/rate; differs from original 5-task Permuted MNIST recipe (one epoch, batch 256, lr .0005, optimizer reset after tasks). Upstream standard MNIST normalization (x-.13062755)/.30810780 retained; existing branch uses x/255 without standardization.',
                'prototype_context':'Original task-mean method: before learning each pair, compute its full training-input mean using upstream compute_prototype. Same prototype supplied to every training sample of that pair. This knows training boundaries and makes an extra current-task feature pass, with no gradients or historical replay. During testing, upstream infer_prototype selects a context independently per image by Euclidean nearest mean; no task IDs or test labels enter autonomous prediction.',
                'input_context':'InputAsContext alternative supported by upstream mixin: each example is its own 784D context during both training and inference. No task IDs, boundary signals, historical example storage, or external prototype router supplied to this model. Training loop never resets the optimizer at boundaries.',
                'branch_control':'Existing branch checkpoints (same three seeds, full two-epoch training) are evaluated with exactly the same five prototype assignments. Each prototype selects its end-of-phase branch route saved during training. This control therefore uses training-boundary-associated routes, not the earlier boundary-free 95-slot online bank. Input normalization is an affine transform, so Euclidean nearest-mean assignments are identical.',
                'limitations':'Split MNIST is Class-IL; the original Permuted MNIST setup reuses the same ten classes across tasks. No original benchmark accuracy is transferred here. Full native model has far more parameters than the 75,610-parameter branch model; this is a model-level comparison, not a parameter-matched ablation or paper-reproduction claim. Seeds and order reused; no validation sweep or test-based tuning.',
                'autonomous_clustering':'The paper also describes statistical batch clustering with Hotelling-type tests and retained historical examples for context inference. That procedure was not found in this pinned code release and is NOT implemented or claimed here. The prototype condition uses known training boundaries; the input condition is a separate official architectural option, not that clustering algorithm.',
                'revisions':{name:subprocess.check_output(['git','-C',str(ROOT/'external'/name),'rev-parse','HEAD'],text=True).strip() for name in ['htmpapers','nupic.research','nupic.torch']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    raw = load_raw(str(ROOT/'data_cache/mnist'))
    x_raw = torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255
    tx_raw = torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    y = torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    ty = raw['test_labels'].astype('int64')
    x, tx = (x_raw-.13062755)/.30810780, (tx_raw-.13062755)/.30810780
    indices = [torch.nonzero((y==a)|(y==b),as_tuple=True)[0] for a,b in SPLITS]
    # Offline construction here only schedules the same per-current-task pass the
    # upstream runner performs; no future prototype enters a training/eval stage.
    contexts = []
    for ix in indices:
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[ix],y[ix]),batch_size=256,shuffle=False)
        contexts.append(prototype_module.compute_prototype(loader))
    contexts = torch.stack(contexts)
    centers_raw = contexts*.30810780+.13062755
    assignment = torch.cdist(tx,contexts).argmin(1)
    assert torch.equal(assignment,torch.cdist(tx_raw,centers_raw).argmin(1))
    context_acc = float((assignment.numpy()==ty//2).mean())
    torch.save(contexts,out/'task_mean_contexts.pt')
    route_records = json.loads((ROOT/'results/split_mnist_compute_comparison_20261013/results.json').read_text())
    branch = [{'seed':seed,**existing_branch_scores(seed,tx_raw,ty,centers_raw,route_records)} for seed in map(int,args.seeds.split(','))]
    (out/'existing_branch_control.json').write_text(json.dumps(branch,indent=2))
    rows = []
    for seed in map(int,args.seeds.split(',')):
        for mode in ['prototype','input']:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = DendriticMLP(input_size=784,output_size=10,hidden_sizes=[2048,2048],num_segments=5,dim_context=784,
                                 kw=True,kw_percent_on=.05,dendrite_weight_sparsity=0.,weight_sparsity=.5,context_percent_on=.1)
            opt = torch.optim.Adam(model.parameters(),lr=.001)
            stages = []
            start = time.perf_counter()
            for stage,ix in enumerate(indices):
                model.train()
                for epoch in range(2):
                    gen = torch.Generator().manual_seed(seed*100000+stage*100+epoch)
                    for ids in ix[torch.randperm(len(ix),generator=gen)].split(128):
                        bx, by = x[ids],y[ids]
                        cx = contexts[stage].expand(len(bx),-1) if mode=='prototype' else bx
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(model(bx,cx),by)
                        loss.backward()
                        opt.step()
                        model.apply(rezero_weights)
                # Evaluate only seen classes' examples; output head remains unrestricted.
                mask = ty//2<=stage
                stage_scores, _ = evaluate(model,tx[mask],ty[mask],contexts[:stage+1],mode,seed)
                stages.append(stage_scores['autonomous'])
                print(seed,mode,'stage',stage+1,'seen accuracy',round(100*stage_scores['autonomous']['accuracy'],2),flush=True)
            elapsed = time.perf_counter()-start
            scores, inferred_acc = evaluate(model,tx,ty,contexts,mode,seed)
            row = {'seed':seed,'context':mode,'scores':scores,'stages':stages,
                   'allocated_parameters':sum(p.numel() for p in model.parameters()),
                   'nonzero_parameters':sum(int(torch.count_nonzero(p)) for p in model.parameters()),
                   'model_state_bytes':sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                   'prototype_numeric_bytes':contexts.numel()*contexts.element_size() if mode=='prototype' else 0,
                   'training_plus_stage_eval_seconds':elapsed,'pair_inference_accuracy':inferred_acc if mode=='prototype' else None}
            rows.append(row)
            torch.save(model.state_dict(),out/f'{seed}_{mode}.pt')
            (out/'results.json').write_text(json.dumps(rows,indent=2))
            write_report(out,rows,branch,context_acc,protocol)
            print('FINAL',seed,mode,{k:round(100*v['accuracy'],2) for k,v in scores.items()},flush=True)
            del opt,model
    print('Completed:',out,flush=True)


if __name__=='__main__':
    main()
