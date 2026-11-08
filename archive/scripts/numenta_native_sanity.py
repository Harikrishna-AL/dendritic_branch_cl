"""Single-seed upstream five-task Permuted MNIST recipe sanity check."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bdh_cl.scripts.numenta_split_mnist_comparison import (
    DendriticMLP, ROOT, prototype_module, rezero_weights, state_hash,
)
from bdh_cl.data.mnist import load_raw


@torch.no_grad()
def evaluate(model, tx, ty, contexts, permutations):
    model.eval()
    before = state_hash(model)
    accuracy, context_accuracy = [], []
    for task,perm in enumerate(permutations):
        hits = route_hits = 0
        for bx,by in zip(tx[:,perm].split(256),ty.split(256)):
            ids = torch.cdist(bx,contexts).argmin(1)
            cx = prototype_module.infer_prototype(contexts)(bx)
            hits += int((model(bx,cx).argmax(1)==by).sum())
            route_hits += int((ids==task).sum())
        accuracy.append(hits/len(ty))
        context_accuracy.append(route_hits/len(ty))
    assert state_hash(model)==before
    return {'task_accuracy':accuracy,'mean_accuracy':float(np.mean(accuracy)),
            'task_context_accuracy':context_accuracy}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    seed = 34
    protocol = {'seed':seed,'tasks':5,'dataset':'Permuted MNIST, identity then four permutations from torch.Generator(seed=42), matching upstream dataset.',
                'recipe':'Unmodified upstream DendriticMLP, 2x2048 hidden, 5 segments, 50% feedforward sparsity, 5% kWinners, context_percent_on=.1; task-mean prototype context, inferred per test image. Standard MNIST normalization, Adam .0005, batch 256, 1 epoch per task, optimizer reset after each task, no clipping/SI. This is the upstream PROTOTYPE_5 recipe.',
                'scope':'One-seed harness sanity check, not an eight-seed paper reproduction. Same cached official 60,000 training / 10,000 test MNIST inputs reused across permutations. Label modulo 10, ten shared logits. Prototype preprocessing pass is current-task only.',
                'differences':'Local lightweight runner replaces the original Ray experiment scheduler. Deterministic local batch ordering rather than reproducing original sampler RNG consumption; otherwise native model and five-task hyperparameters. No test-based tuning or selection.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    raw = load_raw(str(ROOT/'data_cache/mnist'))
    x = (torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    tx = (torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    y = torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    ty = torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen = torch.Generator().manual_seed(42)
    permutations = [torch.arange(784)]+[torch.randperm(784,generator=gen) for _ in range(4)]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DendriticMLP(784,10,[2048,2048],5,784,True,kw_percent_on=.05,
                         dendrite_weight_sparsity=0.,weight_sparsity=.5,context_percent_on=.1)
    contexts, stages = [], []
    started = time.perf_counter()
    for stage,perm in enumerate(permutations):
        current = x[:,perm]
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(current,y),batch_size=256,shuffle=False)
        contexts.append(prototype_module.compute_prototype(loader))
        model.train()
        opt = torch.optim.Adam(model.parameters(),lr=.0005)
        order = torch.randperm(len(y),generator=torch.Generator().manual_seed(seed*100000+stage*100))
        for ids in order.split(256):
            bx,by = current[ids],y[ids]
            cx = contexts[-1].expand(len(ids),-1)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(bx,cx),by)
            loss.backward()
            opt.step()
            model.apply(rezero_weights)
        results = evaluate(model,tx,ty,torch.stack(contexts),permutations[:stage+1])
        stages.append(results)
        (out/'results.json').write_text(json.dumps({'stages':stages,'elapsed_seconds':time.perf_counter()-started},indent=2))
        print('PERMUTED SANITY stage',stage+1,results,flush=True)
    torch.save(model.state_dict(),out/'model.pt')
    torch.save({'contexts':torch.stack(contexts),'permutations':torch.stack(permutations)},out/'context_and_permutations.pt')
    report = ['# Native Numenta Permuted MNIST sanity check','',
              f'Five-task final mean accuracy: **{100*stages[-1]["mean_accuracy"]:.2f}%**, seed 34.', '',
              '| Task | Final digit accuracy | Context inference accuracy |','|---|---:|---:|']
    for i,(a,c) in enumerate(zip(stages[-1]['task_accuracy'],stages[-1]['task_context_accuracy'])):
        report.append(f'| {i+1} | {100*a:.2f}% | {100*c:.2f}% |')
    report += ['',protocol['recipe'],'',protocol['scope'],'',protocol['differences'],'',
               'This verifies behavior on the original benchmark family. It must not be numerically compared against Split MNIST as if the protocols were identical. '
               'Full per-task learning curves, protocol, source, final checkpoint and contexts are saved. Evaluation verifies unchanged model state.','']
    (out/'REPORT.md').write_text('\n'.join(report))
    print('Completed sanity:',out,flush=True)


if __name__=='__main__':
    main()
