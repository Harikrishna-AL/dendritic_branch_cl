"""Local frozen-BDH Split MNIST pilot. No replay or future-class tuning."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bdh_cl.config import preset, as_dict
from bdh_cl.models.bdh_wrapper import BDHWrapper
from bdh_cl.data.mnist import build_pools, render
from bdh_cl.cl.tasks import MNISTCLTask
from bdh_cl.data.bytes import collate
from bdh_cl.cl.runner import make_optimizer, train_one_task, git_hash
from bdh_cl.scripts.run_cl import pick_device


def balanced_indices(labels, classes, n, rng, exclude=()):
    choices = []
    for c in classes:
        ix = np.flatnonzero(labels == c)
        ix = ix[~np.isin(ix, exclude)]
        choices.extend(rng.choice(ix, n // len(classes), replace=False).tolist())
    return rng.permutation(choices)


@torch.no_grad()
def extract(model, pool, indices, device, batch_size, test=False):
    images = pool.test_images if test else pool.train_images
    labels = pool.test_labels if test else pool.train_labels
    output = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start + batch_size]
        # Feed prompt only: the answer byte is never supplied to the encoder.
        prompts = [render(images[i], int(labels[i]), pool.tag).prompt for i in ix]
        x = torch.tensor([list(p.encode('ascii')) for p in prompts], device=device)
        hidden = model(x, None, return_hidden=True)[2]
        output.append(F.normalize(hidden[:, -1].float(), dim=-1).cpu())
    return torch.cat(output), torch.tensor(labels[indices].astype(np.int64))


def fit_sgd(head, x, y, lr, epochs, seed):
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=gen)
        for ix in order.split(32):
            opt.zero_grad()
            F.cross_entropy(head(x[ix]), y[ix]).backward()
            opt.step()


def score(pred, y):
    return float((pred == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default='bdh_cl/results/local_frozen_mnist_seed0')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = pick_device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    side, steps, ntrain, ntest, nval = (7, 10, 32, 20, 20) if args.smoke else (14, 500, 1000, 200, 200)
    bs = 8
    bspec, _, tspec = preset('tiny')
    tspec = dataclasses.replace(tspec, steps_per_task=steps, batch_size=bs,
                                warmup_steps=min(50, steps // 5), eval_every=100,
                                log_every=100, block_size=side*side+4, seed=0)
    pools = build_pools(ROOT / 'data_cache/mnist', side=side, task_tags=False)
    valix = balanced_indices(pools[0].train_labels, pools[0].classes, nval, rng)
    keep = np.ones(len(pools[0].train_labels), dtype=bool)
    keep[valix] = False
    trainpool = dataclasses.replace(pools[0], train_images=pools[0].train_images[keep],
                                    train_labels=pools[0].train_labels[keep])
    task = MNISTCLTask(trainpool, side*side+4, n_eval=ntest, batch_size=bs)
    model = BDHWrapper(bspec).to(device)
    t0 = time.perf_counter()
    print(f'device={device}, side={side}, parameters={sum(p.numel() for p in model.parameters())}', flush=True)
    opt = make_optimizer(model, tspec)
    rec = train_one_task(model, task, opt, tspec, device, np.random.default_rng(0),
                         log_prefix='encoder/01 ', verbose=True)
    del opt
    model.eval()
    model.requires_grad_(False)
    torch.save({'model': model.state_dict(), 'spec': as_dict(bspec)}, out / 'frozen_encoder.pt')
    print('Encoder frozen; extracting first-pair validation features.', flush=True)
    vx, vy = extract(model, pools[0], valix, device, bs)
    trainix = balanced_indices(pools[0].train_labels, pools[0].classes, ntrain, rng, valix)
    firstx, firsty = extract(model, pools[0], trainix, device, bs)
    # Float64 statistics/solves on CPU; no explicit matrix inverse.
    d = firstx.shape[1]
    g0 = firstx.double().T @ firstx.double()
    b0 = firstx.double().T @ F.one_hot(firsty, 10).double()
    eye = torch.eye(d, dtype=torch.float64)
    alphas = [.01, .1, 1., 10., 100.]
    ridge_val = {a: score((vx.double() @ torch.linalg.solve(g0+a*eye, b0))[:, :2].argmax(1), vy) for a in alphas}
    alpha = max(alphas, key=lambda a: (ridge_val[a], -a))
    epochs = 10
    init = torch.nn.Linear(d, 10).state_dict()
    lr_val = {}
    for lr in [.01, .1, 1.]:
        h = torch.nn.Linear(d, 10)
        h.load_state_dict(init)
        fit_sgd(h, firstx, firsty, lr, epochs, 0)
        lr_val[lr] = score(h(vx)[:, :2].argmax(1), vy)
    lr = max(lr_val, key=lambda a: (lr_val[a], -a))
    head = torch.nn.Linear(d, 10)
    head.load_state_dict(init)
    G = torch.zeros_like(g0)
    B = torch.zeros_like(b0)
    sums = torch.zeros(10, d)
    counts = torch.zeros(10)
    evalsets, rows = [], []
    report = {'seed': 0, 'git_hash': git_hash(), 'device': str(device), 'side': side,
              'encoder_spec': as_dict(bspec), 'encoder_train_spec': as_dict(tspec),
              'encoder_training': rec, 'train_per_pair': ntrain, 'test_per_pair': ntest,
              'validation': {'source': 'held-out training digits 0/1; excluded from encoder and head training',
                             'n': nval, 'ridge_scores': ridge_val, 'sgd_scores': lr_val},
              'ridge_alpha': alpha, 'sgd_lr': lr, 'sgd_epochs_per_pair': epochs,
              'feature_normalization': 'L2', 'state_reset': 'independent prompt per image',
              'rows': rows}
    print(f'Validation: ridge={ridge_val}, SGD={lr_val}; selected alpha={alpha}, lr={lr}', flush=True)
    for stage, pool in enumerate(pools):
        st = time.perf_counter()
        if stage == 0:
            x, y = firstx, firsty
        else:
            ix = balanced_indices(pool.train_labels, pool.classes, ntrain, rng)
            x, y = extract(model, pool, ix, device, bs)
        exix = balanced_indices(pool.test_labels, pool.classes, ntest, rng)
        evalsets.append(extract(model, pool, exix, device, bs, test=True))
        G += x.double().T @ x.double()
        B += x.double().T @ F.one_hot(y, 10).double()
        for c in pool.classes:
            sums[c] += x[y == c].sum(0)
            counts[c] += (y == c).sum()
        fit_sgd(head, x, y, lr, epochs, stage)
        C = torch.linalg.solve(G + alpha*eye, B)
        seen = (stage+1)*2
        P = F.normalize(sums[:seen] / counts[:seen, None], dim=-1)
        metrics = {}
        for name in ['sgd', 'ridge', 'prototype']:
            per = []
            for ex, ey in evalsets:
                if name == 'sgd':
                    pred = head(ex)[:, :seen].argmax(1)
                elif name == 'ridge':
                    pred = (ex.double() @ C)[:, :seen].argmax(1)
                else:
                    pred = (ex @ P.T).argmax(1)
                per.append(score(pred, ey))
            metrics[name] = {'per_pair': per, 'all_seen': float(np.mean(per)),
                             'old': float(np.mean(per[:-1])) if stage else None, 'new': per[-1]}
        rows.append({'stage': stage, 'classes_seen': list(range(seen)), 'metrics': metrics,
                     'elapsed_seconds': time.perf_counter()-st})
        report['elapsed_seconds'] = time.perf_counter()-t0
        (out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f'After {pool.name}: {json.dumps(metrics)} ({time.perf_counter()-st:.1f}s)', flush=True)
        # Persist only aggregate classifier state, never past example features.
        torch.save({'G': G, 'B': B, 'alpha': alpha, 'sgd': head.state_dict(),
                    'prototype_sums': sums, 'prototype_counts': counts}, out / 'classifiers.pt')
    report['statistics_bytes_float64'] = G.numel()*8 + B.numel()*8
    (out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Finished in {report["elapsed_seconds"]:.1f}s. Results: {out}', flush=True)


if __name__ == '__main__':
    main()
