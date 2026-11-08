"""Ceiling for image-only task identification on split CIFAR-10 (5 tasks of 2 classes).

Non-continual: the small tanh CNN from metaplastic_control (159k params) is trained jointly on
all tasks' data at once. target=task: 5-way task-ID head. target=class: 10-way head; task-ID read
as argmax class // 2. Optional flip+crop augmentation. Reports val/test each epoch.
Any image-only route picker trained without old data is bounded by this (same architecture).
"""
import json, argparse, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from bdh_cl.data import cifar10
from bdh_cl.scripts.metaplastic_control import CNN


def augment(x, g):
    flip = torch.rand(len(x), generator=g) < .5; x = torch.where(flip[:, None, None, None], x.flip(3), x)
    p = F.pad(x.float(), (4, 4, 4, 4), mode='reflect'); i, j = torch.randint(0, 9, (2,), generator=g)
    return p[:, :, i:i + 32, j:j + 32]


@torch.no_grad()
def evaluate(m, x, y):
    m.eval(); lg = torch.cat([m(x[j:j + 1000].float()) for j in range(0, len(x), 1000)]); m.train(); return lg


def main():
    p = argparse.ArgumentParser(); p.add_argument('--target', default='task'); p.add_argument('--aug', type=int, default=0)
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--seed', type=int, default=201)
    p.add_argument('--out', type=Path, default=Path('results/taskid_ceiling_20261107')); a = p.parse_args()
    torch.set_num_threads(2); torch.set_flush_denormal(True); a.out.mkdir(parents=True, exist_ok=True)
    d = cifar10.load('data_cache/cifar10')
    x, y = torch.from_numpy(d['images'][d['train']].copy()), torch.from_numpy(d['labels'][d['train']].copy()).long()
    vx, vy = torch.from_numpy(d['images'][d['validation']].copy()), torch.from_numpy(d['labels'][d['validation']].copy()).long()
    tx, ty = torch.from_numpy(d['test_images'].copy()), torch.from_numpy(d['test_labels'].copy()).long()
    m = CNN(a.seed)
    if a.target == 'task': m.head = torch.nn.Linear(32, 5)
    opt = torch.optim.Adam(m.parameters(), lr=.0005); g = torch.Generator().manual_seed(a.seed); hist = []
    name = f'{a.target}_aug{a.aug}_s{a.seed}'
    for ep in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        for ids in torch.randperm(len(x), generator=g).split(128):
            xb = augment(x[ids], g) if a.aug else x[ids]; tgt = y[ids] // 2 if a.target == 'task' else y[ids]
            opt.zero_grad(set_to_none=True); F.cross_entropy(m(xb), tgt).backward(); opt.step()
        row = dict(epoch=ep)
        for split, (ex, ey) in dict(val=(vx, vy), test=(tx, ty)).items():
            lg = evaluate(m, ex, ey)
            if a.target == 'task': row[f'{split}_task'] = float((lg.argmax(1) == ey // 2).float().mean())
            else:
                row[f'{split}_task'] = float((lg.argmax(1) // 2 == ey // 2).float().mean()); row[f'{split}_class'] = float((lg.argmax(1) == ey).float().mean())
        hist.append(row); (a.out / f'{name}.json').write_text(json.dumps(hist, indent=1))
        print(name, ep, {k: round(100 * v, 2) for k, v in row.items() if k != 'epoch'}, f'{time.perf_counter() - t0:.0f}s', flush=True)
    import os; os._exit(0)


if __name__ == '__main__':
    main()
