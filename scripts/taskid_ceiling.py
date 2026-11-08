"""Ceiling for single-image task identification on Split CIFAR-10 (not continual).

The ordinary CNN (same as the plain baseline) is trained jointly on ALL tasks at once.
target=task: 5-way task head. target=class: 10-way head, task read as class // 2.
Any image-only route picker with this network, trained without old data, is bounded by this.

  python scripts/taskid_ceiling.py --target class
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.nn import functional as F

warnings.filterwarnings('ignore', message='The value of the smallest subnormal')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from branch_routing import data as D  # noqa: E402
from branch_routing.continual import predict_logits  # noqa: E402
from branch_routing.models import CNN  # noqa: E402


def augment(x, g):
    flip = torch.rand(len(x), generator=g) < .5; x = torch.where(flip[:, None, None, None], x.flip(3), x)
    p = F.pad(x.float(), (4, 4, 4, 4), mode='reflect'); i, j = torch.randint(0, 9, (2,), generator=g)
    return p[:, :, i:i + 32, j:j + 32]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--target', choices=['task', 'class'], default='class')
    p.add_argument('--aug', action='store_true'); p.add_argument('--epochs', type=int, default=20); p.add_argument('--seed', type=int, default=201)
    p.add_argument('--out', type=Path, default=Path('results/taskid_ceiling')); a = p.parse_args()
    torch.set_num_threads(2); torch.set_flush_denormal(True); a.out.mkdir(parents=True, exist_ok=True)
    data = D.load_cifar10(); model = CNN(a.seed, n_out=5 if a.target == 'task' else 10)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4); g = torch.Generator().manual_seed(a.seed); hist = []
    name = f"{a.target}{'_aug' if a.aug else ''}_s{a.seed}"
    for epoch in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        for ids in torch.randperm(len(data.x), generator=g).split(128):
            xb = augment(data.x[ids], g) if a.aug else data.x[ids]; yb = D.task_of(data.y[ids]) if a.target == 'task' else data.y[ids]
            opt.zero_grad(set_to_none=True); F.cross_entropy(model(xb), yb).backward(); opt.step()
        row = dict(epoch=epoch)
        for split in ['val', 'test']:
            x, y = data.split(split); pred = predict_logits(model, x, None).argmax(1)
            row[f'{split}_task'] = float(((pred if a.target == 'task' else pred // 2) == D.task_of(y)).float().mean())
            if a.target == 'class': row[f'{split}_cil'] = float((pred == y).float().mean())
        hist.append(row); (a.out / f'{name}.json').write_text(json.dumps(hist, indent=1))
        print(name, epoch, {k: round(100 * v, 2) for k, v in row.items() if k != 'epoch'}, f'{time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
